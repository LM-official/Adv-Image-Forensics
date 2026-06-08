import argparse
import gc
from   config       import (DESCRIPTION_MODEL, DESCRIPTION_BATCH_SIZE, MIN_DESCRIPTION_TOKENS, MAX_DESCRIPTION_TOKENS,
                            DESCRIPTION_NUM_BEAMS, DESCRIPTION_REPETITION_PENALTY, DESCRIPTION_NO_REPEAT_NGRAM,
                            DESCRIPTION_DO_SAMPLE, DESCRIPTION_TEMPERATURE, DESCRIPTION_TOP_P,
                            EMPTY_MASK_PIXEL_RATIO, EMPTY_MASK_LABEL, DESCRIPTION_DEDUP_WORDS,
                            MASKED_DIR, FILTERED_DATASET_FILE)
import cv2
import numpy        as     np
from   pathlib      import Path
import pandas       as     pd
import re
import torch
from   transformers import BlipProcessor, BlipForConditionalGeneration
from   utils        import load_csv, update_dataset


def load_blip(model_name: str = DESCRIPTION_MODEL) -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    """
    Loads the Hugging Face BLIP processor and model for image captioning.

    Args:
        model_name: Hugging Face repo id of the BLIP captioning model (e.g. blip-image-captioning-large).

    Returns:
        A tuple (processor, model, device) ready for inference, or None if loading failed.
    """

    print(f"Loading Hugging Face model (BLIP): '{model_name}'...")
    try:
        # Check for hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia GPUs) and set the device accordingly
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 for better accuracy
        torch_dtype = torch.float32
        print(f"Using device: {device} in {torch_dtype} precision.")

        # Processor is responsible for preparing the input data (images) in the format expected by the model
        processor = BlipProcessor.from_pretrained(model_name)
        # Model is the actual neural network that generates the caption based on the processed input
        model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)

        # Move the model to the selected device for faster inference
        model.to(device)

        print("BLIP model loaded successfully.")
        return processor, model, device

    except Exception as e:
        print(f"ERROR: loading BLIP model: {e}")
        return None


def _is_empty_mask(image: np.ndarray, ratio: float) -> bool:
    """
    Tells whether a masked image is (near-)all-black.

    BLIP hallucinates on a black image so such images are detected and skipped: the caller assigns them a fixed label instead of a fake caption.

    Args:
        image: Masked BGR image as read by cv2.
        ratio: Maximum fraction of non-black pixels (any non-zero channel) for the image to count as empty.

    Returns:
        True if the non-black pixel fraction is <= ratio (effectively all black), else False.
    """

    return float(np.any(image > 0, axis=2).mean()) <= ratio


def _clean_caption(text: str, dedup_words: bool) -> str:
    """
    Forces a BLIP caption to letters only and (optionally) drops repeated words.

    Guarantees, regardless of what BLIP emitted:
    - only the letters a-z;
    - when dedup_words is set, no word appears more than once consecutivelly.

    Args:
        text:        Raw caption decoded from BLIP.
        dedup_words: If True, remove duplicate consecutive words from the caption.

    Returns:
        The cleaned caption, or EMPTY_MASK_LABEL if cleaning leaves it empty (all punctuation/digits).
    """

    # Only a-z: lowercase + replace every non a-z characters with a single space
    text  = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    words = text.split()
    # No word written multiple times consecutivelly on one image: keep the first occurrence, preserve order
    if dedup_words:
        seen  = set()
        words = [word for word in words if not (word in seen or seen.add(word))]
    cleaned = " ".join(words)
    # A caption that was entirely non a-z characters collapses to "": fall back to the fixed empty label
    return cleaned if cleaned else EMPTY_MASK_LABEL


def _free_memory(device: torch.device) -> None:
    """
    Releases the tensors left over from a batch so memory does not grow across batches.

    Python frees the per-batch locals on its own, but the CUDA/MPS allocator keeps the freed blocks
    cached; emptying that cache returns the memory to the system between batches.

    Args:
        device: Torch device the model and inputs live on.

    Returns:
        None.
    """

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _describe_images_batch(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device,
                           batch: list[Path], min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                           num_beams: int = DESCRIPTION_NUM_BEAMS, rep_penalty: float = DESCRIPTION_REPETITION_PENALTY,
                           no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM, do_sample: bool = DESCRIPTION_DO_SAMPLE,
                           temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P,
                           empty_ratio: float = EMPTY_MASK_PIXEL_RATIO, dedup_words: bool = DESCRIPTION_DEDUP_WORDS) -> list[tuple[str, str]] | None:
    """
    Passes a batch of masked images to the BLIP model to get textual descriptions of the manipulated area.

    (Near-)all-black masks are not sent to BLIP (it would hallucinate); they receive EMPTY_MASK_LABEL instead.
    The remaining images are captioned with beam search + anti-repetition decoding,
    then every caption is forced to letters only and de-duplicated (see _clean_caption).

    Args:
        processor:       BLIP processor used to prepare the images for the model.
        model:           BLIP model that generates the captions.
        device:          Torch device on which the model and inputs live (cpu, cuda or mps).
        batch:           List of paths to the masked images in the current batch.
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        do_sample:       If True, sample (temperature/top_p) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty (see _is_empty_mask).
        dedup_words:     If True, drop duplicate words from each caption (see _clean_caption).

    Returns:
        A list of (filename_stem, description) tuples, or None on failure or empty batch.
    """


    try:
        loads, names, fixed = [], [], []
        # Load all images in the batch
        for image_path in batch:
            # Force the image to be read in color mode
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"ERROR: '{image_path}' image not found or corrupted.")
                continue
            # (Near-)all-black mask: TruFor found no confident manipulation. Skip BLIP (it would hallucinate) and assign a fixed, non-empty label
            if _is_empty_mask(image, empty_ratio):
                fixed.append((image_path.stem, EMPTY_MASK_LABEL))
                continue
            # Hugging Face processors accept numpy arrays directly in RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loads.append(image)
            names.append(image_path.stem)
        # Every image was empty or corrupt, nothing to caption
        if not loads:
            return fixed or None


        # Process the entire batch of images and move the input tensors to the model's device
        inputs = processor(images=loads, return_tensors="pt").to(device)
        # Decoding: beam search + anti-repetition kills the degenerate captions; sampling adds lexical diversity
        gen = dict(min_new_tokens=min_token, max_new_tokens=max_token, num_beams=num_beams,
                   repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram, early_stopping=True)
        if do_sample:
            gen.update(do_sample=True, temperature=temperature, top_p=top_p, num_beams=1)
        # Generate the predictions
        out = model.generate(**inputs, **gen)

        # Decode the output tensors into a list of readable text strings, then enforce the a-z / no-duplicate-word rules
        batch_descriptions = processor.batch_decode(out, skip_special_tokens=True)
        cleaned = [_clean_caption(text, dedup_words) for text in batch_descriptions]
        return fixed + list(zip(names, cleaned))

    except Exception as e:
        print(f"ERROR: generating description: {e}")
        return None


def describe_images(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device,
                    batch_size: int = DESCRIPTION_BATCH_SIZE, min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                    num_beams: int = DESCRIPTION_NUM_BEAMS, rep_penalty: float = DESCRIPTION_REPETITION_PENALTY,
                    no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM, do_sample: bool = DESCRIPTION_DO_SAMPLE,
                    temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P,
                    empty_ratio: float = EMPTY_MASK_PIXEL_RATIO, dedup_words: bool = DESCRIPTION_DEDUP_WORDS,
                    masked: Path = MASKED_DIR, dataset: Path = FILTERED_DATASET_FILE,) -> None:
    """
    Iterates over all masked images in masked_dir in batches and generates textual descriptions,
    rewriting the dataset and freeing memory after each batch (so progress survives an interrupted run).

    Args:
        processor:       BLIP processor used to prepare the images for the model.
        model:           BLIP model that generates the captions.
        device:          Torch device on which the model and inputs live (cpu, cuda or mps).
        masked:          Directory containing the masked images to describe.
        dataset:         Path to the dataset CSV to merge the descriptions into.
        batch_size:      Number of images processed simultaneously per batch.
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        do_sample:       If True, sample (temperature/top_p) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty.
        dedup_words:     If True, drop duplicate words from each caption.

    Returns:
        None.
    """

    if not masked.is_dir() or not dataset.is_file() or max_token <= 0:
        print(f"ERROR: invalid masked dir '{masked}', dataset file '{dataset}' or max_token {max_token}.")
        return None


    # Get image paths, excluding hidden files
    images = [path for path in masked.iterdir() if not path.name.startswith(".")]
    if not images:
        print(f"WARNING: No valid masked images found in '{masked}'.")
        return None
    len_images = len(images)

    # Generating descriptions
    print(f"Describing masked images in folder: '{masked}'...")
    # Decoding / post-processing knobs shared by every batch
    params = (min_token, max_token, num_beams, rep_penalty, no_repeat_ngram, do_sample, temperature, top_p, empty_ratio, dedup_words)


    # No batch
    if batch_size <= 0:
        descriptions = _describe_images_batch(processor, model, device, images, *params)
        if descriptions:
            update_dataset(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
        return None


    # load the dataset once, keep it in memory, and merge the descriptions collected so far after each batch
    df = load_csv(dataset)
    if df is None:
        return None
    # Drop the stale anomaly_txt; the fresh descriptions are merged back in below
    df = df.drop(columns="anomaly_txt", errors="ignore")

    descriptions = []
    # Process images in chunks of size = batch_size: caption -> persist -> free memory, one batch at a time
    for i in range(0, len_images, batch_size):
        print(f"Processing batch {i // batch_size + 1}/{(len_images + batch_size - 1) // batch_size}...", end="\r")

        results = _describe_images_batch(processor, model, device, images[i:i + batch_size], *params)
        if results:
            descriptions.extend(results)
            # Update the dataset at the end of the batch: rewrite the whole table with every description so far
            df.merge(pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]), on="filename").dropna().to_csv(dataset, index=False, encoding="utf-8")
        else:
            print(f"WARNING: No descriptions generated for batch {i // batch_size + 1}.")

        # Release the per-batch image / inference tensors before loading the next batch
        _free_memory(device)

    print(f"\nDataset '{dataset}' updated successfully.")
    return None


def main() -> None:
    """
    Loads BLIP and captions every masked image, merging the descriptions into the dataset.

    Args:
        None (the masked dir and dataset default from config.py; the model and tuning knobs are per-run CLI flags).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Caption the masked anomaly region of every image with BLIP and merge the result into the dataset.")

    # Flags
    # Masked dir and dataset come from config.py (the project backbone); the model and tuning knobs are exposed as flags (defaulting from config)
    parser.add_argument("--model",              type=str,   default=DESCRIPTION_MODEL,              help=f"BLIP captioning model repo id. Default: {DESCRIPTION_MODEL}")
    parser.add_argument("--batch-size",         type=int,   default=DESCRIPTION_BATCH_SIZE,         help=f"Number of images captioned simultaneously. Default: {DESCRIPTION_BATCH_SIZE}")
    parser.add_argument("--min-token",          type=int,   default=MIN_DESCRIPTION_TOKENS,         help=f"Minimum number of tokens in the generated descriptions. Default: {MIN_DESCRIPTION_TOKENS}")
    parser.add_argument("--max-token",          type=int,   default=MAX_DESCRIPTION_TOKENS,         help=f"Maximum number of tokens in the generated descriptions. Default: {MAX_DESCRIPTION_TOKENS}")
    parser.add_argument("--num-beams",          type=int,   default=DESCRIPTION_NUM_BEAMS,          help=f"Beam-search width (ignored with --do-sample). Default: {DESCRIPTION_NUM_BEAMS}")
    parser.add_argument("--repetition-penalty", type=float, default=DESCRIPTION_REPETITION_PENALTY, help=f"Repetition penalty during generation. Default: {DESCRIPTION_REPETITION_PENALTY}")
    parser.add_argument("--no-repeat-ngram",    type=int,   default=DESCRIPTION_NO_REPEAT_NGRAM,    help=f"Size of the n-gram that may not repeat during generation. Default: {DESCRIPTION_NO_REPEAT_NGRAM}")
    parser.add_argument("--do-sample",          action="store_true", default=DESCRIPTION_DO_SAMPLE, help="Sample (temperature/top-p) instead of beam search for more lexical diversity.")
    parser.add_argument("--temperature",        type=float, default=DESCRIPTION_TEMPERATURE,        help=f"Sampling temperature (only with --do-sample). Default: {DESCRIPTION_TEMPERATURE}")
    parser.add_argument("--top-p",              type=float, default=DESCRIPTION_TOP_P,              help=f"Nucleus sampling probability (only with --do-sample). Default: {DESCRIPTION_TOP_P}")
    parser.add_argument("--empty-ratio",        type=float, default=EMPTY_MASK_PIXEL_RATIO,         help=f"Non-black pixel fraction at/below which a mask is 'empty' (labelled '{EMPTY_MASK_LABEL}'). Default: {EMPTY_MASK_PIXEL_RATIO}")
    # Switch: word de-duplication on (default) / off. --no-dedup-words keeps repeated words in the caption.
    parser.add_argument("--no-dedup-words",     dest="dedup_words", action="store_false",           help="Keep repeated words in the caption (disables the de-duplication post-pass).")
    parser.set_defaults(dedup_words=DESCRIPTION_DEDUP_WORDS)
    args = parser.parse_args()

    # Load the BLIP model
    blip = load_blip(args.model)
    if blip is None:
        return None
    processor, model, device = blip

    # Describe the masked images (masked dir and dataset default from config; the model and tuning knobs come from the flags)
    describe_images(processor, model, device, args.batch_size, args.min_token, args.max_token,
                    args.num_beams, args.repetition_penalty, args.no_repeat_ngram, args.do_sample,
                    args.temperature, args.top_p, args.empty_ratio, args.dedup_words)
    return None

if __name__ == "__main__":
    main()