import argparse
from   collections  import Counter
import gc
from   config       import (DESCRIPTION_MODEL, DESCRIPTION_BATCH_SIZE, MIN_DESCRIPTION_TOKENS, MAX_DESCRIPTION_TOKENS, DESCRIPTION_NUM_BEAMS,
                            DESCRIPTION_REPETITION_PENALTY, DESCRIPTION_NO_REPEAT_NGRAM, DESCRIPTION_DO_SAMPLE, DESCRIPTION_TEMPERATURE, DESCRIPTION_TOP_P,
                            DESCRIPTION_TOP_K, DESCRIPTION_SEED, DESCRIPTION_CROP, DESCRIPTION_TOP_WORDS, EMPTY_MASK_PIXEL_RATIO, EMPTY_MASK_LABEL,
                            DESCRIPTION_DEDUP_WORDS, OCR_ENABLED, OCR_CROP_PADDING, MASKED_DIR, FILTERED_DATASET_FILE,
                            DESCRIPTION_REPORT_FILE)
import cv2
import easyocr
from   ocr          import load_reader, read_text
from   pathlib      import Path
import pandas       as     pd
import torch
from   transformers import BlipProcessor, BlipForConditionalGeneration
from   utils        import clean_text, crop_to_content, is_empty_mask, load_csv, save_report, update_dataset


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


def _free_memory(device: torch.device):
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
                           batch: list[Path], reader: easyocr.Reader | None = None,
                           min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                           num_beams: int = DESCRIPTION_NUM_BEAMS, rep_penalty: float = DESCRIPTION_REPETITION_PENALTY,
                           no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM, do_sample: bool = DESCRIPTION_DO_SAMPLE,
                           temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P,
                           top_k: int = DESCRIPTION_TOP_K, crop: bool = DESCRIPTION_CROP,
                           empty_ratio: float = EMPTY_MASK_PIXEL_RATIO,
                           dedup_words: bool = DESCRIPTION_DEDUP_WORDS) -> list[tuple[str, str]] | None:
    """
    Passes a batch of masked images to the BLIP model to get textual descriptions of the manipulated area.

    (Near-)all-black masks are not sent to BLIP (it would hallucinate); they receive EMPTY_MASK_LABEL instead.
    The remaining images are captioned with the configured decoding. The caption is forced to letters only
    and (optionally) de-duplicated (see utils.clean_text); when an OCR reader is given, the already-clean words
    printed inside the altered region (see ocr.read_text) are appended to it (captioning models cannot read text).

    Args:
        processor:       BLIP processor used to prepare the images for the model.
        model:           BLIP model that generates the captions.
        device:          Torch device on which the model and inputs live (cpu, cuda or mps).
        batch:           List of paths to the masked images in the current batch.
        reader:          Optional EasyOCR reader; its words are merged into each description (see ocr.read_text).
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        do_sample:       If True, sample (temperature/top_p/top_k) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        top_k:           Sample only among the K most likely tokens, 0 disables (only used when do_sample is True).
        crop:            If True, caption the non-black bounding box instead of the full mostly-black frame.
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty (see utils.is_empty_mask).
        dedup_words:     If True, drop duplicate words from the caption (see utils.clean_text).

    Returns:
        A list of (filename_stem, description) tuples, or None on failure or empty batch.
    """


    try:
        # One entry per image to caption: (stem, OCR words, RGB image); combined with the caption at the end.
        # Empty masks skip BLIP and go straight to `fixed` with a constant label.
        kept, fixed = [], []
        # Load all images in the batch
        for image_path in batch:
            # Force the image to be read in color mode
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"ERROR: '{image_path}' image not found or corrupted.")
                continue

            # (Near-)all-black mask: TruFor found no confident manipulation. Skip BLIP (it would hallucinate) and assign a fixed, non-empty label
            if is_empty_mask(image, empty_ratio):
                fixed.append((image_path.stem, EMPTY_MASK_LABEL))
                continue
            # Crop to the altered region so, full effective resolution instead of a mostly-black frame
            if crop:
                image = crop_to_content(image, OCR_CROP_PADDING)

            # Read the text printed inside the altered region (EasyOCR wants the BGR image, so read before the RGB conversion)
            # An OCR failure on one image must not lose the whole batch: degrade to caption-only and keep going
            try:
                words = read_text(reader, image) if reader else ""
            except Exception as e:
                print(f"WARNING: OCR failed on '{image_path.name}' ({e}): describing with the caption only.")
                words = ""

            # Hugging Face processors accept numpy arrays directly in RGB format
            kept.append((image_path.stem, words, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        # Every image was empty or corrupt, nothing to caption
        if not kept:
            return fixed or None


        # Process the entire batch of images and move the input tensors to the model's device
        inputs = processor(images=[image for _, _, image in kept], return_tensors="pt").to(device)
        # Decoding: anti-repetition kills the degenerate captions; sampling adds the lexical diversity beam search collapses
        gen = dict(min_new_tokens=min_token, max_new_tokens=max_token, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        if do_sample:
            gen.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            # early_stopping only exists for beam search (transformers warns if it is passed while sampling)
            gen.update(num_beams=num_beams, early_stopping=True)
        # Generate the predictions
        out = model.generate(**inputs, **gen)

        # One description per image: the cleaned BLIP caption joined with its already-clean OCR words (see ocr.read_text)
        captions = processor.batch_decode(out, skip_special_tokens=True)
        described = [(stem, f"{clean_text(caption, dedup_words)} {words}" or EMPTY_MASK_LABEL)
                     for (stem, words, _), caption in zip(kept, captions)]
        return fixed + described

    except Exception as e:
        print(f"ERROR: generating description: {e}")
        return None


def _report_descriptions(descriptions: list[tuple[str, str]], top: int = DESCRIPTION_TOP_WORDS, report: Path = DESCRIPTION_REPORT_FILE):
    """
    Prints and saves a compact quality summary of the generated descriptions.

    Meant as a quick post-run audit: a high empty share, extreme lengths or mask artifacts at the
    top of the word ranking (e.g. black/dark) are visible at a glance instead of requiring spot checks.

    Args:
        descriptions: List of (filename_stem, description) tuples produced by the run.
        top:          Number of most frequent words to show.
        report:       Output file path where the description summary is written.

    Returns:
        None.
    """

    lengths = sorted(len(description.split()) for _, description in descriptions)
    empties = sum(description == EMPTY_MASK_LABEL for _, description in descriptions)
    # Word ranking over the real descriptions only (the fixed empty label would dominate it)
    counts  = Counter(word for _, description in descriptions if description != EMPTY_MASK_LABEL for word in description.split())
    text = (f"Description summary: {len(descriptions)} images of which {empties} empty; "
            f"words per description min/median/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}\n"
            f"Most frequent words: {', '.join(f'{word} x{count}' for word, count in counts.most_common(top))}")
    print(text)
    save_report(text, report)


def describe_images(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device,
                    reader: easyocr.Reader | None = None,
                    batch_size: int = DESCRIPTION_BATCH_SIZE, min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                    num_beams: int = DESCRIPTION_NUM_BEAMS, rep_penalty: float = DESCRIPTION_REPETITION_PENALTY,
                    no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM, do_sample: bool = DESCRIPTION_DO_SAMPLE,
                    temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P,
                    top_k: int = DESCRIPTION_TOP_K, seed: int = DESCRIPTION_SEED, crop: bool = DESCRIPTION_CROP,
                    empty_ratio: float = EMPTY_MASK_PIXEL_RATIO, dedup_words: bool = DESCRIPTION_DEDUP_WORDS,
                    masked: Path = MASKED_DIR, dataset: Path = FILTERED_DATASET_FILE,):
    """
    Iterates over all masked images in masked_dir in batches and generates textual descriptions
    (the BLIP caption merged with the OCR words of the altered region when a reader is given),
    rewriting the dataset and freeing memory after each batch (so progress survives an interrupted run).

    Args:
        processor:       BLIP processor used to prepare the images for the model.
        model:           BLIP model that generates the captions.
        device:          Torch device on which the model and inputs live (cpu, cuda or mps).
        reader:          Optional EasyOCR reader; its words are merged into each description (see ocr.read_text).
        masked:          Directory containing the masked images to describe.
        dataset:         Path to the dataset CSV to merge the descriptions into.
        batch_size:      Number of images processed simultaneously per batch.
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        do_sample:       If True, sample (temperature/top_p/top_k) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        top_k:           Sample only among the K most likely tokens, 0 disables (only used when do_sample is True).
        seed:            Seeds the sampling once per run so reruns reproduce the same captions; negative = unseeded.
        crop:            If True, caption the non-black bounding box instead of the full mostly-black frame.
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty.
        dedup_words:     If True, drop duplicate words from the caption.

    Returns:
        None.
    """

    if not masked.is_dir() or not dataset.is_file() or max_token <= 0:
        print(f"ERROR: invalid masked dir '{masked}', dataset file '{dataset}' or max_token {max_token}.")
        return


    # Get image paths, excluding hidden files
    images = [path for path in masked.iterdir() if not path.name.startswith(".")]
    if not images:
        print(f"WARNING: No valid masked images found in '{masked}'.")
        return
    len_images = len(images)

    # Generating descriptions
    print(f"Describing masked images in folder: '{masked}'...")
    # Seed once per run: captions stay diverse across images but a full rerun reproduces them
    if do_sample and seed >= 0:
        torch.manual_seed(seed)
    # OCR reader + decoding / post-processing knobs shared by every batch
    params = (reader, min_token, max_token, num_beams, rep_penalty, no_repeat_ngram, do_sample, temperature, top_p, top_k, crop, empty_ratio, dedup_words)


    # No batch
    if batch_size <= 0:
        descriptions = _describe_images_batch(processor, model, device, images, *params)
        if descriptions:
            print(descriptions)
            update_dataset(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
            _report_descriptions(descriptions)
        return


    # load the dataset once, keep it in memory, and merge the descriptions collected so far after each batch
    df = load_csv(dataset)
    if df is None:
        return
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
    if descriptions:
        _report_descriptions(descriptions)


def main():
    """
    Loads BLIP (and the OCR reader unless disabled) and describes every masked image (caption + words printed in the altered region),
    merging the descriptions into the dataset.

    Args:
        None (paths and the generation tuning come from config.py; model, batch size and the OCR/sampling modes are per-run CLI flags).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Caption the masked anomaly region of every image with BLIP and merge the result into the dataset.")

    # Flags
    # Masked dir/dataset and the generation/post-processing knobs (token lengths, beam/sampling internals, empty-ratio, crop, dedup)
    # come from config.py
    parser.add_argument("--batch-size", type=int, default=DESCRIPTION_BATCH_SIZE, help=f"Number of images captioned simultaneously. Default: {DESCRIPTION_BATCH_SIZE}")
    # Switches in the override direction only (defaulting from config): caption-only and sampling-on.
    parser.add_argument("--no-ocr",     dest="use_ocr",   action="store_false",   help="Caption only: do not merge the OCR words of the altered region into the descriptions.")
    parser.add_argument("--do-sample",  dest="do_sample", action="store_true",    help="Sample (config temperature/top-p/top-k) instead of beam search for extra caption diversity.")
    parser.set_defaults(use_ocr=OCR_ENABLED, do_sample=DESCRIPTION_DO_SAMPLE)
    args = parser.parse_args()

    # Load the BLIP model
    blip = load_blip()
    if blip is None:
        return
    processor, model, device = blip

    # Load the OCR reader (its words are merged into the descriptions; languages and tuning knobs come from config)
    reader = None
    if args.use_ocr:
        reader = load_reader()
        if reader is None:
            return

    # Describe the masked images (everything but model/batch-size/OCR/sampling defaults from config.py)
    describe_images(processor, model, device, reader, batch_size=args.batch_size, do_sample=args.do_sample)

if __name__ == "__main__":
    main()