import argparse
from   collections  import Counter
import gc
from   config       import (DESCRIPTION_MODEL, FILTERED_DATASET_FILE, MASKED_DIR, DESCRIPTION_REPORT_FILE, DESCRIPTION_TEMP_FILE,
                            DESCRIPTION_BATCH_SIZE, MIN_DESCRIPTION_TOKENS, MAX_DESCRIPTION_TOKENS,
                            DESCRIPTION_NUM_BEAMS, DESCRIPTION_SEED,
                            DESCRIPTION_DO_SAMPLE, DESCRIPTION_TEMPERATURE, DESCRIPTION_TOP_P, DESCRIPTION_TOP_K,
                            DESCRIPTION_REPETITION_PENALTY, DESCRIPTION_NO_REPEAT_NGRAM,
                            DESCRIPTION_CROP, EMPTY_MASK_PIXEL_RATIO, DESCRIPTION_DEDUP_WORDS, EMPTY_MASK_LABEL,
                            DESCRIPTION_TOP_WORDS,

                            OCR_ENABLED, OCR_GPU, OCR_LANGS, OCR_MIN_CONFIDENCE, OCR_MAG_RATIO, OCR_MIN_WORD_LEN,
                            OCR_MAX_WORDS, OCR_CROP_PADDING, OCR_ALLOWLIST)
import cv2
import easyocr
import numpy        as     np
from   pathlib      import Path
import pandas       as     pd
import torch
from   tqdm         import tqdm
from   transformers import BlipProcessor, BlipForConditionalGeneration
from   utils        import clean_text, load_csv, save_report, update_csv
import warnings

# EasyOCR hardcodes pin_memory=True in its internal DataLoader, which MPS does not support:
# torch emits a harmless UserWarning on every readtext() call, flooding the output. Silence it.
warnings.filterwarnings("ignore", message=".*pin_memory.*")


# =============================================================================
# (0) IMAGE HELPERS
# =============================================================================

def _is_empty_mask(image: np.ndarray, ratio: float) -> bool:
    """
    Tells whether a masked image is (near-)all-black.

    Models hallucinate on a black image (BLIP invents a caption, OCR finds no text),
    so such images are detected and skipped: the caller assigns them a fixed label instead.

    Args:
        image: Masked BGR image as read by cv2.
        ratio: Maximum fraction of non-black pixels (any non-zero channel) for the image to count as empty.

    Returns:
        True if the non-black pixel fraction is <= ratio (effectively all black), else False.
    """

    # Non-black pixel fraction: flag each pixel with any non-zero channel, then average the flags over the image
    return float(np.any(image > 0, axis=2).mean()) <= ratio


def _crop_image(image: np.ndarray, padding: int) -> np.ndarray:
    """
    Crops a masked image to the bounding box of its non-black pixels (plus a margin).

    The masked images are mostly black with only the altered region visible: cropping away the black border
    lets the models work at the scale of the region instead of the scale of the full image.

    Args:
        image:   Masked BGR image as read by cv2 (must contain at least one non-black pixel).
        padding: Pixels of margin kept around the non-black bounding box (clamped to the image edges).

    Returns:
        The cropped BGR image, or the image unchanged when there is nothing to crop.
    """

    # Rows (ys) and columns (xs) of all visible pixels
    ys, xs = np.where(np.any(image > 0, axis=2))
    
    # Top-left (y0, x0) and bottom-right (y1, x1) corners of the rectangle around the visible pixels, widened by the padding margin and clamped to the image borders
    y0, x0 = max(int(ys.min()) - padding, 0), max(int(xs.min()) - padding, 0)
    y1, x1 = min(int(ys.max()) + padding, image.shape[0] - 1), min(int(xs.max()) + padding, image.shape[1] - 1)
    # Cut the rectangle out
    return image[y0:y1 + 1, x0:x1 + 1]


# =============================================================================
# (1) IMAGE CAPTIONING BLIP
# =============================================================================

def load_blip(model_name: str = DESCRIPTION_MODEL) -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    """
    Loads the Hugging Face BLIP processor and model for image captioning.

    Args:
        model_name: Hugging Face repo id of the BLIP captioning model (e.g. blip-image-captioning-large).

    Returns:
        A tuple (processor, model, device) ready for inference, or None if loading failed.
    """

    print("Loading Hugging Face model (BLIP)...")
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


def _report_descriptions(descriptions: list[tuple[str, str]], top: int = DESCRIPTION_TOP_WORDS, report: Path = DESCRIPTION_REPORT_FILE):
    """
    Prints and saves a compact quality summary of the generated descriptions.

    Meant as a quick post-run audit: a high empty share, extreme lengths or mask artifacts at the
    top of the word ranking (e.g. black/dark) are visible at a glance instead of requiring spot checks.
    The empty share is counted over every image, while the length stats and the word ranking describe
    the real descriptions only.

    Args:
        descriptions: List of (filename_stem, description) tuples produced by the run.
        top:          Number of most frequent words to show.
        report:       Output file path where the description summary is written.

    Returns:
        None.
    """

    # Length stats and word ranking over the real descriptions only
    described = [description for _, description in descriptions if description != EMPTY_MASK_LABEL]
    empties   = len(descriptions) - len(described)
    lengths   = sorted(len(description.split()) for description in described)
    counts    = Counter(word for description in described for word in description.split())
    # Every mask was empty: there is no length distribution to summarise
    quantiles = f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}" if lengths else "-/-/-"
    text = (
        f"Description summary: {len(descriptions)} images of which {empties} empty; "
        f"words per non-empty description min/median/max = {quantiles}\n"
        f"Most frequent words: {', '.join(f'{word} x{count}' for word, count in counts.most_common(top))}"
    )
    save_report(text, report)


def _describe_images_batch(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, reader: easyocr.Reader | None = None,
                           batch: list[Path] = [], min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                           num_beams: int = DESCRIPTION_NUM_BEAMS,
                           do_sample: bool = DESCRIPTION_DO_SAMPLE,
                           temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P, top_k: int = DESCRIPTION_TOP_K,
                           rep_penalty: float = DESCRIPTION_REPETITION_PENALTY, no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM,
                           crop: bool = DESCRIPTION_CROP, empty_ratio: float = EMPTY_MASK_PIXEL_RATIO,
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
        reader:          Optional EasyOCR reader; its words are merged into each description (see ocr.read_text).
        batch:           List of paths to the masked images in the current batch.
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        do_sample:       If True, sample (temperature/top_p/top_k) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        top_k:           Sample only among the K most likely tokens, 0 disables (only used when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        crop:            If True, caption the non-black bounding box instead of the full mostly-black frame.
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty (see utils._is_empty_mask).
        dedup_words:     If True, drop duplicate words from the caption (see utils.clean_text).

    Returns:
        A list of (filename_stem, description) tuples, or None on failure or empty batch.
    """

    try:
        # One entry per image to caption: (stem, OCR words, RGB image); combined with the caption at the end.
        # Empty masks skip BLIP and go straight to `fixed` with a constant label.
        kept, fixed = [], []
        # Load all images in the batch
        for image_path in tqdm(batch, desc="Reading batch", unit="img", leave=False):
            # Force the image to be read in color mode
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            # (Near-)all-black mask: TruFor found no confident manipulation. Skip BLIP (it would hallucinate) and assign a fixed, non-empty label
            if _is_empty_mask(image, empty_ratio):
                fixed.append((image_path.stem, EMPTY_MASK_LABEL))
                continue
            # Crop to the altered region so, full effective resolution instead of a mostly-black frame
            if crop:
                image = _crop_image(image, OCR_CROP_PADDING)

            # Read the text printed inside the altered region (EasyOCR wants the BGR image, so read before the RGB conversion)
            # An OCR failure on one image must not lose the whole batch: degrade to caption-only and keep going
            try:
                words = read_text(reader, image) if reader else ""
            except Exception:
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
        described = [(stem, f"{clean_text(caption, dedup_words)} {words}".strip() or EMPTY_MASK_LABEL)
                     for (stem, words, _), caption in zip(kept, captions)]
        return fixed + described

    except Exception as e:
        print(f"ERROR: generating descriptions: {e}")
        return None


def describe_images(use_ocr: bool = True, dataset: Path = FILTERED_DATASET_FILE, masked: Path = MASKED_DIR, temp: Path = DESCRIPTION_TEMP_FILE,
                    batch_size: int = DESCRIPTION_BATCH_SIZE, min_token: int = MIN_DESCRIPTION_TOKENS, max_token: int = MAX_DESCRIPTION_TOKENS,
                    num_beams: int = DESCRIPTION_NUM_BEAMS, seed: int = DESCRIPTION_SEED,
                    do_sample: bool = DESCRIPTION_DO_SAMPLE,
                    temperature: float = DESCRIPTION_TEMPERATURE, top_p: float = DESCRIPTION_TOP_P, top_k: int = DESCRIPTION_TOP_K,
                    rep_penalty: float = DESCRIPTION_REPETITION_PENALTY, no_repeat_ngram: int = DESCRIPTION_NO_REPEAT_NGRAM,
                    crop: bool = DESCRIPTION_CROP, empty_ratio: float = EMPTY_MASK_PIXEL_RATIO,
                    dedup_words: bool = DESCRIPTION_DEDUP_WORDS):
    """
    Iterates over all masked images in masked_dir in batches and generates textual descriptions
    (the BLIP caption merged with the OCR words of the altered region when a reader is given),
    rewriting the dataset and freeing memory after each batch (so progress survives an interrupted run).

    Args:
        use_ocr:         If True, merge the OCR words of the altered region into the descriptions.
        dataset:         Path to the dataset CSV to merge the descriptions into.
        masked:          Directory containing the masked images to describe.
        temp:            Path to the temporary CSV where the batch results are appended before merging into the dataset.
        batch_size:      Number of images processed simultaneously per batch.
        min_token:       Minimum number of new tokens in each generated description.
        max_token:       Maximum number of new tokens in each generated description.
        num_beams:       Beam-search width (keep best of X generation), (ignored when do_sample is True).
        seed:            Seeds the sampling once per run so reruns reproduce the same captions; negative = unseeded.
        do_sample:       If True, sample (temperature/top_p/top_k) instead of beam search for more lexical diversity.
        temperature:     Sampling temperature/randomness (only used when do_sample is True).
        top_p:           Nucleus sampling probability (only used when do_sample is True).
        top_k:           Sample only among the K most likely tokens, 0 disables (only used when do_sample is True).
        rep_penalty:     Repetition penalty applied during generation.
        no_repeat_ngram: Size of the n-gram that may not repeat during generation.
        crop:            If True, caption the non-black bounding box instead of the full mostly-black frame.
        empty_ratio:     Non-black pixel fraction at or below which a mask counts as empty.
        dedup_words:     If True, drop duplicate words from the caption.

    Returns:
        None.
    """

    print("Describing masked images...")

    if not dataset.is_file() or not masked.is_dir():
        print("ERROR: invalid dataset.")
        return
    if min_token < 1 or max_token < min_token:
        print("ERROR: invalid token range.")
        return
    if ((do_sample and not (0.0 < top_p <= 1.0)) or
        no_repeat_ngram < 0 or
        not (0.0 <= empty_ratio <= 1.0)):
        print("ERROR: invalid generation parameters.")
        return
    # temperature, repetition_penalty, num_beams and top_k has delegated checks

    # Load BLIP model
    blip = load_blip()
    if blip is None:
        return
    processor, model, device = blip

    # Load OCR reader
    reader = None
    if use_ocr:
        reader = load_reader()
        if reader is None:
            return
    

    # Get image paths, excluding hidden files
    images = [path for path in masked.iterdir() if not path.name.startswith(".")]
    if not images:
        print("ERROR: No valid masked images found.")
        return
    len_images = len(images)

    # The dataset is left untouched while the batches run
    # Each finished batch appends its filename, anomaly_txt rows to a temp file
    # An interrupted run keeps everything computed and never touch the dataset
    # A restart skips the images already in the temp file
    # Once every batch is done the temp file is merged into the dataset in a single update and removed
    if temp.is_file():
        # Resume: skip the images already described in the leftover checkpoint
        done    = load_csv(temp)
        already = set(done["filename"]) if done is not None else set()
        images  = [image for image in images if image.stem not in already]
        len_images = len(images)
        print(f"Resuming interrupted run: {len(already)} already described, {len_images} left.")
    else:
        # New run: create an empty checkpoint (header only) so every batch can simply append
        # Create the output directories if they don't exist
        temp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["filename", "anomaly_txt"]).to_csv(temp, index=False, encoding="utf-8")


    # Generating descriptions
    # Seed once per run: captions stay diverse across images but a full rerun reproduces them
    if do_sample and seed >= 0:
        torch.manual_seed(seed)
    # OCR reader + decoding / post-processing knobs shared by every batch
    params = (min_token, max_token, num_beams, do_sample, temperature, top_p, top_k, rep_penalty, no_repeat_ngram, crop, empty_ratio, dedup_words)

    # No batch
    if batch_size <= 0:
        descriptions = _describe_images_batch(processor, model, device, reader, images, *params)
        if descriptions:
            update_csv(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
            _report_descriptions(descriptions)
            print("Masked images described successfully.")
        else:
            print("ERROR: no descriptions were generated.")
        return

    # Process images in chunks of size = batch_size: caption -> append to the checkpoint -> free memory
    batches = (len_images + batch_size - 1) // batch_size
    for i in range(0, len_images, batch_size):
        print(f"Processing batch {i // batch_size + 1}/{batches}...", end="\r")

        results = _describe_images_batch(processor, model, device, reader, images[i:i + batch_size], *params)
        if results:
            # The checkpoint already has its header, so append the rows only
            pd.DataFrame(results, columns=["filename", "anomaly_txt"]).to_csv(temp, mode="a", header=False, index=False, encoding="utf-8")
        else:
            print(f"WARNING: No descriptions generated for batch {i // batch_size + 1}.")

        # Release the per-batch image / inference tensors before loading the next batch
        _free_memory(device)

    # Merge every description (any resumed + this run) into the dataset in one pass, then drop the checkpoint
    desc_df = load_csv(temp)
    if desc_df is None:
        # Delete temp file
        temp.unlink(missing_ok=True)
        print("ERROR: no descriptions were generated.")
        return

    update_csv(dataset, desc_df)
    _report_descriptions(list(desc_df.itertuples(index=False, name=None)))
    # Delete temp file
    temp.unlink(missing_ok=True)
    print("Masked images described successfully.")


# =============================================================================
# (2) OCR TEXT RECOGNITION
# =============================================================================

def load_reader(gpu: bool = OCR_GPU, langs: tuple[str, ...] = OCR_LANGS) -> easyocr.Reader | None:
    """
    Loads the EasyOCR reader for text recognition.

    Args:
        langs: Language packs to load together (must share a compatible script, e.g. all Latin).
        gpu:   If True use the GPU (CUDA/MPS) when available, otherwise run on CPU.

    Returns:
        An easyocr.Reader ready for inference, or None if loading failed.
    """

    print(f"Loading EasyOCR reader for languages: {list(langs)}...")
    try:
        # The reader bundles a text detector + recognizer; models are downloaded once and cached locally
        reader = easyocr.Reader(list(langs), gpu=gpu)
        print(f"EasyOCR reader loaded successfully (device: {reader.device}).")
        return reader

    except Exception as e:
        print(f"ERROR: loading EasyOCR reader: {e}")
        return None


def read_text(reader: easyocr.Reader, image: np.ndarray,
              min_confidence: float = OCR_MIN_CONFIDENCE, mag_ratio: float = OCR_MAG_RATIO,
              min_word_len: int = OCR_MIN_WORD_LEN, max_words: int = OCR_MAX_WORDS, allowlist: str | None = OCR_ALLOWLIST) -> str:
    """
    Reads the text printed inside the visible (altered) region of one masked image.

    The given image is read by EasyOCR as-is (cropping it to the altered region, if wanted, is the caller's job);
    the recognized words are then filtered for quality:
    - boxes recognized with confidence below min_confidence are dropped (misspelled/garbled recognitions sit low);
    - words with fewer than min_word_len letters are dropped (confident 1-2 letter fragments from logos/graphics);
    - duplicates are dropped, and at most max_words words are kept, preferring the most confident ones
      (so an image dense with text cannot flood the description).
    The kept words are returned cleaned (letters only, see utils.clean_text) in reading order.

    Args:
        reader:         EasyOCR reader that detects and recognizes the text.
        image:          Masked BGR image as read by cv2 (typically already cropped to the altered region by the caller).
        min_confidence: Drop words recognized with confidence below this.
        mag_ratio:      Magnification applied by EasyOCR before detection; > 1 helps small print.
        allowlist:      Restrict recognition to these characters; None = no restriction.
        min_word_len:   Drop words with fewer letters than this.
        max_words:      Keep at most this many words (the most confident ones); 0 disables the cap.

    Returns:
        The recognized words joined by spaces, or "" when the region contains no readable text.
    """
    
    # print("Reading the text printed in the altered region (OCR)...")

    if max_words < 0:
        print(f"ERROR: max_words must be non-negative, got {max_words}")
        return ""


    # Detect + recognize; each result is a (bbox, text, confidence) tuple
    boxes = reader.readtext(image, mag_ratio=mag_ratio, allowlist=allowlist)

    # Candidate words, cleaned (confidence drives the quality budget below)
    candidates = [(confidence, word)
                  for (_, text, confidence) in boxes if confidence >= min_confidence
                  for word in clean_text(text, False).split() if len(word) >= min_word_len]

    # walk the candidates from the most to the least confident and keep each distinct word once, until max_words words are selected
    selected, count, ordered = set(), 0, []
    for _, word in sorted(candidates, key=lambda candidate: -candidate[0]):
        if word not in selected:
            selected.add(word)
            ordered.append(word)
            count += 1
        # 0 disables the cap (keep every distinct word)
        if max_words and count >= max_words:
            break

    # Keep the kept words most-confident first
    # print("Text read successfully.")
    return " ".join(ordered)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Describe every masked images
    (1) BLIP image captioning and
    (2) OCR (if enabled) text recognition.
    Then merge the descriptions into the dataset and generate a summary report.

    Args:
        None (paths and the generation tuning come from config.py; model, batch size and the OCR/sampling modes are per-run CLI flags).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Caption with BLIP and extract text with OCR of each image and merge the result into the dataset.")

    # Flags, many arguments come from config.py (the project backbone)
    # BLIP
    parser.add_argument("--batch-size", type=int,         default=DESCRIPTION_BATCH_SIZE, help=f"Number of images captioned simultaneously. Default: {DESCRIPTION_BATCH_SIZE}")
    parser.add_argument("--do-sample",  dest="do_sample", action="store_true",            help="Sample (config temperature/top-p/top-k) instead of beam search for extra caption diversity.")
    
    # OCR
    # Switches in the override direction only (defaulting from config): caption-only and sampling-on.
    parser.add_argument("--no-ocr", dest="use_ocr", action="store_false", help="Caption only: do not merge the OCR words of the altered region into the descriptions.")
    parser.set_defaults(use_ocr=OCR_ENABLED, do_sample=DESCRIPTION_DO_SAMPLE)
    args = parser.parse_args()

    # Describe
    describe_images(use_ocr=args.use_ocr, batch_size=args.batch_size, do_sample=args.do_sample)

if __name__ == "__main__":
    main()