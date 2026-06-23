from   config import OCR_LANGS, OCR_MIN_CONFIDENCE, OCR_MAG_RATIO, OCR_ALLOWLIST, OCR_MIN_WORD_LEN, OCR_MAX_WORDS, OCR_GPU
import easyocr
import numpy  as     np
from   utils  import clean_text
import warnings

# EasyOCR's internal DataLoader always requests pinned memory, which some devices (e.g. MPS) do not support:
# torch then prints the same harmless UserWarning for every image. Pinning is just skipped, so silence that message.
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported")


def load_reader(langs: tuple[str, ...] = OCR_LANGS, gpu: bool = OCR_GPU) -> easyocr.Reader | None:
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


def read_text(reader: easyocr.Reader, image: np.ndarray, min_confidence: float = OCR_MIN_CONFIDENCE,
              mag_ratio: float = OCR_MAG_RATIO, allowlist: str | None = OCR_ALLOWLIST,
              min_word_len: int = OCR_MIN_WORD_LEN, max_words: int = OCR_MAX_WORDS) -> str:
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
    
    if max_words <= 0:
        print(f"max_words must be non-negative, got {max_words}")
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
        if count >= max_words:
            break

    # Keep the kept words most-confident first
    return " ".join(ordered)