from pathlib import Path


# Primary directories
# Get the current working directory
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPTS_DIR.parent
TRUFOR_DIR  = ROOT_DIR / "TruFor" / "TruFor_train_test"
TEMP_DIR    = SCRIPTS_DIR / "temp"


# Dataset
DATASET_DIR           = ROOT_DIR / "dataset"
DATASET_FILE          = DATASET_DIR / "crowdtangle.jsonl"
DATASET_IMGS_DIR      = DATASET_DIR / "crowdtangle_media"
FILTERED_DATASET_FILE = TEMP_DIR / "filtered_dataset.csv"
# This must match the char_limit used at CrowdTangle download time, otherwise every URL->file lookup silently misses.
CHAR_LIMIT             = 64


# TruFor
TRUFOR_RUN        = TRUFOR_DIR / "test.py"
# Number of images to process per batch
TRUFOR_BATCH_SIZE = 50
# Wait time (seconds) after each batch
TRUFOR_WAIT       = 180
# Downscale images above this pixel count before inference (maps are upsampled back, so .npz arrays always match the original image size)
# Full resolution on multi-megapixel images is RAM expensive; 0 disables the cap
TRUFOR_MAX_PIXELS = 6_000_000
IMGS_DIR          = DATASET_DIR / "images"
NPZ_DIR           = DATASET_DIR / "output"
NPZ_COLS          = frozenset(("map", "imgsize", "score", "conf"))


# Performance
TRUTH_FILE            = TEMP_DIR / "truth.csv"
PERFORMANCE_THRESHOLD = 0.5
PERFORMANCE_FILE      = TEMP_DIR / "performance.txt"


# General statistics
CORRELATION_FILE       = TEMP_DIR / "correlation.txt"
REGRESSION_FILE        = TEMP_DIR / "regression.txt"
DISTRIBUTION_FILE      = TEMP_DIR / "distribution.txt"
DISTRIBUTION_PLOT_FILE = TEMP_DIR / "distribution.png"


# Mask
# TruFor reliability (TCP) cutoff: keep a flagged pixel only if TruFor is at least this sure of its call.
# Gentle floor: drops only very-unsure pixels. Raise it for stricter (but more all-black) masks.
MASK_CONFIDENCE_THRESHOLD = 0.30
# "Flagged as altered" cutoff applied to the anomaly map.
MASK_THRESHOLD            = 0.5
# Mask cleanup (post-processing that removes scattered noise from the raw binary mask)
# Side of the square kernel for the morphological open/close (erases specks, fills holes, smooths edges); <= 1 disables it
MASK_MORPH_KERNEL         = 9
# Drop connected blobs smaller than this fraction of the image area; <= 0 disables it
MASK_MIN_AREA_RATIO       = 0.002
# Mask inversion: fix TruFor's polarity on synthetic-majority composites.
# Detected by texture asymmetry: invert when (background flat-fraction - flagged-region flat-fraction) >= MASK_INVERT_MARGIN.
MASK_INVERT_COMPOSITE     = True
# Gradient magnitude below this (on the 0-255 grayscale image) counts a pixel as "flat" (no texture).
MASK_FLAT_GRAD_THRESH     = 12
# Invert when the flagged region is photographic while the background is a flat banner by at least this much.
MASK_INVERT_MARGIN        = 0.25
# Only invert when the flagged region is at least this fraction of the image:
# prevents inverting a tiny noise speck (near-authentic image) into a fully-colored mask.
MASK_INVERT_MIN_AREA      = 0.03
MASKED_DIR                = TEMP_DIR / "masked"


# Description
# Swappable: "Salesforce/blip-image-captioning-base" / "...-large" / "Salesforce/blip2-*".
DESCRIPTION_MODEL              = "Salesforce/blip-image-captioning-large"
DESCRIPTION_BATCH_SIZE         = 20
# Wait time (seconds) after each batch
DESCRIPTION_WAIT               = 30
# Length of the caption
MIN_DESCRIPTION_TOKENS         = 5
MAX_DESCRIPTION_TOKENS         = 40
# Beam search (keep best of X generation)
DESCRIPTION_NUM_BEAMS          = 5
# Anti-repetition
DESCRIPTION_REPETITION_PENALTY = 1.3
DESCRIPTION_NO_REPEAT_NGRAM    = 2
# Off = deterministic beam search (generic but faithful captions; the OCR words supply the lexical variance);
# on = nucleus sampling for extra caption-side diversity at some fidelity cost
DESCRIPTION_DO_SAMPLE          = False
# Randomness, higher = more random
DESCRIPTION_TEMPERATURE        = 0.7
DESCRIPTION_TOP_P              = 0.9
# Sample only among the top-K candidate tokens; 0 disables the cutoff
DESCRIPTION_TOP_K              = 50
# Seeds the sampling so rerunning the full pipeline reproduces the same captions; negative = different captions every run
DESCRIPTION_SEED               = 42
# Empty masks: if the non-black pixel fraction is <= this, skip BLIP and write EMPTY_MASK_LABEL
# Collapses the clear hallucination zone (pure-black + tiny spots) to an honest label
EMPTY_MASK_PIXEL_RATIO    = 0.02
EMPTY_MASK_LABEL          = "empty"
# Caption post-processing: keep only a-z and drop duplicate consecutive words.
DESCRIPTION_DEDUP_WORDS   = True
# Crop each masked image to its non-black bounding box (margin = OCR_CROP_PADDING) before captioning:
# the model sees the altered region at full effective resolution instead of a mostly-black frame,
# and stops describing the mask itself (black/dark/silhouette captions)
DESCRIPTION_CROP          = True
# Number of most frequent words shown in the post-run description summary
DESCRIPTION_TOP_WORDS     = 10
# Output file for the post-run description quality summary
DESCRIPTION_REPORT_FILE   = TEMP_DIR / "description_report.txt"


# OCR
# Text printed inside the altered region: grounded, naturally high-variance tokens for the tfidf/svd step
# (image captioning models cannot read text, so banners/posters caption generically; OCR recovers the actual words)
# On = merge the OCR words of the altered region into each description; off = captions only
OCR_ENABLED         = True
# EasyOCR language packs loaded together (must share a compatible script): set them to the dataset's languages
OCR_LANGS           = ("it", "en")
# Drop words recognized with confidence below this (misspelled/garbled recognitions sit low, real text sits high)
OCR_MIN_CONFIDENCE  = 0.65
# Drop words with fewer letters than this (kills the 1-2 letter fragments OCR confidently finds in logos/graphics)
OCR_MIN_WORD_LEN    = 3
# Keep at most this many OCR words per image, preferring the most confident ones (so dense
# infographics cannot flood the description); 0 disables the cap
OCR_MAX_WORDS       = 20
# Pixels of margin kept around the non-black bounding box when cropping the masked image for OCR
OCR_CROP_PADDING    = 10
# EasyOCR magnification before detection; > 1 helps small print
OCR_MAG_RATIO       = 1.5
# Optionally restrict recognition to a character set (e.g. "abc...z0-9 "); None = no restriction, works for any language.
# Accented letters need no special handling: the cleaner folds them to plain a-z afterwards
OCR_ALLOWLIST       = None
# Use the GPU (CUDA/MPS) when available; switch off if EasyOCR misbehaves on it
OCR_GPU             = True


# Multimodal regression
SVD_COMPONENTS          = 10
MAX_NUMBER_OF_WORDS     = 100
ANOMALY_REGRESSION_FILE = TEMP_DIR / "anomaly_regression.txt"