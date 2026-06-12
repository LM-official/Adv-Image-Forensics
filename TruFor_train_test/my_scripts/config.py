from pathlib import Path


# Primary directories
# Get the current working directory
SCRIPTS_DIR = Path(__file__).resolve().parent
TRUFOR_DIR  = SCRIPTS_DIR.parent
ROOT_DIR    = TRUFOR_DIR.parent
TEMP_DIR    = SCRIPTS_DIR / "temp"


# Dataset
DATASET_DIR           = TRUFOR_DIR / "dataset/my_dataset"
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
IMGS_DIR          = TRUFOR_DIR / "images"
NPZ_DIR           = TRUFOR_DIR / "output"
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
DESCRIPTION_REPETITION_PENALTY = 1.5
DESCRIPTION_NO_REPEAT_NGRAM    = 2
# Off = deterministic beam search
DESCRIPTION_DO_SAMPLE          = False
# Randomness, higher = more random
DESCRIPTION_TEMPERATURE        = 0.9
DESCRIPTION_TOP_P              = 0.9
# Empty masks: if the non-black pixel fraction is <= this, skip BLIP and write EMPTY_MASK_LABEL
# Collapses the clear hallucination zone (pure-black + tiny spots) to an honest label
EMPTY_MASK_PIXEL_RATIO    = 0.02
EMPTY_MASK_LABEL          = "empty"
# Caption post-processing: keep only a-z and drop duplicate consecutive words.
DESCRIPTION_DEDUP_WORDS   = True


# Multimodal regression
SVD_COMPONENTS          = 10
MAX_NUMBER_OF_WORDS     = 100
ANOMALY_REGRESSION_FILE = TEMP_DIR / "anomaly_regression.txt"