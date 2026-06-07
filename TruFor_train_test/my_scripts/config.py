from pathlib import Path


# Primary directories
# Get the current working directory
SCRIPTS_DIR               = Path(__file__).resolve().parent
TRUFOR_DIR                = SCRIPTS_DIR.parent
ROOT_DIR                  = TRUFOR_DIR.parent
TEMP_DIR                  = SCRIPTS_DIR / "temp"


# Dataset
DATASET_DIR               = TRUFOR_DIR / "dataset/my_dataset"
DATASET_FILE              = DATASET_DIR / "crowdtangle.jsonl"
DATASET_IMGS_DIR          = DATASET_DIR / "crowdtangle_media"
FILTERED_DATASET_FILE     = TEMP_DIR / "filtered_dataset.csv"
# This must match the char_limit used at CrowdTangle download time, otherwise every URL->file lookup silently misses.
CHAR_LIMIT                = 64


# TruFor
TRUFOR_RUN                = TRUFOR_DIR / "test.py"
# Number of images to process per batch
TRUFOR_BATCH_SIZE         = 50
# Wait time (seconds) after each batch
WAIT                      = 180
IMGS_DIR                  = TRUFOR_DIR / "images"
NPZ_DIR                   = TRUFOR_DIR / "output"
NPZ_COLS                  = frozenset(("map", "imgsize", "score", "conf"))


# Performance
TRUTH_FILE                = TEMP_DIR / "truth.csv"
PERFORMANCE_THRESHOLD     = 0.5
PERFORMANCE_FILE          = TEMP_DIR / "performance.txt"


# Correlation and regression
CORRELATION_FILE          = TEMP_DIR / "correlation.txt"
REGRESSION_FILE           = TEMP_DIR / "regression.txt"


# Distribution
DISTRIBUTION_FILE         = TEMP_DIR / "distribution.txt"
DISTRIBUTION_PLOT_FILE    = TEMP_DIR / "distribution.png"


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


# Description and regression
BATCH_SIZE                = 15
# Anomaly description
MAX_DESCRIPTION_TOKENS    = 50
# Multimodal regression
SVD_COMPONENTS            = 10
MAX_NUMBER_OF_WORDS       = 100
ANOMALY_REGRESSION_FILE   = TEMP_DIR / "anomaly_regression.txt"