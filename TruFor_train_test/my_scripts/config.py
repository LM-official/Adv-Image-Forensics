from pathlib import Path


# Primary directories
# Get the current working directory
SCRIPTS_DIR                = Path(__file__).resolve().parent
TRUFOR_DIR                 = SCRIPTS_DIR.parent
ROOT_DIR                   = TRUFOR_DIR.parent
TEMP_DIR                   = SCRIPTS_DIR / "temp"


# Dataset
DATASET_DIR                = TRUFOR_DIR / "dataset/my_dataset"
DATASET_FILE               = DATASET_DIR / "crowdtangle.jsonl"
DATASET_IMGS_DIR           = DATASET_DIR / "crowdtangle_media"
FILTERED_DATASET_FILE      = TEMP_DIR / "filtered_dataset.csv"


# TruFor
TRUFOR_RUN                 = TRUFOR_DIR / "test.py"
IMG_DIR                    = TRUFOR_DIR / "images"
NPZ_DIR                    = TRUFOR_DIR / "output"


# Performance
TRUTH_FILE                 = TEMP_DIR / "truth.csv"
# From 0.0 to 1.0
PERFORMANCE_THRESHOLD      = 0.5
PERFORMANCE_FILE           = TEMP_DIR / "performance.txt"


# Correlation and regression
CORRELATION_FILE           = TEMP_DIR / "correlation.txt"
REGRESSION_FILE            = TEMP_DIR / "regression.txt"


# Mask
# From 0.0 to 1.0
MASK_THRESHOLD             = 0.5
MASKED_DIR                 = TEMP_DIR / "masked"
BATCH_SIZE                 = 10
# Anomaly description
MAX_DESCRIPTION_LENGTH     = 50
# Multimodal regression
SVD_COMPONENTS             = 10
MAX_NUMBER_OF_WORDS        = 500
MULTIMODAL_REGRESSION_FILE = TEMP_DIR / "multimodal_regression.txt"