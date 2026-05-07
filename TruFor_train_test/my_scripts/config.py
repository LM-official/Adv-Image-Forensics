from pathlib import Path


# The absolute path to this config.py file.
SCRIPTS_DIR_PATH                = Path(__file__).resolve().parent
TEMP_DIR_PATH                   = SCRIPTS_DIR_PATH / "temp"


# TruFor
IMG_DIR_PATH                    = SCRIPTS_DIR_PATH.parent / "images"
NPZ_DIR_PATH                    = SCRIPTS_DIR_PATH.parent / "output"



# Performance
TRUTH_FILE_NAME                 = "truth.csv"
PERFORMANCE_FILE_NAME           = "performance.txt"
# From 0.0 to 1.0
PERFORMANCE_THRESHOLD           = 0.8

# Merged metrics
MERGED_METRICS_FILE_NAME        = "merged_metrics.csv"

# Correlation and regression
CORRELATION_FILE_NAME           = "correlation_{}.txt"
REGRESSION_FILE_NAME            = "regression_{}.txt"

# Mask
# From 0.0 to 1.0
MASK_THRESHOLD                  = 0.5
MASKED_DIR_NAME                 = "masked/"
BATCH_SIZE                      = 10

# Description
MAX_DESCRIPTION_LENGTH          = 50

# Embedding regression
# SVD (Singular Value Decomposition) components to extract from text
SVD_COMPONENTS                  = 10
# Max number of different words (features)
MAX_NUMBER_OF_WORDS             = 500
MULTIMODAL_REGRESSION_FILE_NAME = "multimodal_regression_{}.txt"