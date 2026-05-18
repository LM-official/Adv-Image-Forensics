from pathlib import Path


# Paths
CONFIG_FILE_PATH                = Path(__file__).resolve()
SCRIPTS_DIR_PATH                = CONFIG_FILE_PATH.parent
TEMP_DIR_PATH                   = SCRIPTS_DIR_PATH / "temp"


# TruFor
ROOT_DIR_PATH                   = SCRIPTS_DIR_PATH.parent.parent
IMG_DIR_PATH                    = ROOT_DIR_PATH / "Trufor_train_test/images"
NPZ_DIR_PATH                    = ROOT_DIR_PATH / "Trufor_train_test/output"
TRUFOR_RUN_PATH                 = ROOT_DIR_PATH / "TruFor_train_test/test.py"


# Dataset
DATASET_DIR_PATH                = ROOT_DIR_PATH / "Trufor_train_test/dataset/my_dataset"
DATASET_FILE_PATH               = DATASET_DIR_PATH / "crowdtangle.jsonl"
DATASET_IMGS_DIR_PATH           = DATASET_DIR_PATH / "crowdtangle_media"
FILTERED_DATASET_FILE_PATH      = DATASET_DIR_PATH / "filtered_dataset.csv"


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