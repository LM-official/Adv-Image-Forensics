from pathlib import Path


# TruFor
IMG_DIR_PATH                    = Path('/Users/lorenzo/Tirocinio/TruFor/TruFor_train_test/images/')
NPZ_DIR_PATH                    = Path('/Users/lorenzo/Tirocinio/TruFor/TruFor_train_test/output/')


TEMP_DIR_PATH                   = Path('temp/')

# performance
TRUTH_FILE_NAME                 = 'truth.csv'
PERFORMANCE_FILE_NAME           = 'performance.txt'
# from 0.0 to 1.0
PERFORMANCE_THRESHOLD           = 0.5

# merged metrics
MERGED_METRICS_FILE_NAME        = 'merged_metrics.csv'

# correlation and regression
CORRELATION_FILE_NAME           = 'correlation_{}.txt'
REGRESSION_FILE_NAME            = 'regression_{}.txt'

# mask
# from 0.0 to 1.0
MASK_THRESHOLD                  = 0.5
MASKED_DIR_NAME                 = 'masked/'

# description
MAX_DESCRIPTION_LENGTH          = 50

# embedding regression
# SVD (Singular Value Decomposition) components to extract from text
SVD_COMPONENTS                  = 10
# max number of different words
MAX_NUMBER_OF_WORDS             = 500
MULTIMODAL_REGRESSION_FILE_NAME = 'multimodal_regression_{}.txt'