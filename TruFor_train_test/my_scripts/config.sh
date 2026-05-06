#!/bin/zsh
# environment variables

# venv path
export VENV_PATH="/Users/lorenzo/trufor_py_env/bin/activate"

# TruFor paths
export TRUFOR_ROOT="/Users/lorenzo/Tirocinio/TruFor/TruFor_train_test"
export IMG_DIR="${TRUFOR_ROOT}/images"
export NPZ_DIR="${TRUFOR_ROOT}/output"

# scripts relative paths
export SCRIPTS_DIR="${TRUFOR_ROOT}/my_scripts"
export TEMP_DIR="${SCRIPTS_DIR}/temp"


# performance
export TRUTH_FILE="truth.csv"
export PERFORMANCE_FILE="performance.txt"

# merged metrics
export MERGED_METRICS_FILE="merged_metrics.csv"

# correlation and regression
export CORRELATION_FILE_PREFIX="correlation_"
export REGRESSION_FILE_PREFIX="regression_"

# mask
export MASKED_DIR="masked"

# description

# multimodal regression
export MULTIMODAL_REGRESSION_FILE_PREFIX="multimodal_regression_"


# GenImageNet relative paths
export GENIMAGENET_RATINGS="/Users/lorenzo/Tirocinio/GenImageNet/data/ratings"
export CSV_ASSIGNMENT="${GENIMAGENET_RATINGS}/GenImageNet_assignmentLevel.csv"
export CSV_IMAGE="${GENIMAGENET_RATINGS}/GenImageNet_imageLevel.csv"