#!/bin/zsh


# Execute a command with user confirmation
ask_execute() {
    local message=$1
    local command_to_run=$2
    read "confirm?Run ${message}? (y/n): "
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo " - Running ${message}..."
        eval "$command_to_run"
        echo " - ${message} done."
    else
        echo " - ${message} skipped."
    fi
}

# Delete a file or directory with user confirmation
ask_remove() {
    local message=$1
    local command_to_run=$2
    read "confirm?Delete ${message}? (y/n): "
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo " - Removing ${message}..."
        eval "$command_to_run"
        echo " - ${message} removed."
    else
        echo " - ${message} skipped."
    fi
}


# Environment variables

# My scripts directory
# Get the directory of the current script
export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" >/dev/null 2>&1 && pwd )"
export TEMP_DIR="${SCRIPTS_DIR}/temp"


# TruFor paths
export TRUFOR_ROOT="$(dirname "$SCRIPTS_DIR")"
export IMG_DIR="${TRUFOR_ROOT}/images"
export NPZ_DIR="${TRUFOR_ROOT}/output"
export BASE_DIR="$(dirname "$TRUFOR_ROOT")"


# Venv path
export VENV_PATH="$BASE_DIR/.venv/bin/activate"


# Performance
export TRUTH_FILE="truth.csv"
export PERFORMANCE_FILE="performance.txt"

# Merged metrics
export MERGED_METRICS_FILE="merged_metrics.csv"

# Correlation and regression
export CORRELATION_FILE_PREFIX="correlation_"
export REGRESSION_FILE_PREFIX="regression_"

# Mask
export MASKED_DIR="masked"

# Description

# Multimodal regression
export MULTIMODAL_REGRESSION_FILE_PREFIX="multimodal_regression_"


# Dataset paths
export GENIMAGENET_RATINGS="/Users/lorenzo/Tirocinio/GenImageNet/data/ratings"
export CSV_ASSIGNMENT="${GENIMAGENET_RATINGS}/GenImageNet_assignmentLevel.csv"
export CSV_IMAGE="${GENIMAGENET_RATINGS}/GenImageNet_imageLevel.csv"