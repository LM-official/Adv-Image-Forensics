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


# Primary directories
# Get the current working directory
export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" >/dev/null 2>&1 && pwd )"
export TRUFOR_DIR="$(dirname "$SCRIPTS_DIR")"
export ROOT_DIR="$(dirname "$TRUFOR_DIR")"
export TEMP_DIR="${SCRIPTS_DIR}/temp"


# Venv
export VENV_PATH="${ROOT_DIR}/.venv/bin/activate"


# Dataset
export DATASET_DIR="${TRUFOR_DIR}/dataset/my_dataset"
export DATASET_FILE="${DATASET_DIR}/crowdtangle.jsonl"
export DATASET_IMGS_DIR="${DATASET_DIR}/crowdtangle_media"
export FILTERED_DATASET_FILE="${TEMP_DIR}/filtered_dataset.csv"


# TruFor
export IMG_DIR="${TRUFOR_DIR}/images"
export NPZ_DIR="${TRUFOR_DIR}/output"


# Performance
export TRUTH_FILE="${TEMP_DIR}/truth.csv"
export PERFORMANCE_FILE="${TEMP_DIR}/performance.txt"


# Correlation and regression
export CORRELATION_FILE="${TEMP_DIR}/correlation.txt"
export REGRESSION_FILE="${TEMP_DIR}/regression.txt"


# Mask
export MASKED_DIR="${TEMP_DIR}/masked"
# Multimodal regression
export ANOMALY_REGRESSION_FILE="${TEMP_DIR}/anomaly_regression.txt"