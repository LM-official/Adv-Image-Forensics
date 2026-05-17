#!/bin/zsh

# Interrupt the script if any command fails
set -e

# Load environment variables
source config.sh


echo "Starting..."
# Activate the python virtual environment
echo " - Activating python virtual environment..."
source "${VENV_PATH}"
# Trap ensure the execution before exit even if the script is interrupted
trap deactivate EXIT
echo " - Python virtual environment activated."


# Filter dataset
ask_execute "dataset filtering" "python \"${SCRIPTS_DIR}/filter.py\" --dataset \"${DATASET_FILE}\" --dataset_images \"${DATASET_IMGS_DIR}\""


# Move to TruFor directory
cd "${TRUFOR_ROOT}"

# Run TruFor
ask_execute "TruFor" "python test.py --input \"${IMG_DIR}\" --output \"${NPZ_DIR}\""


# Move to scripts directory
cd "${SCRIPTS_DIR}"


# Generate truth
#ask_execute "truth file generation" "python truth_generator.py"

# Performance report
#ask_execute "performance evaluation" "python performance.py"

# Merged metrics
ask_execute "merge metrics" "python merge_metrics.py --dataset \"${DATASET_DIR}/${FILTERED_DATASET_FILE}\""

# Correlation and regression
ask_execute "correlation and regression evaluation" "python correlation_regression.py --metric \"realismScore\""

# Masking image manipulation
ask_execute "masking manipulated areas" "python masking.py"

# Generate description
ask_execute "image description" "python image_to_text.py"

# Multimodal regression
ask_execute "multimodal regression" "python multimodal_regression.py --metric \"realismScore\""


# Deactivate the python virtual environment
echo " - Deactivating python virtual environment..."
echo " - Python virtual environment deactivated."


echo "Run completed."
# Give execution permissions
# chmod +x run.sh

# Run the script
# ./run.sh