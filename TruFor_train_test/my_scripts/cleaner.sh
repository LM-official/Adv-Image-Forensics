#!/bin/zsh

# Interrupt the script if any command fails
set -e

# Load environment variables
source config.sh


echo "Starting cleanup of the environment..."


# Remove filtered dataset
ask_remove "'${FILTERED_DATASET_FILE}'" "rm -f \"${FILTERED_DATASET_FILE}\""

# Remove all images in the input directory
ask_remove "all images in input directory" "rm -rf \"${IMGS_DIR}\"/*(N)"

# Remove all .npz files in the output directory
# Warning: compute all .npz is expensive
# ask_remove "all .npz files in output directory" "rm -rf \"${NPZ_DIR}\"/*.npz(N)"

# Remove truth file
ask_remove "'${TRUTH_FILE}'" "rm -f \"${TRUTH_FILE}\""

# Remove performance file
ask_remove "'${PERFORMANCE_FILE}'" "rm -f \"${PERFORMANCE_FILE}\""

# Remove general analysis files
ask_remove "'${CORRELATION_FILE}'" "rm -f \"${CORRELATION_FILE}\""
ask_remove "'${REGRESSION_FILE}'" "rm -f \"${REGRESSION_FILE}\""
ask_remove "'${DISTRIBUTION_FILE}'" "rm -f \"${DISTRIBUTION_FILE}\""
ask_remove "'${DISTRIBUTION_PLOT_FILE}'" "rm -f \"${DISTRIBUTION_PLOT_FILE}\""

# Remove all masked images
ask_remove "all masked images" "rm -rf \"${MASKED_DIR}\""

# Remove anomaly regression file
ask_remove "'${ANOMALY_REGRESSION_FILE}'" "rm -f \"${ANOMALY_REGRESSION_FILE}\""


# Cache cleanup
ask_remove "Python cache" "rm -rf \"${SCRIPTS_DIR}\"/__pycache__"
ask_remove "Hugging Face cache" "rm -rf \"${HOME}\"/.cache/huggingface"


echo "Cleanup completed."
# Give execution permissions
# chmod +x cleaner.sh

# Run the script
# ./cleaner.sh