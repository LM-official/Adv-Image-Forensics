#!/bin/zsh

# Interrupt the script if any command fails
set -e

# Load environment variables
source config.sh


echo "Starting cleanup of the environment..."


# Remove filtered dataset
ask_remove "filtered dataset" "rm -f \"${TEMP_DIR}/${FILTERED_DATASET_FILE}\""

# Remove all images in the input directory
ask_remove "all images in input directory" "rm -rf \"${IMG_DIR}\"/*(N)"

# Remove all .npz files in the output directory
ask_remove "all .npz files in output directory" "rm -rf \"${NPZ_DIR}\"/*.npz(N)"

# Remove truth file
ask_remove "'${TRUTH_FILE}'" "rm -f \"${TEMP_DIR}/${TRUTH_FILE}\""

# Remove the performance file
ask_remove "'${PERFORMANCE_FILE}'" "rm -f \"${TEMP_DIR}/${PERFORMANCE_FILE}\""

# Remove all correlation files
ask_remove "all '${CORRELATION_FILE_PREFIX}' files" "rm -f \"${TEMP_DIR}/${CORRELATION_FILE_PREFIX}\"*.txt(N)"

# Remove all regression files
ask_remove "all '${REGRESSION_FILE_PREFIX}' files" "rm -f \"${TEMP_DIR}/${REGRESSION_FILE_PREFIX}\"*.txt(N)"

# Remove all masked images
ask_remove "all masked images" "rm -f \"${TEMP_DIR}/${MASKED_DIR}\"/*(N)"

# Remove merged metrics file
ask_remove "'${MERGED_METRICS_FILE}'" "rm -f \"${TEMP_DIR}/${MERGED_METRICS_FILE}\""

# Remove multimodal regression files
ask_remove "all '${MULTIMODAL_REGRESSION_FILE_PREFIX}' files" "rm -f \"${TEMP_DIR}/${MULTIMODAL_REGRESSION_FILE_PREFIX}\"*.txt(N)"

# Cache cleanup
ask_remove "Python cache" "rm -rf \"${SCRIPTS_DIR}\"/__pycache__"
ask_remove "Hugging Face cache" "rm -rf \"${HOME}\"/.cache/huggingface"


echo "Cleanup completed."
# Give execution permissions
# chmod +x cleaner.sh

# Run the script
# ./cleaner.sh