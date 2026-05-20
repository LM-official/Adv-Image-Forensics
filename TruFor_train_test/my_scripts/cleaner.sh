#!/bin/zsh

# Interrupt the script if any command fails
set -e

# Load environment variables
source config.sh


echo "Starting cleanup of the environment..."


# Remove filtered dataset
ask_remove "'${FILTERED_DATASET_FILE}'" "rm -f \"${FILTERED_DATASET_FILE}\""

# Remove all images in the input directory
ask_remove "all images in input directory" "rm -rf \"${IMG_DIR}\"/*(N)"

# Remove all .npz files in the output directory
ask_remove "all .npz files in output directory" "rm -rf \"${NPZ_DIR}\"/*.npz(N)"

# Remove truth file
ask_remove "'${TRUTH_FILE}'" "rm -f \"${TRUTH_FILE}\""

# Remove performance file
ask_remove "'${PERFORMANCE_FILE}'" "rm -f \"${PERFORMANCE_FILE}\""

# Remove correlation file
ask_remove "'${CORRELATION_FILE}'" "rm -f \"${CORRELATION_FILE}\""

# Remove regression file
ask_remove "'${REGRESSION_FILE}'" "rm -f \"${REGRESSION_FILE}\""

# Remove all masked images
#ask_remove "all masked images" "rm -f \"${TEMP_DIR}/${MASKED_DIR}\"/*(N)"

# Remove merged metrics file
#ask_remove "'${MERGED_METRICS_FILE}'" "rm -f \"${TEMP_DIR}/${MERGED_METRICS_FILE}\""

# Remove multimodal regression files
#ask_remove "all '${MULTIMODAL_REGRESSION_FILE_PREFIX}' files" "rm -f \"${TEMP_DIR}/${MULTIMODAL_REGRESSION_FILE_PREFIX}\"*.txt(N)"

# Cache cleanup
#ask_remove "Python cache" "rm -rf \"${SCRIPTS_DIR}\"/__pycache__"
#ask_remove "Hugging Face cache" "rm -rf \"${HOME}\"/.cache/huggingface"


echo "Cleanup completed."
# Give execution permissions
# chmod +x cleaner.sh

# Run the script
# ./cleaner.sh