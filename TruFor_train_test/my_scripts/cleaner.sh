#!/bin/zsh

# interrupt the script if any command fails
set -e

# load environment variables
source config.sh


echo "Starting cleanup of the environment..."


# remove all images in the input directory
read "confirm?Delete all images in '${IMG_DIR}' directory? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Cleaning up input directory..."
    rm -f "${IMG_DIR}"/*(N)
    echo " - '${IMG_DIR}' directory cleared."
else
    echo " - '${IMG_DIR}' directory not cleared."
fi


# remove all .npz files in the output directory
read "confirm?Delete all .npz files? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Cleaning up output directory..."
    rm -f "${NPZ_DIR}"/*.npz(N)
    echo " - '${NPZ_DIR}' directory cleared."
else
    echo " - '${NPZ_DIR}' directory not cleared."
fi


# move to temp directory
cd "${TEMP_DIR}"


# remove truth file
read "confirm?Delete '${TRUTH_FILE}'? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${TRUTH_FILE}'..."
    rm -f "${TRUTH_FILE}"
    echo " - '${TRUTH_FILE}' removed."
else
    echo " - '${TRUTH_FILE}' not removed."
fi


# remove the performance file
read "confirm?Delete '${PERFORMANCE_FILE}'? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${PERFORMANCE_FILE}'..."
    rm -f "${PERFORMANCE_FILE}"
    echo " - '${PERFORMANCE_FILE}' removed."
else
    echo " - '${PERFORMANCE_FILE}' not removed."
fi


# remove all correlation files
read "confirm?Delete all '${CORRELATION_FILE_PREFIX}' files? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${CORRELATION_FILE_PREFIX}'..."
    rm -f "${CORRELATION_FILE_PREFIX}"*.txt(N)
    echo " - '${CORRELATION_FILE_PREFIX}' removed."
else
    echo " - '${CORRELATION_FILE_PREFIX}' not removed."
fi


# remove all regression files
read "confirm?Delete all '${REGRESSION_FILE_PREFIX}' files? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${REGRESSION_FILE_PREFIX}'..."
    rm -f "${REGRESSION_FILE_PREFIX}"*.txt(N)
    echo " - '${REGRESSION_FILE_PREFIX}' removed."
else
    echo " - '${REGRESSION_FILE_PREFIX}' not removed."
fi


# remove all masked images
read "confirm?Delete all masked images? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing masked images..."
    rm -f "${MASKED_DIR}"/*(N)
    echo " - Masked images removed."
else
    echo " - Masked images not removed'."
fi


# remove merged metrics file
read "confirm?Delete '${MERGED_METRICS_FILE}'? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${MERGED_METRICS_FILE}'..."
    rm -f "${MERGED_METRICS_FILE}"
    echo " - '${MERGED_METRICS_FILE}' removed."
else
    echo " - '${MERGED_METRICS_FILE}' not removed."
fi


# remove multimodal regression files
read "confirm?Delete all '${MULTIMODAL_REGRESSION_FILE_PREFIX}' files? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Removing '${MULTIMODAL_REGRESSION_FILE_PREFIX}'..."
    rm -f "${MULTIMODAL_REGRESSION_FILE_PREFIX}"*.txt(N)
    echo " - '${MULTIMODAL_REGRESSION_FILE_PREFIX}' removed."
else
    echo " - '${MULTIMODAL_REGRESSION_FILE_PREFIX}' not removed."
fi


# cache cleanup
read "confirm?Clear cache? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Clearing cache..."
    rm -rf __pycache__(N)
    rm -rf /.cache/huggingface(N)
    echo " - Cache cleared."
else
    echo " - Cache not cleared."
fi


echo "Cleanup completed."
# give execution permissions
# chmod +x cleaner.sh

# run the script
# ./cleaner.sh