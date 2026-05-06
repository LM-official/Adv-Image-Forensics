#!/bin/zsh

# interrupt the script if any command fails
set -e

# load environment variables
source config.sh


echo "Starting..."
# activate the python virtual environment
echo " - Activating python virtual environment..."
source "${VENV_PATH}"
echo " - Python virtual environment activated."


# move to TruFor directory
cd "${TRUFOR_ROOT}"

# run TruFor
read "confirm?Run TruFor? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running TruFor..."
    python test.py --input "${IMG_DIR}" --output "${NPZ_DIR}"
    # wait
    echo " - TruFor execution completed."
else
    echo " - TruFor execution skipped."
fi


# move to scripts directory
cd "${SCRIPTS_DIR}"


# generate truth
read "confirm?Generate '${TRUTH_FILE}'? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Generating truth file..."
    python truth_generator.py
    # wait
    echo " - '${TRUTH_FILE}' generated."
else
    echo " - '${TRUTH_FILE}' generation skipped."
fi


# performance report
read "confirm?Run performance evaluation? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running performance evaluation..."
    python performance.py
    # wait
    echo " - Performance evaluation completed."
else
    echo " - Performance evaluation skipped."
fi


# merged metrics
read "confirm?Run merge metrics? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running merge metrics..."
    python merge_metrics.py --dataset "${CSV_ASSIGNMENT}" "${CSV_IMAGE}"
    # wait
    echo " - Merge metrics completed."
else
    echo " - Merge metrics skipped."
fi


# correlation and regression
read "confirm?Run correlation and regression evaluation? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running correlation and regression..."
    METRIC="realismScore"
    python correlation_regression.py --metric "${METRIC}"
    # wait
    echo " - Correlation and regression evaluation completed."
else
    echo " - Correlation and regression evaluation skipped."
fi


# masking image manipulation
read "confirm?Run masking manipulated areas? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running masking manipulated areas..."
    python masking.py
    # wait
    echo " - Masking manipulated areas completed."
else
    echo " - Masking manipulated areas skipped."
fi


# generate description
read "confirm?Run image description? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running image description..."
    python image_to_text.py
    # wait
    echo " - Image description completed."
else
    echo " - Image description skipped."
fi


# multimodal regression
read "confirm?Run multimodal regression? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo " - Running multimodal regression..."
    METRIC="realismScore"
    python multimodal_regression.py --metric "${METRIC}"
    # wait
    echo " - Multimodal regression completed."
else
    echo " - Multimodal regression skipped."
fi


echo "Run completed."
# give execution permissions
# chmod +x run.sh

# run the script
# ./run.sh

# visualize a single image
# source /Users/lorenzo/trufor_py_env/bin/activate
# cd /Users/lorenzo/Tirocinio/TruFor/TruFor_train_test
# python visualize.py --image images/image.png --output output/image.png.npz