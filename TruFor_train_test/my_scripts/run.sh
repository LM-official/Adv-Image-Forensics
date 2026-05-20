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
ask_execute "dataset filtering" "python \"${SCRIPTS_DIR}/filter.py\""

# Run TruFor and update dataset
ask_execute "TruFor and update dataset" "python \"${SCRIPTS_DIR}/Trufor.py\""

# Performance report
ask_execute "generate truth file and performance report" "python \"${SCRIPTS_DIR}/performance.py\" --generate_truth"
ask_execute "performance report" "python \"${SCRIPTS_DIR}/performance.py\""

# Correlation and regression
ask_execute "correlation and regression" "python \"${SCRIPTS_DIR}/correlation_regression.py\""

# Masking image manipulation
#ask_execute "masking manipulated areas" "python masking.py"

# Generate description
#ask_execute "image description" "python image_to_text.py"

# Multimodal regression
#ask_execute "multimodal regression" "python multimodal_regression.py --metric \"realismScore\""


# Deactivate the python virtual environment
echo " - Deactivating python virtual environment..."
echo " - Python virtual environment deactivated."


echo "Run completed."
# Give execution permissions
# chmod +x run.sh

# Run the script
# ./run.sh