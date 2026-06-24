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
ask_execute "TruFor and update dataset" "python \"${SCRIPTS_DIR}/trufor.py\""

# Performance report
read "confirm?Run performance report? (y/n): "
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    read "confirm?Generate truth file? (y/n): "
    echo " - Running performance report..."
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        python "${SCRIPTS_DIR}/performance.py" --generate-truth
    else
        python "${SCRIPTS_DIR}/performance.py"
    fi
    echo " - Performance report done."
else
    echo " - Performance report skipped."
fi

# General statistics
ask_execute "general statistics" "python \"${SCRIPTS_DIR}/general_stats.py\""

# Masking - generating descriptions - anomaly regression
ask_execute "anomaly regression" "python \"${SCRIPTS_DIR}/anomaly_regression.py\""


# Deactivate the python virtual environment
echo " - Deactivating python virtual environment..."
echo " - Python virtual environment deactivated."


echo "Run completed."
# Give execution permissions
# chmod +x run.sh

# Run the script
# ./run.sh