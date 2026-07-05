import argparse
from   config          import (TRUFOR_BATCH_SIZE, FILTERED_IMGS_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE,
                               TRUFOR_WAIT, TRUFOR_MAX_PIXELS, TRUFOR_THRESHOLD, TRUTH_FILE,
                               TRUFOR_CONFUSION_MATRIX_FILE, CORRELATION_FILE, REGRESSION_FILE)
import pandas          as     pd
from   pathlib         import Path
from   scipy.stats     import pearsonr, spearmanr
import statsmodels.api as     sm
import subprocess
import sys
import tempfile
import time
from   utils           import load_npzs, update_csv, load_csv, save_report


# =============================================================================
# (1) RUN TRUFOR
# =============================================================================

def _run_trufor_batch(images: Path, npz: Path, trufor_run: Path):
    """
    Runs TruFor on the specified input and output directories.

    Args:
        images:     Directory containing the input images to analyze.
        npz:        Directory where the resulting .npz files will be saved.
        trufor_run: Path to the TruFor entry-point script (test.py).

    Returns:
        True if TruFor completed successfully, False otherwise (invalid paths or subprocess error).
    """

    # Initialize the command
    command = [
        # Current Python interpreter (.venv)
        sys.executable,
        # File
        str(trufor_run),
        "--input",
        str(images),
        "--output",
        str(npz),
        "--max-pixels",
        str(TRUFOR_MAX_PIXELS)
    ]

    try:
        # Run
        subprocess.run(
            command,
            check=True,
            cwd=trufor_run.parent
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")


def run_trufor(images: Path = FILTERED_IMGS_DIR, npz: Path = NPZ_DIR, trufor_run: Path = TRUFOR_RUN,
               batch_size: int = TRUFOR_BATCH_SIZE, wait: int = TRUFOR_WAIT):
    """
    Runs TruFor in chunks of `batch_size` images, writing every output straight into `npz`,
    and waits `wait` seconds between batches to let the machine cool down.
    
    saves each .npz as it finishes, so an interrupted run leaves completed images on disk;
    re-running skips every image whose .npz already exists and resumes from there.

    Each batch is a separate TruFor process, so the model reloads once per batch.

    Args:
        images:     Directory containing the input images to analyze.
        npz:        Directory where the resulting .npz files will be saved.
        trufor_run: Path to the TruFor entry-point script (test.py).
        batch_size: Number of images to send per TruFor invocation.
        wait:       Seconds to wait between batches.

    Returns:
        True if every batch completed successfully, False otherwise.
    """

    if not images.is_dir() or not trufor_run.is_file():
        print("ERROR: Invalid images directory or TruFor run file.")
        return

    # Create the output directory if it doesn't exist
    npz.mkdir(parents=True, exist_ok=True)

    # No batch
    if batch_size <= 0:
        _run_trufor_batch(images, npz, trufor_run)
        return

    # TruFor skips already analyzed images
    image_files   = [p for p in images.iterdir() if p.is_file()]
    len_images    = len(image_files)
    batches = (len_images + batch_size - 1) // batch_size
    print(f"{len_images} images -> {batches} batches of up to {batch_size}.")
    print(f"Waiting {wait} seconds before each next batch")
    for i, start in enumerate(range(0, len_images, batch_size), start=1):
        batch = image_files[start:start + batch_size]
        print(f"Processing batch {i}/{batches}...")

        # TruFor --input takes a directory, not a file list, so stage the batch as symlinks in a temp dir;
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for image in batch:
                (staging / image.name).symlink_to(image.absolute())
            _run_trufor_batch(staging, npz, trufor_run)

        # Wait between batches (not after the last one)
        if i < batches and wait > 0:
            time.sleep(wait)

    print("Batches computed successufully.")


# =============================================================================
# (2) GENERATE GROUND TRUTH
# =============================================================================

def generate_truth(dataset: Path = FILTERED_DATASET_FILE, truth: Path = TRUTH_FILE):
    """
    Generates an empty labels CSV template from the dataset.
    The user fills the 'label' column manually (0 = Real, 1 = AI) before running compute_performance.

    Args:
        dataset: Path to the dataset CSV (must contain the 'filename' column).
        truth:   Output path where the empty truth template CSV is saved.

    Returns:
        None.
    """

    print("Generating truth template...")
    # Load dataset
    df = load_csv(dataset, ["filename"])
    if df is None:
        return

    # Create the output directories if they don't exist
    truth.parent.mkdir(parents=True, exist_ok=True)
    # Empty column for 0 = Real, 1 = AI
    df["label"] = ""

    # Save
    df.to_csv(truth, index=False, encoding="utf-8")
    print("Truth template saved successfully.\nFill the 'label' column (0 = Real, 1 = Altered).")


# =============================================================================
# (2) CONFUSION MATRIX
# =============================================================================

def confusion_matrix(dataset: Path = FILTERED_DATASET_FILE, truth: Path = TRUTH_FILE, threshold: float = TRUFOR_THRESHOLD,
                     report: Path = TRUFOR_CONFUSION_MATRIX_FILE):
    """
    Loads the dataset and truth file, classifies each sample using the given threshold,
    then computes and saves accuracy, precision, recall and F1-score.

    Args:
        dataset:   Path to the dataset CSV (must contain trufor_score).
        truth:     Path to the truth CSV with the 'label' column filled (0 = Real, 1 = AI).
        threshold: Decision threshold in [0.0, 1.0]: trufor_score > threshold is predicted as AI.
        report:    Output file path where the performance report is written.

    Returns:
        None.
    """

    if not dataset.is_file() or not (0.0 <= threshold <= 1.0):
        print("ERROR: Invalid input parameters.")
        return

    # The ground truth is optional and created on demand (--generate-truth). Without it there is
    # nothing to score against, so skip the detector confusion matrix cleanly instead of erroring.
    if not truth.is_file():
        print("Skipping detector confusion matrix: no ground-truth file found (run 'trufor.py --generate-truth' to create one).")
        return


    # Load truth
    df = load_csv(truth, ["filename", "label"])
    if df is None:
        return
    # Validate labels must all be 0 or 1
    invalid = ~df["label"].isin([0, 1])
    if invalid.any():
        print("ERROR: Invalid labels found in truth file. All labels must be 0 (Real) or 1 (AI).")
        return


    # Load scores
    df_score = load_csv(dataset, ["filename", "trufor_score"])
    if df_score is None:
        return
    

    # Merge
    df = pd.merge(df, df_score, on="filename", how="inner")
    if df is None or df.empty:
        print("ERROR: No valid data after merging scores and truth.")
        return


    # Compute performance
    print(f"Computing performance with threshold {threshold:.2f}...")

    # Vectorized prediction: 1 if trufor_score > threshold, else 0
    df["prediction"] = (df["trufor_score"] > threshold).astype(int)

    # Compute confusion matrix counters using vectorized operations
    tp = int(((df["label"] == 1) & (df["prediction"] == 1)).sum())   # true positives (manipulated -> correctly identified as manipulated)
    tn = int(((df["label"] == 0) & (df["prediction"] == 0)).sum())   # true negatives (Real -> correctly identified as real)
    fp = int(((df["label"] == 0) & (df["prediction"] == 1)).sum())   # false positives (Real -> mistakenly flagged as manipulated)
    fn = int(((df["label"] == 1) & (df["prediction"] == 0)).sum())   # false negatives (manipulated -> mistakenly passed as real)
    
    # Build fp/fn lists
    fp_list = df.loc[(df["label"] == 0) & (df["prediction"] == 1), ["filename", "trufor_score"]].values.tolist()
    fn_list = df.loc[(df["label"] == 1) & (df["prediction"] == 0), ["filename", "trufor_score"]].values.tolist()

    """
    accuracy:
    Measures the overall correctness of the model.
    It represents the ratio of correctly predicted observations
    (both true positives and true negatives) to the total number of observations.
    It can be misleading if the dataset is imbalanced.
    """
    accuracy = (tp + tn) / len(df)

    """
    precision:
    Measures the reliability of positive predictions.
    'Out of all the instances the model flagged as positive, how many were actually positive?'
    A high precision indicates a low rate of false positives (false alarms).
    """
    # tp + fp > 0 check to avoid division by zero when no positive predictions were made
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    """
    recall (sensitivity):
    Measures the model's ability to discover all actual positive instances.
    'Out of all the true positive cases in the dataset, how many did the model successfully find?'
    A high recall indicates a low rate of false negatives (missed detections).
    """
    # tp + fn > 0 check to avoid division by zero when no actual positives are present
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  
    """
    f1-score:
    The harmonic mean of precision and recall.
    Since precision and recall often have an inverse relationship
    (improving one can worsen the other), the f1-score provides a single, balanced metric.
    It is widely considered the most robust and reliable indicator of a model"s performance,
    especially when dealing with imbalanced data.
    """
    # precision + recall > 0 check to avoid division by zero when both precision and recall are zero
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("Confusion matrix computed successfully.")

    # Report
    text = (
        f"{'=' * 77}\n"
        "TRUFOR CONFUSION MATRIX\n"
        f"{'=' * 77}\n"
        f"Total files analyzed: {len(df)}\n"
        f"Threshold: {threshold:.2f}\n\n"
        f"Accuracy  :\t{accuracy:.2%}\t(percentage of correct predictions)\n"
        f"Precision :\t{precision:.2%}\t(reliability when it claims 'It is AI')\n"
        f"Recall    :\t{recall:.2%}\t(sensitivity, ability to detect all AI images)\n"
        f"F1-score  :\t{f1_score:.4f}\t(final harmonic mean)\n\n"
        "Confusion matrix:\n"
        f"True positives  :\t{tp}\n"
        f"True negatives  :\t{tn}\n"
        f"False positives :\t{fp}\t(real mistaken for AI)\n"
        f"False negatives :\t{fn}\t(AI mistaken for real)\n"
    )

    lines = []
    if fp_list:
        lines.append("\nList of false positives:")
        for filename, score in fp_list:
            lines.append(f"  '{filename}', score: {score:.4f}")
    if fn_list:
        lines.append("\nList of false negatives:")
        for filename, score in fn_list:
            lines.append(f"  '{filename}', score: {score:.4f}")
    if lines:
        text += "\n".join(lines)

    # Save
    save_report(text, report)


# =============================================================================
# (3) CORRELATION TRUFOR SCORE - CROWDTANGLE SCORE
# =============================================================================

def correlation(dataset: Path = FILTERED_DATASET_FILE, report: Path = CORRELATION_FILE):
    """
    Calculates Pearson and Spearman correlations between TruFor score and Crowdtangle score.

    Args:
        dataset: Path to the dataset CSV (must contain trufor_score and crowdtangle_score).
        report:  Output file path where the correlation report will be written.

    Returns:
        None.
    """

    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return
    # Pearson requires at least 2 unique values in each variable to compute correlation, otherwise it will return NaN.
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points after dropping NaN values or insufficient variability in the data.")
        return

    
    # Correlation
    print("Calculating correlation between TruFor scores and Crowdtangle scores...")
    
    """
    Pearson Correlation: Measures linearity. It asks, 'Do these data points form a perfectly straight line?'
    It calculates using exact numbers, making it very sensitive to extreme outliers.
    """
    # Pearson correlation (measures linear relationship)
    pearson_corr, p_value_p = pearsonr(df["trufor_score"], df["crowdtangle_score"])
    
    """
    Spearman Correlation: Measures trend (monotonicity). It asks, 'As X goes up, does Y consistently go up (or down), regardless of the curve?'
    Instead of exact numbers, it looks at the rank of the values (1st, 2nd, 3rd...), making it highly resistant to outliers and non-linear shapes.
    """
    # Spearman correlation (measures monotonic relationship / rank-based)
    spearman_corr, p_value_s = spearmanr(df["trufor_score"], df["crowdtangle_score"])
    print("Correlation calculated successfully.")
    

    # Report
    text = (
        f"{'=' * 77}\n"
        "CORRELATION TRUFOR SCORE - CROWDTANGLE SCORE\n"
        f"{'=' * 77}\n"
        f"Rows analyzed: {len(df)}\n"
        f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value_p:.3e})\n"
        f"Spearman correlation: {spearman_corr:.3f} (p-value: {p_value_s:.3e})"
    )

    # Save
    save_report(text, report)


# =============================================================================
# (3) REGRESSION TRUFOR SCORE - CROWDTANGLE SCORE
# =============================================================================

def regression(dataset: Path = FILTERED_DATASET_FILE, report: Path = REGRESSION_FILE):
    """
    Calculates Ordinary Least Squares (OLS) regression between TruFor score and Crowdtangle score.

    Args:
        dataset: Path to the dataset CSV (must contain trufor_score and crowdtangle_score).
        report:  Output file path where the regression report will be written.

    Returns:
        None.
    """
    
    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return


    # Regression
    print("Calculating regression between TruFor Scores and Crowdtangle scores...")
    
    # Independent variable (X): What we use to predict
    X = df["trufor_score"]
    # In statsmodels, we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
    X = sm.add_constant(X)
    # Dependent variable (y): What we want to predict
    Y = df["crowdtangle_score"]

    """
    Ordinary Least Squares (OLS) regression is a statistical method used to model the relationship between a dependent variable
    (Y, e.g., CrowdTangle score) and one independent variable (X, e.g., TruFor Score).
    1. R-squared: the coefficient of determination.
       It measures the proportion of variance in the dependent variable that is predictable from the independent variable.
       Values range from 0 to 1 (0% to 100%). Explains how much of the 'variability' of the engagement score is justified by the TruFor score.
       If it is 0.01, it means that TruFor explains only 1% of the variance in engagement.
    2. Adj. R-squared: a version of R-squared adjusted for the number of predictors in the model.
    3. Coefficients (coef): represent the mean change in the dependent variable for one unit of change in the independent variable while
       holding other predictors constant.
       It tells you the 'slope'.
       If the TruFor coefficient is negative, it means that as the anomaly score increases (more 'fake'), the engagement score decreases.
    4. P>|t| (p-value): tests the null hypothesis that the coefficient is equal to zero (no effect).
       A low p-value (< 0.05) indicates that the predictor is statistically significant.
    5. F-statistic: evaluates the overall significance of the model.
    """
    # Define and fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X).fit()
    print("Regression calculated successfully.")
    

    # Report
    text = (
        f"{'=' * 77}\n"
        "REGRESSION TRUFOR SCORE - CROWDTANGLE SCORE\n"
        f"{'=' * 77}\n"
        f"{model.summary().as_text()}"
    )

    # Save
    save_report(text, report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    (1) Runs TruFor on the input images and updates the dataset CSV with the resulting scores.
    (2) Optionally generates the truth template and then computes the confusion matrix.
    (3) Runs correlation and OLS regression between TruFor scores and Crowdtangle scores.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor, update dataset with scores and compute confusion matrix")
    # Flags, paths come from config.py (the project backbone)
    # Run trufor
    parser.add_argument("--batch-size", type=int, default=TRUFOR_BATCH_SIZE, help=f"Images per TruFor invocation. Default: {TRUFOR_BATCH_SIZE}")
    parser.add_argument("--wait",       type=int, default=TRUFOR_WAIT,       help=f"Seconds to wait between batches. Default: {TRUFOR_WAIT}")
    
    # Confusion matrix
    parser.add_argument("--threshold",      type=float, default=TRUFOR_THRESHOLD, help=f"Threshold for classifying an image as AI. Range: 0.0 to 1.0. Default: '{TRUFOR_THRESHOLD}'.")
    parser.add_argument("--generate-truth", action="store_true",                  help="Generate an empty truth template CSV before computing performance.")
    args = parser.parse_args()
    

    # =============================================================================
    # (1) RUN TRUFOR AND UPDATE DATASET
    # =============================================================================
    # Run TruFor in batches (paths default from config; batch size and wait come from the flags)
    run_trufor(batch_size=args.batch_size, wait=args.wait)

    # Divide the update in batched is useless because the bottleneck is running TruFor
    # Run TruFor on a big dataset is very expensive, so the dataset should be small so divide in batches is not necessary

    # Load TruFor scores from all .npz files
    scores = load_npzs(NPZ_DIR, {"score"})
    if scores is None:
        return
    # Update dataset
    update_csv(FILTERED_DATASET_FILE, scores, rename={"score": "trufor_score"})


    # =============================================================================
    # (2) COMPUTE CONFUSION MATRIX
    # =============================================================================
    if args.generate_truth:
        # Generate truth template
        generate_truth()

        # User confirms they have filled the truth file
        answer = input("Truth file filled with labels? (y/n): ").strip().lower()
        if answer != "y":
            return

    # Compute performance (paths default from config; threshold comes from the flag)
    confusion_matrix(threshold=args.threshold)


    # =============================================================================
    # (3) CORRELATION AND REGRESSION TRUFOR SCORE - CROWDTANGLE SCORE
    # =============================================================================
    correlation()
    regression()

if __name__ == "__main__":
    main()