import argparse
from   config          import (TRUFOR_BATCH_SIZE, FILTERED_IMGS_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE,
                               TRUFOR_WAIT, TRUFOR_MAX_PIXELS, TRUFOR_THRESHOLD, TRUTH_FILE,
                               TRUFOR_CONFUSION_MATRIX_FILE, CORRELATION_FILE, REGRESSION_FILE,
                               GLM_FILE, ENGAGEMENT_DEAD_ZONE)
import numpy           as     np
import pandas          as     pd
from   pathlib         import Path
from   scipy.stats     import pearsonr, spearmanr
from   sklearn.metrics import roc_auc_score
import statsmodels.api as     sm
import subprocess
import sys
import tempfile
import time
from   utils           import load_npzs, update_csv, load_csv, save_report, binarize_engagement


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

    print("Running TruFor on the images...")

    if not images.is_dir() or not trufor_run.is_file():
        print("ERROR: Invalid images directory or TruFor run file.")
        return

    # Create the output directory if it doesn't exist
    npz.mkdir(parents=True, exist_ok=True)

    # No batch
    if batch_size <= 0:
        _run_trufor_batch(images, npz, trufor_run)
        print("TruFor run successfully.")
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

    print("TruFor batches computed successfully.")


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

    print(f"Computing performance...")

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
    len_df = len(df)


    # Compute performance
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
    accuracy = (tp + tn) / len_df

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

    """
    majority-class baseline:
    The accuracy reached by always answering with whichever label is more frequent in the truth file, without looking the images.
    Accuracy is not interpretable on its own: on an unbalanced ground truth a detector that blindly answers the majority class
    already scores the share of that class, so only the excess over this line is evidence that the detector works.
    When accuracy sits at the baseline it has shown nothing.
    """
    positives = tp + fn # images actually manipulated
    negatives = tn + fp # images actually real
    baseline  = max(positives, negatives) / len_df if len_df > 0 else 0.0

    """
    ROC AUC:
    Probability that a randomly chosen manipulated image scores higher than a randomly chosen real one.
    The four metrics above read `prediction`, so thresholding has already collapsed the score to 0/1 and
    two images on the same side of the cut become indistinguishable;
    this reads `trufor_score` directly and keeps the whole ranking.
    It therefore answers a different question - does the score separate the two classes at all? -
    independently of whether the chosen threshold cuts it in a good place.
    An AUC of one half means chance, 1.0 a perfect ranking.
    """
    # Needs both classes present, otherwise the ranking is undefined
    auc = roc_auc_score(df["label"], df["trufor_score"]) if 0 < positives < len_df else float("nan")

    # Threshold sweep: the default cut is inherited from TruFor's own evaluation, it is not tuned on this dataset,
    # so it is a choice and not a result.
    # Recomputing the metrics across a range of cuts separates what the detector can do from what the particular cut happens to give.
    sweep = []
    for cut in np.arange(0.20, 0.55, 0.05):
        prediction = (df["trufor_score"] > cut).astype(int)
        s_tp = int(((df["label"] == 1) & (prediction == 1)).sum())
        s_tn = int(((df["label"] == 0) & (prediction == 0)).sum())
        s_fp = int(((df["label"] == 0) & (prediction == 1)).sum())
        s_fn = int(((df["label"] == 1) & (prediction == 0)).sum())
        s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_recall    = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1        = 2 * ((s_precision * s_recall) / (s_precision + s_recall)) if (s_precision + s_recall) > 0 else 0.0
        sweep.append((cut, (s_tp + s_tn) / len_df, s_precision, s_recall, s_f1))

    # Report
    text = (
        f"{'=' * 77}\n"
        "TRUFOR CONFUSION MATRIX\n"
        f"{'=' * 77}\n"
        f"Total files analyzed: {len_df}\n"
        f"Truth balance: {positives} manipulated / {negatives} real\n"
        f"Threshold: {threshold:.2f}\n\n"
        f"ROC AUC   :\t{auc:.4f}\t(threshold-independent: does the score separate the classes at all? one half = chance)\n\n"
        f"At threshold {threshold:.2f}:\n"
        f"Accuracy  :\t{accuracy:.2%}\t(percentage of correct predictions)\n"
        f"Baseline  :\t{baseline:.2%}\t(always answering the majority class - accuracy is only meaningful above this)\n"
        f"Precision :\t{precision:.2%}\t(reliability when it claims 'It is AI')\n"
        f"Recall    :\t{recall:.2%}\t(sensitivity, ability to detect all AI images)\n"
        f"F1-score  :\t{f1_score:.4f}\t(final harmonic mean)\n\n"
        "Threshold sweep (the cut above is TruFor's own default, it is not tuned on this dataset):\n"
        f"  {'cut':>5}  {'accuracy':>9}  {'precision':>10}  {'recall':>8}  {'F1':>7}\n"
        f"  {'-' * 46}\n"
        + "".join(f"  {cut:>5.2f}  {acc:>8.2%}  {prec:>9.2%}  {rec:>7.2%}  {f1:>7.4f}"
                  f"{'   <- reported above' if abs(cut - threshold) < 1e-9 else ''}\n"
                  for cut, acc, prec, rec, f1 in sweep)
        + "\nConfusion matrix:\n"
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
    print("Confusion matrix computed successfully.")


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

    print("Calculating correlation between TruFor scores and Crowdtangle scores...")

    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return
    # Pearson requires at least 2 unique values in each variable to compute correlation, otherwise it will return NaN.
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points after dropping NaN values or insufficient variability in the data.")
        return

    
    # Correlation
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
    print("Correlation computed successfully.")


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

    print("Calculating regression between TruFor Scores and Crowdtangle scores...")
    
    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return


    # Regression
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


    # Report
    text = (
        f"{'=' * 77}\n"
        "REGRESSION TRUFOR SCORE - CROWDTANGLE SCORE\n"
        f"{'=' * 77}\n"
        f"{model.summary().as_text()}"
    )

    # Save
    save_report(text, report)
    print("Regression computed successfully.")


# =============================================================================
# (3) GLM TRUFOR SCORE - CROWDTANGLE SCORE
# =============================================================================

def glm(dataset: Path = FILTERED_DATASET_FILE, report: Path = GLM_FILE, dead_zone: float = ENGAGEMENT_DEAD_ZONE):
    """
    Fits a binomial GLM (logit link) predicting the binary engagement label from the TruFor score alone.

    Args:
        dataset:   Path to the dataset CSV (must contain trufor_score and crowdtangle_score).
        report:    Output file path where the GLM report will be written.
        dead_zone: Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising.
                   negatives are clamped to 0. 0 disables the cut and keeps every post.

    Returns:
        None.
    """

    print("Fitting binomial GLM (logit link) between TruFor scores and Crowdtangle scores...")
    
    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return

    # clamp to 0
    if dead_zone < 0:
        dead_zone = 0

    # Dead-zone cut + binarisation
    total  = len(df)
    result = binarize_engagement(df, dead_zone)
    if result is None:
        return
    df, y  = result


    # GLM
    try:
        # add_constant adds the intercept column; Binomial() defaults to the logit link
        model     = sm.GLM(y, sm.add_constant(df["trufor_score"]), family=sm.families.Binomial()).fit()
        converged = model.converged and np.isfinite(model.llf)
    except Exception as e:
        print(f"ERROR: during binomial GLM: {e}")
        return

    if not converged:
        print("ERROR: the GLM did not converge, its coefficients and p-values are not interpretable.")
        return

    # McFadden pseudo-R2 from the deviance: deviance = -2*LL, so 1 - dev/null_dev = 1 - LL/LL_null
    pseudo_r2 = 1 - model.deviance / model.null_deviance


    # Report
    text = (
        f"{'=' * 77}\n"
        "GLM TRUFOR SCORE - CROWDTANGLE SCORE\n"
        f"{'=' * 77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len(y)} of {total} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len(y)})\n"
        f"In-sample McFadden pseudo-R2: {pseudo_r2:.4f}\n\n"
        f"{model.summary().as_text()}"
    )

    # Save
    save_report(text, report)
    print("GLM fitted successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    (1) Runs TruFor on the input images and updates the dataset CSV with the resulting scores.
    (2) Optionally generates the truth template and then computes the confusion matrix.
    (3) Runs correlation, OLS regression and a binomial GLM between TruFor scores and Crowdtangle scores.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor, update dataset with scores, compute confusion matrix and correlation/regression/GLM against engagement")
    # Flags, paths come from config.py (the project backbone)
    # Run trufor
    parser.add_argument("--batch-size", type=int, default=TRUFOR_BATCH_SIZE, help=f"Images per TruFor invocation. Default: {TRUFOR_BATCH_SIZE}")
    parser.add_argument("--wait",       type=int, default=TRUFOR_WAIT,       help=f"Seconds to wait between batches. Default: {TRUFOR_WAIT}")
    
    # Confusion matrix
    parser.add_argument("--threshold",      type=float, default=TRUFOR_THRESHOLD, help=f"Threshold for classifying an image as AI. Range: 0.0 to 1.0. Default: '{TRUFOR_THRESHOLD}'.")
    parser.add_argument("--generate-truth", action="store_true",                  help="Generate an empty truth template CSV before computing performance.")

    # GLM
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
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
    # (3) CORRELATION, REGRESSION AND GLM TRUFOR SCORE - CROWDTANGLE SCORE
    # =============================================================================
    correlation()
    regression()
    glm(dead_zone=args.dead_zone)

if __name__ == "__main__":
    main()