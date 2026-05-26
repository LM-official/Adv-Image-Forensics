import argparse
from   config  import FILTERED_DATASET_FILE, PERFORMANCE_THRESHOLD, TRUTH_FILE, PERFORMANCE_FILE
import pandas  as     pd
from   pathlib import Path
from   utils   import load_csv, save_report


def generate_truth(dataset: Path, truth: Path) -> None:
    """
    Generates an empty labels CSV template from the dataset.
    The user fills the 'label' column manually (0 = Real, 1 = AI) before running compute_performance.

    Args:
        dataset: Path to the dataset CSV (must contain the 'filename' column).
        truth:   Output path where the empty truth template CSV is saved.

    Returns:
        None.
    """

    if not dataset.is_file():
        print(f"ERROR: dataset '{dataset}' not found.")
        return None


    print("Generating truth template...")
    # Load dataset
    df = load_csv(dataset, ["filename"])
    if df is None:
        return None

    # Create the output directories if they don't exist
    truth.parent.mkdir(parents=True, exist_ok=True)
    # Empty column for 0 = Real, 1 = AI
    df["label"] = ""

    # Save
    df.to_csv(truth, index=False, encoding="utf-8")
    print(f"Truth template saved to '{truth}'.\nFill the 'label' column (0 = Real, 1 = AI).")
    return None


def compute_performance(dataset: Path, truth: Path, threshold: float, report: Path) -> None:
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
        return None


    # Load truth
    df = load_csv(truth, ["filename", "label"])
    if df is None:
        return None
    # Validate labels must all be 0 or 1
    invalid = ~df["label"].isin([0, 1])
    if invalid.any():
        print(f"ERROR: Invalid labels found in truth file '{truth}'. All labels must be 0 (Real) or 1 (AI).")
        return None


    # Load scores
    df_score = load_csv(dataset, ["filename", "trufor_score"])
    if df_score is None:
        return None
    

    # Merge
    df = pd.merge(df, df_score, on="filename", how="inner")
    if df is None or df.empty:
        print("ERROR: No valid data after merging scores and truth.")
        return None


    # Compute performance
    print(f"Computing performance with threshold {threshold:.2f}...")

    # Vectorized prediction: 1 if trufor_score > threshold, else 0
    df["prediction"] = (df["trufor_score"] > threshold).astype(int)

    # Compute confusion matrix counters using vectorized operations
    tp = int(((df["label"] == 1) & (df["prediction"] == 1)).sum())   # true positives (AI -> correctly identified as AI)
    tn = int(((df["label"] == 0) & (df["prediction"] == 0)).sum())   # true negatives (Real -> correctly identified as real)
    fp = int(((df["label"] == 0) & (df["prediction"] == 1)).sum())   # false positives (Real -> mistakenly flagged as AI)
    fn = int(((df["label"] == 1) & (df["prediction"] == 0)).sum())   # false negatives (AI -> mistakenly passed as real)
    
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

    print("Performance metrics computed successfully.")

    # Report
    text = (
        f"TruFor performance report\n"
        f"Total files analyzed: {len(df)}\n"
        f"Threshold: {threshold:.2f}\n\n"
        f"Accuracy  :\t{accuracy:.2%}\t(percentage of correct predictions)\n"
        f"Precision :\t{precision:.2%}\t(reliability when it claims 'It is AI')\n"
        f"Recall    :\t{recall:.2%}\t(sensitivity, ability to detect all AI images)\n"
        f"F1-score  :\t{f1_score:.4f}\t(final harmonic mean)\n\n"
        f"Confusion matrix:\n"
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
    return None


def main() -> None:
    """
    Optionally generates the truth template and then computes the performance metrics of the TruFor results.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Compute performance metrics of TruFor results.")

    # Flags
    parser.add_argument("--dataset",        type=Path,  default=FILTERED_DATASET_FILE, help=f"Path to the dataset file. Default: '{FILTERED_DATASET_FILE}'")
    parser.add_argument("--threshold",      type=float, default=PERFORMANCE_THRESHOLD, help=f"Threshold for classifying an image as AI. Range: 0.0 to 1.0. Default: '{PERFORMANCE_THRESHOLD}'.")
    parser.add_argument("--truth",          type=Path,  default=TRUTH_FILE,            help=f"Path to the truth CSV file (filename, label). Default: '{TRUTH_FILE}'")
    parser.add_argument("--generate-truth", action="store_true",                       help=f"Generate an empty truth template CSV. Default: '{TRUTH_FILE}'")
    parser.add_argument("--performance",    type=Path,  default=PERFORMANCE_FILE,      help=f"Output performance file path. Default: '{PERFORMANCE_FILE}'")
    args = parser.parse_args()


    if args.generate_truth:
        # Generate truth template
        generate_truth(args.dataset, args.truth)

        # Pause until the user confirms they have filled the truth file
        while True:
            answer = input("Truth file filled with labels? (y/n): ").strip().lower()
            if answer == "y":
                break


    # Compute performance
    compute_performance(args.dataset, args.truth, args.threshold, args.performance)

if __name__ == "__main__":
    main()