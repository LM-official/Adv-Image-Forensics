import  argparse
from    config      import  TEMP_DIR_PATH, TRUTH_FILE_NAME, NPZ_DIR_PATH, PERFORMANCE_THRESHOLD, PERFORMANCE_FILE_NAME
import  csv
import  numpy       as      np
import  pandas      as      pd
from    pathlib     import  Path


def load_truth_csv(truth_path: Path) -> dict[str, int] | None:
    """
    Loads labels dictionary from the csv file.
    """

    print(f"Loading '{truth_path}'...")


    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    truth = {}
    try:
        with open(truth_path, mode = "r", encoding = "utf-8") as file:
            reader = csv.reader(file)

            # Skip header
            next(reader, None)

            for row in reader:
                # Only process lines that have exactly 2 elements, separated by ','
                if not row or len(row) != 2:
                    print(f"WARNING: Invalid row '{row}'. Skipping the entry.")
                    continue

                # EAFP = Easier to Ask for Forgiveness than Permission, so we try to cast the label to int and catch any exceptions
                try:
                    # Int() converts and ignore whitespaces
                    label = int(row[1])
                    if label not in (0, 1):
                        raise ValueError('Label must be 0 or 1.')
                    truth[row[0].strip()] = label
                except ValueError:
                    print(f"WARNING: Invalid row '{row}'. Skipping the entry.")
                    continue

    except Exception as e:
        print(f"ERROR: reading '{truth_path}': {e}")

    print(f"'{truth_path}' loaded successfully. Total entries: {len(truth)}")
    return truth


def scores_to_dataframe(truth: dict[str, int], npz_dir_path: Path) -> pd.DataFrame | None:
    """
    Extracts scores from .npz files and creates a DataFrame with columns: 'filename', 'score', 'label'.
    """

    print(f"Extracting scores from .npz files in '{npz_dir_path}'...")

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if not truth or not npz_dir_path.is_dir():
        print("ERROR: Invalid input parameters.")
        return
    

    # Get scores from .npz files and create a DataFrame
    scores = []
    # Analyze each .npz file in the dir
    for file in npz_dir_path.glob("*.npz"):
        # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
        filename, filePath = Path(file.stem).stem, file.resolve()
        # Only process files in truth dictionary
        if filename not in truth:
            print(f"WARNING: '{filename}' not found in truth csv. Skipping the entry.")
            continue

        # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
        try:
            # Use context manager to ensure the file is properly closed after reading
            with np.load(filePath) as data:
                scores.append({"filename": filename, "score": float(data["score"])})
        except Exception as e:
            print(f"ERROR: reading '{filePath}': {e}")

            continue
            
    if not scores:
        print(f"ERROR: No valid .npz files found in '{npz_dir_path}'.")
        return


    # Convert truth dictionary to DataFrame
    truth_df = pd.DataFrame(list(truth.items()), columns = ["filename", "label"])
    # Convert scores list to DataFrame and merge with truth DataFrame on "filename"
    scores_df = pd.DataFrame(scores)
    df = pd.merge(truth_df, scores_df, on = "filename", how = "inner")

    print(f"Scores extracted successfully. Total entries: {len(df)}")
    return df


def compute_performance(df: pd.DataFrame, threshold: float) -> dict | None:
    """
    Computes performance metrics (accuracy, precision, recall, f1-score) of the TruFor results using Pandas for vectorized operations
    """
    
    print(f"Computing performance metrics with threshold {threshold:.2f}...")

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if df is None or df.empty or not (0.0 <= threshold <= 1.0):
        print("ERROR: Invalid input parameters.")
        return


    # Apply threshold to the entire column
    df["prediction"] = (df["score"] > threshold).astype(int)

    # Compute confusion matrix counters using vectorized operations
    tp = ((df["label"] == 1) & (df["prediction"] == 1)).sum()   # true positives (AI -> correctly identified as AI)
    tn = ((df["label"] == 0) & (df["prediction"] == 0)).sum()   # true negatives (Real -> correctly identified as real)
    fp = ((df["label"] == 0) & (df["prediction"] == 1)).sum()   # false positives (Real -> mistakenly flagged as AI)
    fn = ((df["label"] == 1) & (df["prediction"] == 0)).sum()   # false negatives (AI -> mistakenly passed as real)


    total_analyzed = len(df)
    if total_analyzed == 0:
        print(f"ERROR: No valid data to analyze.")
        return    

    """
    accuracy:
    Measures the overall correctness of the model.
    It represents the ratio of correctly predicted observations
    (both true positives and true negatives) to the total number of observations.
    It can be misleading if the dataset is imbalanced.
    """
    accuracy = (tp + tn) / total_analyzed

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
    Measures the model"s ability to discover all actual positive instances.
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


    print(f"Performance metrics computed successfully.")
    return {
        "total": total_analyzed, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,
        "fp_list": df[(df["label"] == 0) & (df["prediction"] == 1)][["filename", "score"]].values.tolist(),
        "fn_list": df[(df["label"] == 1) & (df["prediction"] == 0)][["filename", "score"]].values.tolist()
    }


def save_report(metrics: dict, npz_dir_path: Path, report_path: Path) -> None:
    """
    Generates a performance report and saves it to a text file.
    """

    print(f"Generating performance report...")

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if not metrics:
        print("ERROR: No metrics to report.")
        return
    

    # Report
    report = f"""TruFor performance report
Directory analyzed: '{npz_dir_path}/'
Total files analyzed: {metrics["total"]}
Threshold: {metrics["threshold"]:.2f}\n
Accuracy  :\t{metrics["accuracy"]:.2%}\t(percentage of correct predictions)
Precision :\t{metrics["precision"]:.2%}\t(reliability when it claims "It is AI")
Recall    :\t{metrics["recall"]:.2%}\t(sensitivity, ability to detect all AI images)
F1-score  :\t{metrics["f1_score"]:.4f}\t(final harmonic mean)\n
Confusion matrix:
True positives  :\t{metrics["tp"]}
True negatives  :\t{metrics["tn"]}
False negatives :\t{metrics["fn"]}\t(AI mistaken for real)
False positives :\t{metrics["fp"]}\t(real mistaken for AI)\n"""
    
    temp = []
    if metrics["fn_list"]:
        temp.append("\nList of false negatives:")
        for element, score in metrics["fn_list"]:
            temp.append(f"'{element}', score: {score:.4f}")

    if metrics["fp_list"]:
        temp.append("\nList of false positives:")
        for element, score in metrics["fp_list"]:
            temp.append(f"'{element}', score: {score:.4f}")

    report += "\n".join(temp) if temp else ''


    # Save
    print(report)
    # Parents = True creates any necessary parent directories
    # Exist_ok = True means that if the dir already exists, it won"t raise an error
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, mode = "w", encoding = "utf-8") as file:
        file.write(report)

    print(f"Performance report saved successfully to '{report_path}'.")


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Compute performance metrics of TruFor results.")

    # Flags
    TRUTH_FILE_PATH = TEMP_DIR_PATH / TRUTH_FILE_NAME
    parser.add_argument("--truth",      type = Path,     default = TRUTH_FILE_PATH,         help = f"Truth csv file path. Default: '{TRUTH_FILE_PATH}'")
    parser.add_argument("--npz",        type = Path,     default = NPZ_DIR_PATH,            help = f"Directory path containing .npz files. Default: '{NPZ_DIR_PATH}'")
    parser.add_argument("--threshold",  type = float,    default = PERFORMANCE_THRESHOLD,   help = f"Threshold for classifying an image as AI. Range: 0.0 to 1.0. Default: '{PERFORMANCE_THRESHOLD}'.")
    parser.add_argument("--output",     type = Path,     default = TEMP_DIR_PATH,           help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")

    # Read the command line arguments
    args = parser.parse_args()


    # Load truth csv
    truth = load_truth_csv(args.truth)

    # Extract scores from .npz files and create a DataFrame
    df = scores_to_dataframe(truth, args.npz)

    # Compute performance metrics
    metrics_results = compute_performance(df, args.threshold)

    # Save
    PERFORMANCE_FILE_PATH = args.output / PERFORMANCE_FILE_NAME
    save_report(metrics_results, args.npz, PERFORMANCE_FILE_PATH)