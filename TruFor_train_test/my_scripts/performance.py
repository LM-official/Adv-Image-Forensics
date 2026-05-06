import  argparse
from    config      import  TEMP_DIR_PATH, TRUTH_FILE_NAME, NPZ_DIR_PATH, PERFORMANCE_THRESHOLD, PERFORMANCE_FILE_NAME
import  csv
from    pathlib     import  Path
import  numpy       as      np


def load_truth_csv(truth_path: Path) -> dict[str, int] | None:
    '''
    Loads labels dictionary from the csv file.
    '''

    print(f'Loading "{truth_path}"...')

    truth = {}
    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    try:
        with open(truth_path, mode = 'r', encoding = 'utf-8') as file:
            reader = csv.reader(file)

            # skip header
            next(reader, None)

            for row in reader:
                # only process lines that have exactly 2 elements, separated by ","
                if not row or len(row) != 2:
                    print(f'WARNING: Invalid row "{row}". Skipping the entry.')
                    continue

                # EAFP = Easier to Ask for Forgiveness than Permission, so we try to cast the label to int and catch any exceptions
                try:
                    # int() converts and ignore whitespaces
                    label = int(row[1])
                    if label not in (0, 1):
                        raise ValueError("Label must be 0 or 1.")
                    truth[row[0].strip()] = label
                except ValueError:
                    print(f'WARNING: Invalid row "{row}". Skipping the entry.')
                    continue

    except Exception as e:
        print(f'ERROR: reading "{truth_path}": {e}')

    print(f'"{truth_path}" loaded successfully. Total entries: {len(truth)}')
    return truth


def compute_performance(truth: dict[str, int], npz_dir_path: Path, threshold: float, performance_file_path: Path) -> None:
    '''
    Computes performance metrics (accuracy, precision, recall, f1-score) of the TruFor results
    '''

    print(f'Computing performance...')

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if not truth or not npz_dir_path.is_dir() or not (0.0 <= threshold <= 1.0):
        print('ERROR: Invalid input parameters.')
        return
    

    # confusion matrix counters
    tp = 0  # true positives (AI -> correctly identified as AI)
    tn = 0  # true negatives (Real -> correctly identified as real)
    fp = 0  # false positives (Real -> mistakenly flagged as AI)
    fn = 0  # false negatives (AI -> mistakenly passed as real)
    fp_list, fn_list = [], []
    ignored_files = 0

    # analyze each .npz file in the output dir
    for file in npz_dir_path.glob('*.npz'):
        # .stem.stem removes the 2 extensions e.g. ".jpg.npz"
        filename, filePath = Path(file.stem).stem, file.resolve()

        # only process files in truth dictionary
        if filename not in truth:
            print(f'WARNING: "{filename}" not found in truth csv. Skipping the entry.')
            ignored_files += 1
            continue

        # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
        try:
            # use context manager to ensure the file is properly closed after reading
            with np.load(filePath) as data:
                score = float(data['score'])
        except Exception as e:
            print(f'ERROR: reading "{filePath}": {e}')
            ignored_files += 1
            continue
            

        # classification (0 = real, 1 = AI)
        prediction = 1 if score > threshold else 0
        
        # update the confusion matrix
        match (truth[filename], prediction):
            case (1, 1):
                tp += 1
            case (0, 0):
                tn += 1
            case (0, 1):
                fp += 1
                fp_list.append((filename, score))
            case (1, 0):
                fn += 1
                fn_list.append((filename, score))


    # compute metrics
    total_analyzed = tp + tn + fp + fn
    if total_analyzed == 0:
        print(f'ERROR: No valid .npz files found in "{npz_dir_path}".')
        return

    '''
    accuracy:
    Measures the overall correctness of the model.
    It represents the ratio of correctly predicted observations
    (both true positives and true negatives) to the total number of observations.
    It can be misleading if the dataset is imbalanced.
    '''
    accuracy = (tp + tn) / total_analyzed

    '''
    precision:
    Measures the reliability of positive predictions.
    "Out of all the instances the model flagged as positive, how many were actually positive?"
    A high precision indicates a low rate of false positives (false alarms).
    '''
    # tp + fp > 0 check to avoid division by zero when no positive predictions were made
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    '''
    recall (sensitivity):
    Measures the model's ability to discover all actual positive instances.
    "Out of all the true positive cases in the dataset, how many did the model successfully find?"
    A high recall indicates a low rate of false negatives (missed detections).
    '''
    # tp + fn > 0 check to avoid division by zero when no actual positives are present
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    '''
    f1-score:
    The harmonic mean of precision and recall.
    Since precision and recall often have an inverse relationship
    (improving one can worsen the other), the f1-score provides a single, balanced metric.
    It is widely considered the most robust and reliable indicator of a model's performance,
    especially when dealing with imbalanced data.
    '''
    # precision + recall > 0 check to avoid division by zero when both precision and recall are zero
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0


    # report
    report = f'''TruFor performance report
Directory analyzed: "{npz_dir_path}/"
Total files analyzed: {total_analyzed}
Ignored files: {ignored_files}
Threshold: {threshold:.2f}\n
Accuracy  :\t{accuracy:.2%}\t(percentage of correct predictions)
Precision :\t{precision:.2%}\t(reliability when it claims 'It is AI')
Recall    :\t{recall:.2%}\t(sensitivity, ability to detect all AI images)
F1-score  :\t{f1_score:.4f}\t(final harmonic mean)\n
Confusion matrix:
True positives  :\t{tp}
True negatives  :\t{tn}
False negatives :\t{fn}\t(AI mistaken for real)
False positives :\t{fp}\t(real mistaken for AI)\n'''
    
    temp = []
    if fn_list:
        temp.append('\nList of false negatives:')
        for (element, score) in fn_list:
            temp.append(f'"{element}", score: {score:.4f}')

    if fp_list:
        temp.append('\nList of false positives:')
        for (element, score) in fp_list:
            temp.append(f'"{element}", score: {score:.4f}')

    if temp:
        report += '\n'.join(temp)

    # save
    print(report)

    # create the output dir if it doesn't exist
    # parents = True creates any necessary parent directories
    # exist_ok = True means that if the dir already exists, it won't raise an error
    performance_file_path.parent.mkdir(parents = True, exist_ok = True)

    # write report to file
    with open(performance_file_path, mode = 'w', encoding = 'utf-8') as file:
        file.write(report)

    print(f'Performance computed successfully.')


if __name__ == '__main__':
    # initialize the argument parser
    parser = argparse.ArgumentParser(description = 'Compute performance metrics of TruFor results.')

    # flags
    TRUTH_FILE_PATH = TEMP_DIR_PATH / TRUTH_FILE_NAME
    parser.add_argument('--truth',      type = Path,     default = TRUTH_FILE_PATH,         help = f'Truth csv file path. Default: "{TRUTH_FILE_PATH}"')
    parser.add_argument('--npz',        type = Path,     default = NPZ_DIR_PATH,            help = f'Directory path containing .npz files. Default: "{NPZ_DIR_PATH}"')
    parser.add_argument('--threshold',  type = float,    default = PERFORMANCE_THRESHOLD,   help = f'Threshold for classifying an image as AI. Range: 0.0 to 1.0. Default: "{PERFORMANCE_THRESHOLD}".')
    parser.add_argument('--output',     type = Path,     default = TEMP_DIR_PATH,           help = f'Output directory path. Default: "{TEMP_DIR_PATH}"')

    # read the command line arguments
    args = parser.parse_args()


    truth = load_truth_csv(args.truth)


    PERFORMANCE_FILE_PATH = args.output / PERFORMANCE_FILE_NAME
    compute_performance(truth, args.npz, args.threshold, PERFORMANCE_FILE_PATH)
    # visualize a single image
    # cd /Users/lorenzo/Tirocinio/TruFor/TruFor_train_test/
    # source /Users/lorenzo/trufor_py_env/bin/activate
    # python visualize.py --image images/image.png --output output/image.png.npz