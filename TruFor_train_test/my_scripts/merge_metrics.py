import  argparse
from    config      import  NPZ_DIR_PATH, TEMP_DIR_PATH, MERGED_METRICS_FILE_NAME
import  numpy       as      np
import  pandas      as      pd
from    pathlib     import  Path
import  sys


def load_dataset(dataset: list[Path]) -> pd.DataFrame | None:
    """
    Loads and preprocesses the dataset from the provided files.
    The function is tailored to a specific dataset structure.
    """
    
    print("""Loading and preprocessing dataset...
WARNING: 'load_dataset' function is tailored to a specific structure. If you use a custom dataset, you may need to adjust the parsing logic inside the function.""")
    

    FILE_NUMBER = 2 # Expected number of files in the dataset list
    # dataset[0] = ASSIGNMENT_PATH             = "/Users/lorenzo/Tirocinio/GenImageNet/data/ratings/GenImageNet_assignmentLevel.csv"
    # dataset[1] = IMAGE_PATH                  = "/Users/lorenzo/Tirocinio/GenImageNet/data/ratings/GenImageNet_imageLevel.csv"
    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if len(dataset) != FILE_NUMBER:
        print(f"ERROR: Expected exactly {FILE_NUMBER} dataset files, but got {len(dataset)}.")
        return

    """
    in a DataFrame each column can store a different datatype.
    indexes: by default rows have an integer index starting at 0,
    columns have string names that can be used to access the data.
    """
    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    try:
        # Load human evaluation scores
        df_assign = pd.read_csv(dataset[0], sep = ";")

        # There are 10 votes for each image, so we need to calculate the mean score
        # Group by filename to calculate the mean score for each image
        df_mean_scores = df_assign.groupby("filename")[["qualityScore", "realismScore"]].mean().reset_index()
        

        # Load the image metadata
        df_image = pd.read_csv(dataset[1], sep = ";")

        # Replace commas with dots for decimal values and cast to float
        df_image["aestheticsScore"]     = df_image["aestheticsScore"].astype(str).str.replace(",", ".").astype(float)
        df_image["colorSaturation"]     = df_image["colorSaturation"].astype(str).str.replace(",", ".").astype(float)
        df_image["visualComplexity"]    = df_image["visualComplexity"].astype(str).str.replace(",", ".").astype(float)
        

        # How = "inner" ensures we only keep rows with matching filenames in both dataframes
        df_metrics = pd.merge(df_mean_scores, df_image, on = "filename", how = "inner")

        # Remove extension from files
        df_metrics["filename"] = df_metrics["filename"].apply(lambda x: Path(x).stem)

        print(f"Dataset loaded successfully.")
        return df_metrics
    
    except Exception as e:
        print(f"ERROR: loading dataset files: {e}")
        return
    


def load_scores(npz_dir_path: Path) -> pd.DataFrame | None:
    """
    Iterates through the .npz files in the specified directory,
    extracts the TruFor anomaly score for each image, and returns a dataframe.
    """     

    print(f"Extracting TruFor scores from .npz files in '{npz_dir_path}'...")

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if not npz_dir_path.is_dir():
        print(f"ERROR: '{npz_dir_path}' directory not found.")
        return

    trufor_data = []
    # Glob search only for .npz files
    for npz in npz_dir_path.glob("*.npz"):
        # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
        filename, filePath = Path(npz.stem).stem, npz.resolve()

        # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
        try:
            # Use context manager to ensure the file is properly closed after reading
            with np.load(filePath) as data:
                # Load only "score"
                trufor_data.append({"filename": filename, "trufor_score": float(data["score"])})
        except Exception as e:
            print(f"ERROR: reading '{filePath}': {e}")
            continue
            
    print(f"TruFor scores extracted successfully.")
    return pd.DataFrame(trufor_data)


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Compute performance metrics of TruFor results.")

    # Flags
    parser.add_argument("--dataset",    type = Path,     required = True,   nargs = "+",       help = "Dataset files. Required.")
    parser.add_argument("--npz",        type = Path,     default = NPZ_DIR_PATH,               help = f"Input directory path containing .npz files. Default: '{NPZ_DIR_PATH}'")
    parser.add_argument("--output",     type = Path,     default = TEMP_DIR_PATH,              help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")
    
    # Read the command line arguments
    args = parser.parse_args()


    # Load dataset
    metrics = load_dataset(args.dataset)
    if metrics is None or metrics.empty:
        sys.exit(1)
        

    # Load npz
    trufor = load_scores(args.npz)
    if trufor is None or trufor.empty:
        sys.exit(1)


    # How = "inner" ensures we only keep rows with matching filenames in both dataframes
    metrics = pd.merge(metrics, trufor, on = "filename", how = "inner")
    # Remove rows with missings
    primary_cols = ["filename", "trufor_score"]
    metrics = metrics.dropna(subset = primary_cols)


    # Reorder: custom order + all remaining columns
    metrics = metrics[primary_cols + [col for col in metrics.columns.to_list() if col not in primary_cols]]

    # Save
    # Create output directory if it doesn"t exist
    MERGED_METRICS_FILE_PATH = args.output / MERGED_METRICS_FILE_NAME
    MERGED_METRICS_FILE_PATH.parent.mkdir(parents = True, exist_ok = True)
    # Index = False to avoid writing row numbers
    metrics.to_csv(MERGED_METRICS_FILE_PATH, sep = ",", index = False, encoding = "utf-8")