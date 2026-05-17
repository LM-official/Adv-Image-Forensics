import  pandas      as      pd
from    pathlib     import  Path


def load_target_metrics(metrics_file: Path, columns: list[str] | None) -> pd.DataFrame | None:
    """
    Loads the target metrics from a csv file. If columns is None, loads all columns.
    """

    print(f"Loading '{metrics_file}'...")
    

    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    try:
        # Load only the necessary columns
        metrics = pd.read_csv(metrics_file, sep = ",", encoding = "utf-8", usecols = columns)
        print(f"'{metrics_file}' loaded successfully.")
        return metrics
    
    except Exception as e:
        print(f"ERROR: loading '{metrics_file}': {e}")
        return