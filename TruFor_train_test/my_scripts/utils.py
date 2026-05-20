import numpy   as np
import pandas  as     pd
from   pathlib import Path


def load_csv(csv: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """
    Loads the target columns from a csv file. If columns is None, loads all columns.ù
    
    By default loads all columns.
    """

    print(f"Loading '{csv}'...")
    try:
        # Load only the necessary columns
        metrics = pd.read_csv(csv, sep=",", encoding="utf-8", usecols=columns)
        if metrics is None or metrics.empty:
            print(f"ERROR: No valid data found in '{csv}'.")
            return None
        metrics.dropna(inplace=True)
        if metrics is None or metrics.empty:
            print(f"ERROR: No valid data found in '{csv}' after dropping NaN values.")
            return None

        print(f"'{csv}' loaded successfully.")
        return metrics
    except Exception as e:
        print(f"ERROR: loading '{csv}': {e}")
        return None
    

def load_npz(npz: Path, columns: set[str]) -> pd.DataFrame | None:
    """
    Loads the target columns from all .npz files in a directory.
    Each row corresponds to one .npz file, with the filename (without extensions) as an additional column.
    Columns can be "map", "imgsize", "score", "conf".
    """

    VALID_COLS = frozenset(("map", "imgsize", "score", "conf"))
    if not npz.is_dir() or not columns or any(col not in VALID_COLS for col in columns):
        print(f"ERROR: '{npz}' is not a valid directory or columns are invalid. Expected columns: {VALID_COLS}")
        return None


    columns = sorted(columns)
    # Extract values from .npz files
    print(f"Extracting values from .npz files in '{npz}'...")
    result = []
    for item in npz.glob("*.npz"):
        try:
            # Use context manager to ensure the file is properly closed after reading
            with np.load(item) as data:
                # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
                row = {"filename": Path(item.stem).stem}

                # Load only the necessary columns
                for col in columns:
                    value = data[col]
                    # Scalar values (score) -> float, arrays (map, conf, imgsize) -> list
                    row[col] = float(value) if value.ndim == 0 else value.tolist()
                result.append(row)
        except Exception as e:
            print(f"ERROR: reading '{item}': {e}")
            continue

    if not result:
        print(f"ERROR: No valid .npz files found in '{npz}'.")
        return None
    print(f"Values extracted successfully.")
    return pd.DataFrame(result)


def save_report(text: str, path: Path) -> None:
    """
    Saves the given text to a file at the specified path.
    """
    
    if not text:
        print("ERROR: No text to save in the report.")
        return None
    

    # Save
    # Create the output directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as file:
        file.write(text)
    print(f"Report saved to '{path}'.")
    return None