from   config  import NPZ_COLS
import numpy   as     np
import pandas  as     pd
from   pathlib import Path


def update_dataset(dataset: Path, data: pd.DataFrame, rename: dict[str, str] | None = None) -> None:
    """
    Merges a DataFrame into the dataset CSV and saves it back to the same file.

    Merges on the 'filename' column.

    Args:
        dataset: Path to the dataset CSV file.
        data:    DataFrame to merge into the dataset.
        rename:  Optional mapping to rename columns before merging.
    """

    df = load_csv(dataset)
    if df is None:
        return None
    

    # Update
    print(f"Updating dataset '{dataset}'...")
    if rename:
        data = data.rename(columns=rename)
    # Drop columns already present in the dataset (except the merge key) so the incoming values replace the old ones instead of producing _x/_y duplicates
    overlap = [col for col in data.columns if col != "filename" and col in df.columns]
    df = df.drop(columns=overlap)
    df = pd.merge(df, data, on="filename", how="inner")
    # Keep the dataset dense: drop any row that has an empty/null value
    df = df.dropna()
    df.to_csv(dataset, index=False, encoding="utf-8")
    print("Dataset updated successfully.")
    return None


def load_csv(csv: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """
    Loads the target columns from a csv file. If columns is None, loads all columns.

    Args:
        csv:     Path to the CSV file to load.
        columns: List of column names to load, or None to load every column.

    Returns:
        A DataFrame with the requested columns and no missing values, or None on failure / empty result.
    """

    print(f"Loading '{csv}'...")
    try:
        # Load only the necessary columns
        df = pd.read_csv(csv, sep=",", encoding="utf-8", usecols=columns)
        if df is None or df.empty:
            print(f"ERROR: No valid data found in '{csv}'.")
            return None

        print(f"'{csv}' loaded successfully.")
        return df
    except Exception as e:
        print(f"ERROR: loading '{csv}': {e}")
        return None
    

def load_npz(npz: Path, columns: set[str]) -> pd.DataFrame | None:
    """
    Loads the target columns from a .npz file.

    Columns can be "map", "imgsize", "score", "conf".

    Args:
        npz:     Path to the .npz file.
        columns: Set of column names to extract.

    Returns:
        A single-row DataFrame with the requested columns, or None on failure.
    """
    
    if not npz.is_file() or not columns or any(col not in NPZ_COLS for col in columns):
        print(f"ERROR: Invalid .npz file or columns {columns}. Expected a non-empty subset of: {NPZ_COLS}")
        return None


    columns = sorted(columns)
    # Extract values
    try:
        # Use context manager to ensure the file is properly closed after reading
        with np.load(npz) as data:
            result = {}
            for col in columns:
                value = data[col]
                result[col] = float(value) if value.ndim == 0 else value.tolist()
        return pd.DataFrame([result])
    except Exception as e:
        print(f"ERROR: reading '{npz}': {e}")
        return None


def load_npzs(npzs: Path, columns: set[str]) -> pd.DataFrame | None:
    """
    Loads the target columns from all .npz files in a directory.
    Each row corresponds to one .npz file, with the filename (without extension) as an additional column.

    Columns can be "map", "imgsize", "score", "conf".

    Args:
        npzs:    Path to the directory containing .npz files.
        columns: Set of column names to extract.
    Returns:
        A DataFrame where each row is one .npz file, or None on failure.
    """

    if not npzs.is_dir():
        print(f"ERROR: Invalid directory '{npzs}'.")
        return None

    # Extracting values
    print(f"Extracting values from .npz files in '{npzs}'...")
    result = []
    for item in npzs.glob("*.npz"):
        df = load_npz(item, columns)
        if df is not None:
            # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
            row = {"filename": Path(item.stem).stem, **df.iloc[0].to_dict()}
            result.append(row)

    if not result:
        print(f"ERROR: No valid .npz files found in '{npzs}'.")
        return None

    print(f"Values extracted successfully from '{npzs}'.")
    return pd.DataFrame(result)


def save_report(text: str, path: Path) -> None:
    """
    Saves the given text to a file at the specified path.

    Args:
        text: Content to write to the file. If empty, the function does nothing.
        path: Destination file path; parent directories are created automatically.

    Returns:
        None.
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