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
        print(f"'{csv}' loaded successfully.")
        return metrics
    except Exception as e:
        print(f"ERROR: loading '{csv}': {e}")
        return None