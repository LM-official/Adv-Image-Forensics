from   config  import NPZ_COLS
import numpy   as     np
import pandas  as     pd
from   pathlib import Path
import re
import unicodedata


# =============================================================================
# TEXT HELPERS
# =============================================================================

def clean_text(text: str, dedup_words: bool) -> str:
    """
    Forces model-generated text (BLIP caption or OCR words) to letters only and (optionally) drops repeated words.

    Guarantees, regardless of what the model emitted:
    - accented letters are folded to plain ascii (libertà -> liberta) instead of being truncated;
    - only the letters a-z survive;
    - when dedup_words is set, no word appears more than once.

    Args:
        text:        Raw text decoded from the model.
        dedup_words: If True, remove duplicate words (keep the first occurrence, preserve order).
        fallback:    Returned when cleaning leaves the text empty (it was all punctuation/digits).

    Returns:
        The cleaned text, or fallback if nothing survives.
    """

    # print("Cleaning text...")

    if not text:
        print("ERROR: no text to clean.")
        return ""

    # Fold accents to plain ascii so the a-z filter keeps the whole word
    text  = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Only a-z: lowercase + replace every non a-z characters with a single space
    text  = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    words = text.split()
    # No word written multiple times, preserve order
    if dedup_words:
        seen  = set()
        words = [word for word in words if not (word in seen or seen.add(word))]
    cleaned = " ".join(words)
    if not cleaned:
        # print("ERROR: nothing left after cleaning the text.")
        return ""

    # print("Text cleaned successfully.")
    return cleaned


def save_report(text: str, path: Path):
    """
    Saves the given text to a file at the specified path.

    Args:
        text: Content to write to the file. If empty, the function does nothing.
        path: Destination file path; parent directories are created automatically.

    Returns:
        None.
    """
    
    print("Saving report...")

    if not text:
        print("ERROR: No text to save in the report.")
        return

    # Save
    # Create the output directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as file:
        file.write(text)
    print("Report saved successfully.")


# =============================================================================
# CSV HELPERS
# =============================================================================

def load_csv(csv: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """
    Loads the target columns from a csv file. If columns is None, loads all columns.

    Args:
        csv:     Path to the CSV file to load.
        columns: List of column names to load, or None to load every column.

    Returns:
        A DataFrame with the requested columns and no missing values, or None on failure / empty result.
    """

    print("Loading CSV...")
    try:
        # Keep only requested columns
        df = pd.read_csv(csv, sep=",", encoding="utf-8", usecols=columns)
        if df is None or df.empty:
            print("ERROR: No valid data found in CSV.")
            return None

        print("CSV loaded successfully.")
        return df
    except Exception as e:
        print(f"ERROR: loading CSV: {e}")
        return None


def update_csv(csv: Path, data: pd.DataFrame, rename: dict[str, str] | None = None):
    """
    Merges a DataFrame into the CSV and saves it back to the same file.

    Merges on the 'filename' column.

    Args:
        csv:     Path to the CSV file.
        data:    DataFrame to merge into the dataset.
        rename:  Optional mapping to rename columns before merging.
    """

    print("Updating CSV...")

    df = load_csv(csv)
    if df is None:
        return

    # Update
    if rename:
        data = data.rename(columns=rename)
    # Drop columns already present in the dataset (except the merge key) so the incoming values replace the old ones instead of producing _x/_y duplicates
    overlap = [col for col in data.columns if col != "filename" and col in df.columns]
    df = df.drop(columns=overlap)
    df = pd.merge(df, data, on="filename", how="inner")
    # Keep the dataset dense: drop any row that has an empty/null value
    df = df.dropna()
    df.to_csv(csv, index=False, encoding="utf-8")
    print("CSV updated successfully.")


# =============================================================================
# STATISTIC HELPERS
# =============================================================================

def binarize_engagement(df: pd.DataFrame, dead_zone: float) -> tuple[pd.DataFrame, pd.Series] | None:
    """
    Turns the continuous crowdtangle_score into the binary engagement target shared by every model.

    The score distribution is bimodal:
    posts whose |crowdtangle_score| <= dead_zone fall in the low-density valley between the under-over-performing modes
    and are dropped first, so the 0 cut-off separates two clearly-distinct clusters rather than slicing through ambiguous middle posts.

    Args:
        df:        DataFrame containing the crowdtangle_score column.
        dead_zone: Drop posts with |crowdtangle_score| <= this (the distribution's valley).
                   0 disables the cut and keeps every post; negatives are clamped to 0.

    Returns:
        A (filtered df, binary y) tuple where y is 1 = over-performed (score > 0) and 0 = under-performed,
        or None if no usable posts remain.
    """

    print("Binarising the engagement target...")

    # Negatives are meaningless: clamp them to 0. 0 disables the cut and keeps every post
    if dead_zone < 0:
        dead_zone = 0

    if dead_zone > 0:
        keep = df["crowdtangle_score"].abs() > dead_zone
        df   = df[keep].reset_index(drop=True)
        if df.empty or df["crowdtangle_score"].nunique() <= 1:
            print(f"ERROR: no usable posts left after dropping the |score| <= {dead_zone:g} dead zone.")
            return None

    # Binarise the target: 1 = over-performed (score > 0), 0 = under-performed (score <= 0)
    # With the dead zone removed every remaining post sits clearly on one side
    y = (df["crowdtangle_score"] > 0).astype(int)
    print("Engagement target binarised successfully.")
    return df, y


# =============================================================================
# NPZ HELPERS
# =============================================================================

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
    
    #print("Loading .npz file...")

    if not columns or any(col not in NPZ_COLS for col in columns):
        #print(f"ERROR: Invalid .npz file or columns. Expected a non-empty subset of: {NPZ_COLS}")
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
        #print(".npz file loaded successfully.")
        return pd.DataFrame([result])
    except Exception:
        #print(f"ERROR: reading npz: {e}")
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

    print("Loading .npz files...")

    if not npzs.is_dir():
        print("ERROR: Invalid directory.")
        return None

    # Extracting values
    result = []
    for item in npzs.glob("*.npz"):
        df = load_npz(item, columns)
        if df is not None:
            # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
            row = {"filename": Path(item.stem).stem, **df.iloc[0].to_dict()}
            result.append(row)

    if not result:
        print("ERROR: No valid .npz files found.")
        return None

    print(".npz files loaded successfully.")
    return pd.DataFrame(result)