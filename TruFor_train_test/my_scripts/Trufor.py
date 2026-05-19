import argparse
from   config     import IMG_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE
import numpy      as     np
import pandas     as     pd
from   pathlib    import Path
import subprocess
import sys
from   utils      import load_csv


def run_trufor(images_dir: Path, npz_dir: Path, trufor_run: Path) -> bool:
    """
    Runs TruFor on the specified input and output directories.
    """

    if not images_dir.is_dir() or not npz_dir.is_dir() or not trufor_run.is_file():
        print(f"ERROR: Invalid images directory, npz directory, or TruFor run file.")
        return False


    # Initialize the command
    command = [
        # Current Python interpreter (.venv)
        sys.executable,
        # File
        str(trufor_run),
        "--input",
        str(images_dir),
        "--output",
        str(npz_dir)
    ]

    try:
        # Run
        print("Running TruFor...")
        subprocess.run(
        command,
        check=True,
        cwd=trufor_run.parent
    )
        print("TruFor completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        return False


def load_scores(npz_dir: Path) -> pd.DataFrame | None:
    """
    Iterates through the .npz files in the specified directory, extracts the TruFor anomaly score for each image, and returns a DataFrame.
    """

    if not npz_dir.is_dir():
        print(f"ERROR: '{npz_dir}' directory not found.")
        return None


    print(f"Extracting TruFor scores from .npz files in '{npz_dir}'...")
    trufor_data = []
    # Iterate through .npz files in the directory
    for npz in npz_dir.glob("*.npz"):
        try:
            # Use context manager to ensure the file is properly closed after reading
            with np.load(npz) as data:
                # Load only "score"
                # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
                trufor_data.append({
                    "filename":     Path(npz.stem).stem,
                    "trufor_score": float(data["score"]),
                })
        except Exception as e:
            print(f"ERROR: reading '{npz}': {e}")
            continue

    print(f"TruFor scores extracted successfully.")
    return pd.DataFrame(trufor_data)


def update_dataset(dataset: Path, npz_dir: Path) -> None:
    """
    Updates the dataset with the TruFor scores from the .npz files in the specified directory.
    """

    if not dataset.is_file() or not npz_dir.is_dir():
        print(f"ERROR: Invalid dataset file or npz directory.")
        return None


    print("Updating dataset with TruFor scores...")
    # Load the dataset
    df = load_csv(dataset)
    if df is None:
        print("ERROR: Failed to load dataset.")
        return None


    # Load TruFor scores from all .npz files in npz_dir
    scores = load_scores(npz_dir)
    if scores is None or scores.empty:
        print("ERROR: No TruFor scores found.")
        return None


    # Merge the dataset with the scores on the filename column
    df = pd.merge(df, scores, on = "filename", how = "inner")


    # Save
    df.to_csv(dataset, index=False, encoding="utf-8")
    print(f"Dataset updated successfully.")


def main() -> None:
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor and update dataset with scores")

    # Flags
    parser.add_argument("--images_dir", type=Path, default=IMG_DIR,               help=f"Directory path where the images are stored. Default: '{IMG_DIR}'")
    parser.add_argument("--npz_dir",    type=Path, default=NPZ_DIR,               help=f"Directory path where the npz files will be saved. Default: '{NPZ_DIR}'")
    parser.add_argument("--trufor_run", type=Path, default=TRUFOR_RUN,            help=f"Path to the TruFor run.py file. Default: '{TRUFOR_RUN}'")
    parser.add_argument("--dataset",    type=Path, default=FILTERED_DATASET_FILE, help=f"Path to the dataset file. Default: '{FILTERED_DATASET_FILE}'")
    args = parser.parse_args()

    # Run TruFor
    #if not run_trufor(args.images_dir, args.npz_dir, args.trufor_run):
    #    return None

    # Update dataset
    # Divide the update in batched is useless because the bottleneck is running TruFor
    # Run TruFor on a big dataset is very expensive, so the dataset should be small so divide in barches is not necessary 
    update_dataset(args.dataset, args.npz_dir)
    return None

if __name__ == "__main__":
    main()