import argparse
from   config     import IMG_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE
from   pathlib    import Path
import subprocess
import sys
from   utils      import load_npzs, update_dataset


def run_trufor(images: Path, npz: Path, trufor_run: Path) -> bool:
    """
    Runs TruFor on the specified input and output directories.
    """

    if not images.is_dir() or not npz.is_dir() or not trufor_run.is_file():
        print("ERROR: Invalid images directory, npz directory, or TruFor run file.")
        return False


    # Initialize the command
    command = [
        # Current Python interpreter (.venv)
        sys.executable,
        # File
        str(trufor_run),
        "--input",
        str(images),
        "--output",
        str(npz)
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


def main() -> None:
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor and update dataset with scores")

    # Flags
    parser.add_argument("--images",     type=Path, default=IMG_DIR,               help=f"Directory path where the images are stored. Default: '{IMG_DIR}'")
    parser.add_argument("--npz",        type=Path, default=NPZ_DIR,               help=f"Directory path where the npz files will be saved. Default: '{NPZ_DIR}'")
    parser.add_argument("--trufor-run", type=Path, default=TRUFOR_RUN,            help=f"Path to the TruFor run file. Default: '{TRUFOR_RUN}'")
    parser.add_argument("--dataset",    type=Path, default=FILTERED_DATASET_FILE, help=f"Path to the dataset file. Default: '{FILTERED_DATASET_FILE}'")
    args = parser.parse_args()

    # Run TruFor
    if not run_trufor(args.images, args.npz, args.trufor_run):
        return None

    # Load TruFor scores from all .npz files
    scores = load_npzs(args.npz, {"score"})
    if scores is None:
        return None

    
    # Update dataset
    # Divide the update in batched is useless because the bottleneck is running TruFor
    # Run TruFor on a big dataset is very expensive, so the dataset should be small so divide in batches is not necessary
    update_dataset(args.dataset, scores, rename={"score": "trufor_score"})
    return None

if __name__ == "__main__":
    main()