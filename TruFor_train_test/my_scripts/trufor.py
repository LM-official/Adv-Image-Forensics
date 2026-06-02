import argparse
from   config     import TRUFOR_BATCH_SIZE, IMGS_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE, WAIT
from   pathlib    import Path
import subprocess
import sys
import tempfile
import time
from   utils      import load_npzs, update_dataset


def _run_trufor_batch(images: Path, npz: Path, trufor_run: Path) -> bool:
    """
    Runs TruFor on the specified input and output directories.

    Args:
        images:     Directory containing the input images to analyze.
        npz:        Directory where the resulting .npz files will be saved.
        trufor_run: Path to the TruFor entry-point script (test.py).

    Returns:
        True if TruFor completed successfully, False otherwise (invalid paths or subprocess error).
    """

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
        subprocess.run(
            command,
            check=True,
            cwd=trufor_run.parent
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        return False


def run_trufor(images: Path, npz: Path, trufor_run: Path, batch_size: int, wait: int = WAIT) -> bool:
    """
    Runs TruFor in chunks of `batch_size` images, writing every output straight into `npz`,
    and waits `wait` seconds between batches to let the machine cool down.
    
    saves each .npz as it finishes, so an interrupted run leaves completed images on disk;
    re-running skips every image whose .npz already exists and resumes from there.

    Each batch is a separate TruFor process, so the model reloads once per batch.

    Args:
        images:     Directory containing the input images to analyze.
        npz:        Directory where the resulting .npz files will be saved.
        trufor_run: Path to the TruFor entry-point script (test.py).
        batch_size: Number of images to send per TruFor invocation.
        wait:       Seconds to wait between batches.

    Returns:
        True if every batch completed successfully, False otherwise.
    """

    if not images.is_dir() or not trufor_run.is_file() or batch_size <= 0 or wait < 0:
        print("ERROR: Invalid images directory, TruFor run file, batch size, or wait time.")
        return False

    # Create the output directory if it doesn't exist
    npz.mkdir(parents=True, exist_ok=True)

    # TruFor skips already analyzed images
    image_files   = [p for p in images.iterdir() if p.is_file()]
    len_images    = len(image_files)
    batches = (len_images + batch_size - 1) // batch_size
    print(f"{len_images} images -> {batches} batches of up to {batch_size}.")
    print(f"Waiting {wait} seconds before each next batch")
    for i, start in enumerate(range(0, len_images, batch_size), start=1):
        batch = image_files[start:start + batch_size]
        print(f"Processing batch {i}/{batches}...")

        # TruFor --input takes a directory, not a file list, so stage the batch as symlinks in a temp dir;
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for image in batch:
                (staging / image.name).symlink_to(image.absolute())
            if not _run_trufor_batch(staging, npz, trufor_run):
                print(f"ERROR: batch {i} failed, stopping.")
                return False

        # Wait between batches (not after the last one)
        if i < batches and wait > 0:
            time.sleep(wait)

    print("All batches completed successfully.")
    return True


def main() -> None:
    """
    Runs TruFor on the input images and updates the dataset CSV with the resulting scores.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor and update dataset with scores")

    # Flags
    parser.add_argument("--images",     type=Path, default=IMGS_DIR,              help=f"Directory path where the images are stored. Default: '{IMGS_DIR}'")
    parser.add_argument("--npz",        type=Path, default=NPZ_DIR,               help=f"Directory path where the npz files will be saved. Default: '{NPZ_DIR}'")
    parser.add_argument("--trufor-run", type=Path, default=TRUFOR_RUN,            help=f"Path to the TruFor run file. Default: '{TRUFOR_RUN}'")
    parser.add_argument("--dataset",    type=Path, default=FILTERED_DATASET_FILE, help=f"Path to the dataset file. Default: '{FILTERED_DATASET_FILE}'")
    parser.add_argument("--batch-size", type=int,  default=TRUFOR_BATCH_SIZE,     help=f"Images per TruFor invocation. Default: {TRUFOR_BATCH_SIZE}")
    parser.add_argument("--wait",       type=int,  default=WAIT,                  help=f"Seconds to wait between batches. Default: {WAIT}")
    args = parser.parse_args()

    # Run TruFor in batches
    if not run_trufor(args.images, args.npz, args.trufor_run, args.batch_size, args.wait):
        return None
    # Fast for testing
    # _run_trufor_batch(args.images, args.npz, args.trufor_run):
    
    # Divide the update in batched is useless because the bottleneck is running TruFor
    # Run TruFor on a big dataset is very expensive, so the dataset should be small so divide in batches is not necessary
    
    # Load TruFor scores from all .npz files
    scores = load_npzs(args.npz, {"score"})
    if scores is None:
        return None
    
    # Update dataset
    update_dataset(args.dataset, scores, rename={"score": "trufor_score"})
    return None

if __name__ == "__main__":
    main()