import argparse
from   config     import TRUFOR_BATCH_SIZE, IMGS_DIR, NPZ_DIR, TRUFOR_RUN, FILTERED_DATASET_FILE, TRUFOR_WAIT, TRUFOR_MAX_PIXELS
from   pathlib    import Path
import subprocess
import sys
import tempfile
import time
from   utils      import load_npzs, update_dataset


def _run_trufor_batch(images: Path, npz: Path, trufor_run: Path):
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
        str(npz),
        "--max-pixels",
        str(TRUFOR_MAX_PIXELS)
    ]

    try:
        # Run
        subprocess.run(
            command,
            check=True,
            cwd=trufor_run.parent
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")


def run_trufor(images: Path = IMGS_DIR, npz: Path = NPZ_DIR, trufor_run: Path = TRUFOR_RUN, batch_size: int = TRUFOR_BATCH_SIZE, wait: int = TRUFOR_WAIT):
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

    if not images.is_dir() or not trufor_run.is_file():
        print("ERROR: Invalid images directory or TruFor run file.")
        return

    # Create the output directory if it doesn't exist
    npz.mkdir(parents=True, exist_ok=True)


    # No batch
    if batch_size <= 0:
        _run_trufor_batch(images, npz, trufor_run)
        return
    if wait < 0:
        print("ERROR: Wait time cannot be negative.")
        return


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
            _run_trufor_batch(staging, npz, trufor_run)

        # Wait between batches (not after the last one)
        if i < batches and wait > 0:
            time.sleep(wait)

    print("Batches completed.")


def main():
    """
    Runs TruFor on the input images and updates the dataset CSV with the resulting scores.

    Args:
        None (paths come from config.py; batch size and wait are per-run CLI flags).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run TruFor and update dataset with scores")
    # Flags
    # Paths come from config.py (the project backbone)
    parser.add_argument("--batch-size", type=int, default=TRUFOR_BATCH_SIZE, help=f"Images per TruFor invocation. Default: {TRUFOR_BATCH_SIZE}")
    parser.add_argument("--wait",       type=int, default=TRUFOR_WAIT,       help=f"Seconds to wait between batches. Default: {TRUFOR_WAIT}")
    args = parser.parse_args()

    # Run TruFor in batches (paths default from config; batch size and wait come from the flags)
    run_trufor(batch_size=args.batch_size, wait=args.wait)

    # Divide the update in batched is useless because the bottleneck is running TruFor
    # Run TruFor on a big dataset is very expensive, so the dataset should be small so divide in batches is not necessary

    # Load TruFor scores from all .npz files
    scores = load_npzs(NPZ_DIR, {"score"})
    if scores is None:
        return

    # Update dataset
    update_dataset(FILTERED_DATASET_FILE, scores, rename={"score": "trufor_score"})

if __name__ == "__main__":
    main()