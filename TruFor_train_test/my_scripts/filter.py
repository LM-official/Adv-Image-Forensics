import argparse
from   config   import DATASET_FILE, DATASET_IMGS_DIR, FILTERED_DATASET_FILE, IMG_DIR
import csv
import hashlib
import json
from   pathlib  import Path


def is_valid_post(data: dict, hash_map: dict[str, list[Path]]) -> list[Path] | None:
    """
    Checks if a post is valid based on the specified filters and returns the list of image paths if valid.

    A post is valid if:
    - It has a score
    - All its media items are photos and have a matching file in the hash map
    """

    if not isinstance(data, dict) or len(data) == 0 or not hash_map:
        return None


    # Must have a score
    if data.get("score") is None:
        return

    # Must have at least one media item
    media = data.get("media")
    if not media or not isinstance(media, list) or len(media) == 0:
        return None

    images = []
    for item in media:
        # Reject the entire post if ANY media item is not a photo
        if item.get("type") != "photo":
            return None
        if not (image_url := item.get("full")):
            return None

        # Hash the URL to find the corresponding file
        url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()

        # Dictionary lookup
        founds = hash_map.get(url_hash, [])
        if not founds:
            return None

        # Append all found images
        images.extend(founds)

    return images


def build_hash_map(images_dir: Path) -> dict[str, list[Path]] | None:
    """
    Scans the images directory and returns a dictionary mapping each filename stem (hash) to the list of matching file paths.
    """

    if not images_dir.is_dir():
        print(f"ERROR: images directory '{images_dir}' not found.")
        return None


    hash_map = {}
    for image in images_dir.iterdir():
        if image.is_file():
            hash_map.setdefault(image.stem, []).append(image)
    return hash_map


def filter_jsonl(dataset_path: Path, dataset_images: Path, output_csv: Path, output_images: Path) -> None:
    """
    Filters the JSONL dataset based on the specified criteria and saves the filtered records to a new CSV file.
    It also creates symlinks for the valid images in the output directory.
    """

    if not dataset_path.is_file() or not dataset_images.is_dir():
        print(f"ERROR: dataset '{dataset_path}' or images directory '{dataset_images}' not found.")
        return None


    # Build the hash map
    print("Indexing images directory...")
    hash_map = build_hash_map(dataset_images)
    if hash_map is None:
        return None


    # Resolve the images dir once — avoids one .resolve() call per image in the loop
    resolved_images_dir = dataset_images.resolve()


    # Create the output directories if they don't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)


    # Filter
    print("Filtering dataset...")
    try:
        # buffering=1<<16 (64KB) reduces the number of disk writes significantly on large outputs compared to the default line-by-line flushing
        with open(dataset_path, mode="r", encoding="utf-8") as input, \
             open(output_csv, mode="w", encoding="utf-8", newline="", buffering=1 << 16) as output:

            # Define the CSV columns
            cols = ["id", "crowdtangle_score", "filename"]
            # Initialize the CSV DictWriter
            writer = csv.DictWriter(output, fieldnames=cols)
            # Write the header
            writer.writeheader()


            for line in input:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: skipping corrupted JSON line, {e}.")
                    continue

                # Filters
                images = is_valid_post(data, hash_map)
                if not images:
                    continue

                for image in images:
                    # Save only interesting columns
                    writer.writerow({
                        "id":                 data.get("id"),
                        "crowdtangle_score":  data.get("score"),
                        "filename":           image.stem
                    })

                    # Create symlinks for the images                    
                    symlink = output_images / image.name
                    if not symlink.exists():
                        symlink.symlink_to(resolved_images_dir / image.name)

        print(f"Dataset filtered successfully.\nOutput saved to: '{output_csv}'.")

    except Exception as e:
        print(f"ERROR: processing dataset: {e}")
        return None


def main() -> None:
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Filter dataset")

    # Flags
    parser.add_argument("--dataset",        type=Path, default=DATASET_FILE,          help=f"Input dataset file path. Default: '{DATASET_FILE}'")
    parser.add_argument("--dataset_images", type=Path, default=DATASET_IMGS_DIR,      help=f"Directory path where the dataset images are stored. Default: '{DATASET_IMGS_DIR}'")
    parser.add_argument("--output_csv",     type=Path, default=FILTERED_DATASET_FILE, help=f"Output file path for csv. Default: '{FILTERED_DATASET_FILE}'")
    parser.add_argument("--output_images",  type=Path, default=IMG_DIR,               help=f"Output directory path for the images. Default: '{IMG_DIR}'")
    args = parser.parse_args()

    # Process dataset
    filter_jsonl(args.dataset, args.dataset_images, args.output_csv, args.output_images)
    return None

if __name__ == "__main__":
    main()