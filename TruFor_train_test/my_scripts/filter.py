from   config   import CHAR_LIMIT, DATASET_FILE, DATASET_IMGS_DIR, FILTERED_DATASET_FILE, IMGS_DIR
import csv
import hashlib
import json
from   pathlib  import Path


def _url_to_stem(url: str, char_limit: int = CHAR_LIMIT) -> str:
    """
    Reproduces the filename stem (no extension) that CrowdTangle/Meta's assigns to a media URL

    Args:
        url:        Media URL to hash.
        char_limit: Number of leading hex characters to keep (see config.CHAR_LIMIT).

    Returns:
        The truncated sha256 hex digest used as the on-disk filename stem.
    """


    return hashlib.sha256(url.encode()).hexdigest()[:char_limit]


def _is_valid_post(data: dict, hash_map: dict[str, list[Path]]) -> tuple[list[Path]] | None:
    """
    Checks if a post is valid based on the specified filters and returns the list of image paths if valid.

    A post is valid if:
    - It has a score
    - All its media items are photos and have a matching file in the hash map

    Args:
        data:     Parsed JSON object representing a single post.
        hash_map: Mapping from URL sha256 hash (filename stem) to the list of matching image paths.

    Returns:
        A list of image paths associated with the post, or None if the post does not pass the filters.
    """

    if not isinstance(data, dict) or len(data) == 0:
        print(f"ERROR: invalid post data.")
        return None


    # Must have a score
    if data.get("score") is None:
        print(f"DEBUG: post '{data.get('id')}' rejected for missing score.")
        return None

    # Must have at least one media item
    media = data.get("media")
    if not media or not isinstance(media, list) or len(media) == 0:
        print(f"DEBUG: post '{data.get('id')}' rejected for missing media.")
        return None

    images = []
    for item in media:
        # Reject the entire post if ANY media item is not a photo
        if item.get("type") != "photo":
            print(f"DEBUG: post '{data.get('id')}' rejected for non-photo media item.")
            return None
        image_url = item.get("url") or item.get("full")
        if not image_url:
            print(f"DEBUG: post '{data.get('id')}' rejected for missing image URL.")
            return None

        # Hash the URL the same way the file was named on disk
        url_hash = _url_to_stem(image_url)

        # Dictionary lookup
        founds = hash_map.get(url_hash, [])
        if not founds:
            print(f"DEBUG: post '{data.get('id')}' rejected for missing image file for URL '{image_url}' (hash '{url_hash}').")
            return None
        # With a full-length hash this is always a single file; more than one means a truncated-hash collision (different URLs sharing a stem)
        if len(founds) > 1:
            print(f"WARNING: hash collision for stem '{url_hash}': {founds}")
            return None

        # Append all found images
        images.extend(founds)

    return images


def _build_hash_map(images: Path) -> dict[str, list[Path]] | None:
    """
    Scans the images directory and returns a dictionary mapping each filename stem (hash) to the list of matching file paths.

    Args:
        images: Directory containing the dataset images named after their URL hash.

    Returns:
        A dict mapping filename stem to the list of file paths, or None if the directory is invalid.
    """


    hash_map = {}
    for image in images.iterdir():
        if image.is_file():
            hash_map.setdefault(image.stem, []).append(image)
    return hash_map


def filter_jsonl(dataset: Path = DATASET_FILE, dataset_images: Path = DATASET_IMGS_DIR, output_csv: Path = FILTERED_DATASET_FILE, output_images: Path = IMGS_DIR) -> None:
    """
    Filters the JSONL dataset based on the specified criteria and saves the filtered records to a new CSV file.
    
    It also creates symlinks for the valid images in the output directory.

    Args:
        dataset:        Path to the input JSONL dataset file.
        dataset_images: Directory containing the original dataset images.
        output_csv:     Output CSV file path for the filtered records.
        output_images:  Output directory where symlinks to the kept images are created.

    Returns:
        None.
    """

    if not dataset.is_file() or not dataset_images.is_dir():
        print(f"ERROR: dataset '{dataset}' or images directory '{dataset_images}' not found.")
        return None


    # Build the hash map
    print("Indexing images directory...")
    hash_map = _build_hash_map(dataset_images)
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
        with open(dataset, mode="r", encoding="utf-8") as infile, \
             open(output_csv, mode="w", encoding="utf-8", newline="", buffering=1 << 16) as outfile:

            # Define the CSV columns
            cols = ["id", "crowdtangle_score", "filename"]
            # Initialize the CSV DictWriter
            writer = csv.DictWriter(outfile, fieldnames=cols)
            # Write the header
            writer.writeheader()


            for line in infile:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: skipping corrupted JSON line, {e}.")
                    continue

                # Filters
                images = _is_valid_post(data, hash_map)
                if not images:
                    continue

                for image in images:
                    # Save only interesting columns
                    writer.writerow({
                        "id":                 data.get("id"),
                        "crowdtangle_score":  data.get("score"),
                        "filename":           image.stem
                    })

                    # Create a symlink for the image
                    symlink = output_images / image.name
                    if not (symlink.exists() or symlink.is_symlink()):
                        symlink.symlink_to(resolved_images_dir / image.name)

        print(f"Dataset filtered successfully.\nOutput saved to: '{output_csv}'.")

    except Exception as e:
        print(f"ERROR: processing dataset: {e}")
        return None
    

def main() -> None:
    """
    Filters the dataset, producing the filtered CSV and the symlinked images directory.

    Args:
        None (paths come from config.py, the project backbone).

    Returns:
        None.
    """

    # Process dataset (every path defaults from config.py)
    filter_jsonl()
    return None

if __name__ == "__main__":
    main()