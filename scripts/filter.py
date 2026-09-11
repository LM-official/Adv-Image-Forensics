from   config           import (CHAR_LIMIT, DATASET_FILE, DATASET_IMGS_DIR, FILTERED_DATASET_FILE, FILTERED_IMGS_DIR, DISTRIBUTION_FILE,
                                PLOT_MAIN_COLOR, PLOT_SECOND_COLOR, DISTRIBUTION_PLOT_FILE)
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg") # Non-interactive backend: render straight to file, no display needed
import matplotlib.pyplot as     plt
from   pathlib           import Path
from   utils             import load_csv, save_report


# =============================================================================
# (1) FILTER DATASET
# =============================================================================

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


def _is_valid_post(data: dict, hash_map: dict[str, list[Path]]) -> Path | None:
    """
    Checks if a post is valid based on the specified filters and returns the image representing it if valid.

    A post is valid if:
    - It has a score
    - Its first media item is a photo and has a matching file in the hash map

    Args:
        data:     Parsed JSON object representing a single post.
        hash_map: Mapping from URL sha256 hash (filename stem) to the list of matching image paths.

    Returns:
        The image path representing the post, or None if the post does not pass the filters.
    """

    if not isinstance(data, dict) or len(data) == 0:
        print("WARNING: invalid post data.")
        return None


    # Must have a score
    if data.get("score") is None:
        print(f"WARNING: post '{data.get('id')}' rejected for missing score.")
        return None

    # Must have at least one media item
    media = data.get("media")
    if not media or not isinstance(media, list) or len(media) == 0:
        print(f"WARNING: post '{data.get('id')}' rejected for missing media.")
        return None
    # Save only the first media (the one that represents more the post)
    image = media[0]

    # Reject the post if its first media item is not a photo
    if image.get("type") != "photo":
        print(f"WARNING: post '{data.get('id')}' rejected for non-photo media item.")
        return None
    image_url = image.get("url") or image.get("full")
    if not image_url:
        print(f"WARNING: post '{data.get('id')}' rejected for missing image URL.")
        return None

    # Hash the URL the same way the file was named on disk
    url_hash = _url_to_stem(image_url)

    # Dictionary lookup
    founds = hash_map.get(url_hash, [])
    if not founds:
        print(f"WARNING: post '{data.get('id')}' rejected for missing image file for URL.")
        return None
    # With a full-length hash this is always a single file; more than one means a truncated-hash collision (different URLs sharing a stem)
    if len(founds) > 1:
        print(f"WARNING: hash collision for stem '{url_hash}': {founds}")
        return None

    return founds[0]


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


def filter(dataset: Path = DATASET_FILE, dataset_images: Path = DATASET_IMGS_DIR,
           output_csv: Path = FILTERED_DATASET_FILE, output_images: Path = FILTERED_IMGS_DIR):
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

    print("Filtering dataset...")

    if not dataset.is_file() or not dataset_images.is_dir():
        print("ERROR: dataset or images directory not found.")
        return
    # Resolve the images dir once — avoids one .resolve() call per image in the loop
    resolved_images_dir = dataset_images.resolve()


    # Build the hash map
    print("Indexing images directory...")
    hash_map = _build_hash_map(dataset_images)
    if hash_map is None:
        print("ERROR: could not index the images directory.")
        return


    # Create the output directories if they don't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)


    # Filter
    try:
        with open(dataset, mode="r", encoding="utf-8") as infile, \
             open(output_csv, mode="w", encoding="utf-8", newline="") as outfile:

            # Define the CSV columns
            cols = ["id", "crowdtangle_score", "filename"]
            # Initialize the CSV DictWriter
            writer = csv.DictWriter(outfile, fieldnames=cols)
            # Write the header
            writer.writeheader()


            # Buffer to keep only the best-performing post for each image (same filename stem)
            best = {}
            for line in infile:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: skipping corrupted JSON line, {e}.")
                    continue

                # Filters
                image = _is_valid_post(data, hash_map)
                if image is None:
                    continue

                # Save only interesting columns
                row = {
                    "id":                 data.get("id"),
                    "crowdtangle_score":  data.get("score"),
                    "filename":           image.stem
                }

                # May be a duplicate image (same filename stem) across multiple posts, keep only the best-performing post for each image
                current = best.get(row["filename"])
                if current is None or row["crowdtangle_score"] > current[0]["crowdtangle_score"]:
                    best[row["filename"]] = (row, image)


            # Order the output by filename (hash stem), ascending
            rows = sorted(best.values(), key=lambda item: item[0]["filename"])

            for row, image in rows:
                # Write the row to the CSV
                writer.writerow(row)

                # Create symlink
                symlink = output_images / image.name
                if not (symlink.exists() or symlink.is_symlink()):
                    symlink.symlink_to(resolved_images_dir / image.name)
        print("Dataset filtered and saved successfully.")

    except Exception as e:
        print(f"ERROR: processing dataset: {e}")


# =============================================================================
# (2) FILTERED DATASET DISTRIBUTION
# =============================================================================

def distribution(dataset: Path = FILTERED_DATASET_FILE, report: Path = DISTRIBUTION_FILE, plot: Path = DISTRIBUTION_PLOT_FILE):
    """
    Analyzes the distribution of the Crowdtangle score: whether it is centered on zero and how many outliers it has.

    Args:
        dataset: Path to the dataset CSV (must contain crowdtangle_score).
        report:  Output file path where the distribution report will be written.
        plot:    Base path used to derive the two output SVG plot files:
                 "<plot.stem>_histogram.svg" and "<plot.stem>_boxplot.svg", saved in plot's parent directory.

    Returns:
        None.
    """
    
    print("Analyzing the distribution of Crowdtangle scores...")
    # Load dataset
    df = load_csv(dataset, ["crowdtangle_score"])
    if df is None:
        return
    if df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return


    # Distribution
    scores = df["crowdtangle_score"]
    del df

    """
    A distribution is 'centered on zero' when its median sits close to 0.
    Comparing the median with the mean reveals how strongly extreme values drag the average, while the skewness quantifies the asymmetry of the tails.
    Tukey fences flag outliers as the points lying farther than 1.5 * IQR (inter-quartile range) from the quartiles.
    """
    # Quartiles and Tukey fences (IQR-based outlier bounds)
    q1, q3 = scores.quantile([0.25, 0.75])
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = scores[(scores < low_fence) | (scores > high_fence)]


    # Report
    len_score, len_outliers = len(scores), len(outliers)
    text = (
        f"{'=' * 77}\n"
        "FILTERED DATASET DISTRIBUTION OF CROWDTANGLE SCORES\n"
        f"{'=' * 77}\n"
        f"Rows analyzed: {len_score}\n"
        f"{scores.describe().to_string()}\n"
        f"Median: {scores.median():.3f} (mean: {scores.mean():.3f})\n"
        f"Skewness: {scores.skew():.3f}\n"
        f"Tukey fences: [{low_fence:.3f}, {high_fence:.3f}]\n"
        f"Outliers: {len_outliers} ({100 * len_outliers / len_score:.1f}%)"
    )

    # Save
    save_report(text, report)
    print("Distribution analyzed successfully.")


    # Plot: histogram clipped to the fences (the bulk) and boxplot over the full range (the outliers)
    print("Plotting the distribution...")
    # Create the output directory if it doesn't exist
    plot.parent.mkdir(parents=True, exist_ok=True)
    histogram_plot = plot.with_name(f"{plot.stem}_istogramma.pdf")
    boxplot_plot = plot.with_name(f"{plot.stem}_boxplot.pdf")

    # Histogram: clip to the fences so the outliers don't squash the bulk into a single bar
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    # Bars in the main color, with a light separator between adjacent bars
    scores.clip(low_fence, high_fence).hist(
        bins=40, ax=ax_hist, color=PLOT_MAIN_COLOR, edgecolor="white", linewidth=0.8
    )
    # Reference dotted black line at zero to judge the centering
    ax_hist.axvline(0, color="black", ls="--")
    ax_hist.set_title("Distribuzione del punteggio di engagement")
    ax_hist.set_xlabel("Punteggio di engagement")
    ax_hist.set_ylabel("Numero di immagini")
    # Clean background: white, no horizontal gridlines, only light vertical gridlines running the full height at each x tick
    fig_hist.patch.set_facecolor("white")
    ax_hist.set_facecolor("white")
    ax_hist.set_axisbelow(True)
    ax_hist.grid(axis="y", visible=False)
    ax_hist.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    fig_hist.tight_layout()
    fig_hist.savefig(histogram_plot, bbox_inches="tight")
    plt.close(fig_hist)

    # Boxplot: full range (no clip) so the outliers are visible as separate points
    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    ax_box.boxplot(
        scores, orientation="horizontal",
        boxprops=dict(color=PLOT_MAIN_COLOR),
        whiskerprops=dict(color=PLOT_MAIN_COLOR),
        capprops=dict(color=PLOT_MAIN_COLOR),
        medianprops=dict(color=PLOT_MAIN_COLOR),
        flierprops=dict(markeredgecolor=PLOT_SECOND_COLOR),
    )
    # Reference dotted black line at zero to judge the centering
    ax_box.axvline(0, color="black", ls="--")
    ax_box.set_title("Boxplot sull'intero campione")
    ax_box.set_xlabel("Punteggio di engagement")
    fig_box.tight_layout()
    fig_box.savefig(boxplot_plot, bbox_inches="tight")
    plt.close(fig_box)

    print("Plots saved successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    (1) Filters the dataset, producing the filtered CSV and the symlinked images directory.
    (2) Calculates the distribution of Crowdtangle scores and plots it.

    Args:
        None (arguments come from config.py).

    Returns:
        None.
    """

    # Filter dataset
    filter()

    # Distribution
    distribution()

if __name__ == "__main__":
    main()