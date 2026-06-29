import argparse
from   config            import (FILTERED_DATASET_FILE, FILTERED_IMGS_DIR, IMAGE_FEATURES_FILE, IMAGE_FEATURES_PLOT_FILE,
                                 FEATURE_PLOT_X_TARGET, FEATURE_PLOT_Y_TARGET, FEATURE_PLOT_X_LABEL, FEATURE_PLOT_Y_LABEL,
                                 FEATURE_CORR_METHOD, FEATURE_RESIZE_LONG_SIDE,
                                 FEATURE_CANNY_LOW, FEATURE_CANNY_HIGH, FEATURE_THIRDS_RADIUS)
import cv2
import matplotlib
matplotlib.use("Agg") # Non-interactive backend: render straight to file, no display needed
import matplotlib.pyplot as     plt
import numpy             as     np
import pandas            as     pd
from   pathlib           import Path
from   scipy             import stats
from   skimage.measure   import shannon_entropy
from   tqdm              import tqdm
from   utils             import load_csv, save_report, update_csv


# Canonical order of the classical features produced by extract_features()
# Used by the plot to pick the feature columns out of the dataset CSV
FEATURE_NAMES = (
    # Format
    "aspect_ratio",
    # Tone / exposure
    "brightness", "contrast", "dynamic_range",
    # Focus / sharpness
    "sharpness", "clearness", "blur", "depth_of_field",
    # Color
    "colorfulness", "saturation", "warmth", "hue_mean",
    # Intensity distribution
    "kurtosis", "skewness", "shannon_entropy",
    # Composition / structure
    "symmetry", "diagonal_dominance", "rule_of_thirds", "visual_complexity",
)


# =============================================================================
# (1) HELPER FEATURE EXTRACTION
# =============================================================================

def _luminance(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perceived brightness of each pixel (Rec. 601 weights), as a 2-D map in the 0-255 range.

    The eye is most sensitive to green and least to blue, so the channels are not averaged equally but weighted.

    Args:
        r: Red channel as a 2-D float array, values in 0-255.
        g: Green channel as a 2-D float array, values in 0-255.
        b: Blue channel as a 2-D float array, values in 0-255.

    Returns:
        A 2-D float array of per-pixel luminance (same height and width as the channels), in 0-255.
    """

    return r * 0.299 + g * 0.587 + b * 0.114


def _crete_blur(gray: np.ndarray) -> float:
    """
    Crete-Roffet (2007) blur score, from 0 (sharp) to 1 (very blurry).
    It needs no reference image, only the picture itself.

    The trick: blur the image again and see how much it changes.
    A sharp image has crisp edges that get clearly softened, so it changes a lot.
    An already-blurry image has little detail left to lose, so it barely changes.
    (It compares neighbouring pixels, so it measures something different from the Laplacian-variance 'sharpness' feature).

    Args:
        gray: Grayscale image as a 2-D float array, values in 0-255.

    Returns:
        The blur score as a float in [0, 1] (0 = sharp, 1 = very blurry).
    """

    # Re-blur the image: once horizontally (9-wide box) and once vertically
    blur_ver = cv2.blur(gray, (1, 9))
    blur_hor = cv2.blur(gray, (9, 1))

    # Local variation = how much each pixel differs from its neighbour, before and after re-blurring
    dif_or_ver   = np.abs(np.diff(gray,     axis=0)) # original, top-to-bottom
    dif_or_hor   = np.abs(np.diff(gray,     axis=1)) # original, left-to-right
    dif_blur_ver = np.abs(np.diff(blur_ver, axis=0)) # re-blurred, top-to-bottom
    dif_blur_hor = np.abs(np.diff(blur_hor, axis=1)) # re-blurred, left-to-right

    # How much variation the extra blur destroyed (clamped so it is never negative)
    var_ver = np.maximum(0.0, dif_or_ver - dif_blur_ver)
    var_hor = np.maximum(0.0, dif_or_hor - dif_blur_hor)

    # Fraction of the original variation that was KEPT = 1 - lost/total.
    # Keeping little (a lot was lost) means the image was sharp -> low blur score.
    sum_or_ver, sum_or_hor = dif_or_ver.sum(), dif_or_hor.sum()
    score_ver = (sum_or_ver - var_ver.sum()) / sum_or_ver if sum_or_ver else 0.0
    score_hor = (sum_or_hor - var_hor.sum()) / sum_or_hor if sum_or_hor else 0.0
    # Blurrier of the two directions
    return float(max(score_ver, score_hor))


def _depth_of_field(gray: np.ndarray) -> float:
    """
    Tells apart a photo with a sharp subject on a blurry background
    (shallow depth of field, typical of portraits / good cameras) from a photo that is sharp everywhere.

    It splits the image into the central third and the surrounding border,
    measures how much fine detail each part has, and divides them:
        high ratio -> sharp middle, soft border (shallow depth of field)
        ~ 1        -> equally sharp everywhere
    Detail is measured with the Laplacian (an edge/detail filter); 
    its variance is large where there is a lot of fine detail and small on smooth, blurry areas.

    Args:
        gray: Grayscale image as a 2-D float array, values in 0-255.

    Returns:
        The center-to-border detail ratio as a float
        (high = sharp subject on a soft background; ~1 = sharp everywhere).
    """

    h, w = gray.shape
    # Detail map: strong where the image is sharp
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    # Bounds of the central third, computed once and reused for both the center and the border
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3

    # Central third (the likely subject)
    center = lap[y0:y1, x0:x1]
    # Degenerate tiny image (a dimension < 3): no central region to measure
    if center.size == 0:
        return 0.0
    # Everything outside that centre = the border
    mask = np.ones_like(lap, dtype=bool)
    mask[y0:y1, x0:x1] = False
    border_var = lap[mask].var()

    # Ratio of center detail to border detail. The + 1e-6 is a tiny safety term on the denominator:
    # a perfectly smooth border (e.g. a flat colored backdrop) has variance 0,
    # and dividing by 0 would give inf or NaN. 1e-6 is far smaller than any real Laplacian variance,
    # so it prevents that crash without noticeably changing the ratio for normal images
    return float(center.var() / (border_var + 1e-6))


def _colorfulness(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
    """
    Hasler & Susstrunk (2003) colorfulness metric: 0 for a gray image,
    larger when the colors are more saturated and more varied.

    Instead of looking at R/G/B directly it uses two "color opposition" channels,
    which is closer to how we judge color:
        rg = red - green              (red vs green)
        yb = (red + green)/2 - blue   (yellow vs blue)
    A colorful image both spreads out a lot on these channels (big standard deviation) and
    sits far from neutral gray on average (big mean), so the score adds those two together.
    sqrt(a*a + b*b) is used to merge the rg and yb.

    Args:
        r: Red channel as a 2-D float array, values in 0-255.
        g: Green channel as a 2-D float array, values in 0-255.
        b: Blue channel as a 2-D float array, values in 0-255.

    Returns:
        The colorfulness score as a float: 0 for a gray image, larger for more colorful ones.
    """

    rg = r - g
    yb = (r + g) * 0.5 - b
    # how much the colors vary (spread) and how far from gray they are on average (distance)
    spread   = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    distance = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(spread + 0.3 * distance)


def _symmetry(gray: np.ndarray) -> float:
    """
    Left-right mirror symmetry, from 0 (very asymmetric) to 1 (perfectly symmetric).

    We flip the right half of the image like a mirror and check how well it matches the left half.
    A small average pixel difference means the halves look alike -> high symmetry.

    Args:
        gray: Grayscale image as a 2-D float array, values in 0-255.

    Returns:
        The mirror-symmetry score as a float in [0, 1] (1 = perfectly symmetric).
    """

    # Split at the vertical center; for an odd width the middle column is left out of both halves
    w     = gray.shape[1]
    half  = w // 2
    # A 1-pixel-wide image is its own mirror image -> perfectly symmetric
    if half == 0:
        return 1.0
    left  = gray[:, :half]
    right = np.fliplr(gray[:, w - half:])

    # Average brightness difference per pixel, scaled to 0-1, flipped so 1 = identical halves
    return float(1.0 - np.abs(left - right).mean() / 255.0)


def _diagonal_dominance(gray: np.ndarray) -> float:
    """
    How much of the image's edge strength runs along the diagonals (about 45 deg or 135 deg)
    instead of straight horizontal/vertical.
    Higher = a more diagonal, dynamic composition.

    Args:
        gray: Grayscale image as a 2-D float array, values in 0-255.

    Returns:
        The share of total edge strength carried by diagonal edges, as a float in [0, 1]
        (0.0 when the image is perfectly flat).
    """

    # Sobel measures the brightness slope in x and y
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Edge strength = gradient magnitude
    mag = np.sqrt(gx * gx + gy * gy)

    total = mag.sum()
    # A perfectly flat image has no edges at all
    if total <= 0:
        return 0.0

    # Edge direction: edge and edge rotated 180 deg are the same line, so fold the angle into [0, 180)
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    # Keep the edges pointing within 22.5 deg of either diagonal (45 or 135)
    near_diag = (np.abs(angle - 45.0) <= 22.5) | (np.abs(angle - 135.0) <= 22.5)
    # Share of the total edge strength carried by diagonal edges
    return float(mag[near_diag].sum() / total)


def _rule_of_thirds(gray: np.ndarray, radius_frac: float) -> float:
    """
    Photographers often place the subject on the "thirds": imagine a 3x3 grid over the image;
    the four points where its lines cross are the strong spots ("power points").

    This checks whether the busy/detailed parts of the image sit on those four points.
    "Busy" is measured by edge strength (the gradient magnitude).
    We compare the average edge strength inside small circles around the four points to the average over the whole image:
        > 1  -> the interesting content really is on the thirds points
        ~ 1  -> it is spread out evenly

    Args:
        gray:        Grayscale image as a 2-D float array, values in 0-255.
        radius_frac: Radius of each power-point circle, as a fraction of the image diagonal.

    Returns:
        The ratio of mean edge strength on the power-points to the overall mean, as a float
        (0.0 when the image is flat or the circles are too small to cover any pixel).
    """

    # Edge strength at every pixel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sal = np.sqrt(gx * gx + gy * gy)

    global_mean = sal.mean()
    # Flat image, no edges anywhere
    if global_mean <= 0:
        return 0.0

    # Circle radius as a fraction of the image diagonal, so it scales with the image size
    h, w = gray.shape
    radius = radius_frac * np.sqrt(w * w + h * h)

    # Mark every pixel that falls inside one of the four power-point circles
    ys, xs = np.ogrid[:h, :w]
    disks = np.zeros((h, w), dtype=bool)
    for px in (w / 3.0, 2.0 * w / 3.0):
        for py in (h / 3.0, 2.0 * h / 3.0):
            mask = ((xs - px) ** 2 + (ys - py) ** 2) <= radius ** 2
            disks |= mask
    # Circles too small to cover any pixel
    if not disks.any():
        return 0.0

    # Average edge strength on the power points vs over the whole image
    return float(sal[disks].mean() / global_mean)


# =============================================================================
# (1) EXTRACT FEATURES
# =============================================================================

def extract_features(image_path: Path) -> dict | None:
    """
    Computes the visual features for a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        A dict mapping every feature name to a finite float, or None if the image could not be read.
    """

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    # Aspect ratio is resolution-independent; read it before any resize
    h, w = bgr.shape[:2]
    aspect_ratio = w / h if h else 0.0

    # Downscale large images so the cost is uniform across photos; FEATURE_RESIZE_LONG_SIDE = 0 disables it
    long_side = max(h, w)
    if FEATURE_RESIZE_LONG_SIDE and long_side > FEATURE_RESIZE_LONG_SIDE:
        scale = FEATURE_RESIZE_LONG_SIDE / long_side
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b    = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum     = _luminance(r, g, b)

    p1, p99 = np.percentile(lum, [1, 99])

    features = {
        # Format
        "aspect_ratio":       aspect_ratio,

        # Tone / exposure
        "brightness":         float(lum.mean() / 255.0),
        "contrast":           float(lum.std() / 255.0),                      # RMS contrast
        "dynamic_range":      float((p99 - p1) / 255.0),                     # robust luminance spread

        # Focus / sharpness
        "sharpness":          float(cv2.Laplacian(gray, cv2.CV_32F).var()),  # Laplacian variance
        "clearness":          float(np.sqrt(((gray - cv2.GaussianBlur(gray, (0, 0), 2.0)) ** 2).mean()) / 255.0),  # acutance / local detail
        "blur":               _crete_blur(gray),                             # perceptual blur, distinct from sharpness
        "depth_of_field":     _depth_of_field(gray),

        # Color
        "colorfulness":       _colorfulness(r, g, b),
        "saturation":         float(hsv[..., 1].mean() / 255.0),
        "warmth":             float(np.mean((r - b) / (r + b + 1e-6))),      # warm vs cool balance, [-1, 1]
        "hue_mean":           float(hsv[..., 0].mean() / 179.0),             # OpenCV hue is 0-179

        # Intensity distribution
        "kurtosis":           float(stats.kurtosis(gray, axis=None)),        # excess kurtosis of intensities
        "skewness":           float(stats.skew(gray, axis=None)),
        "shannon_entropy":    float(shannon_entropy(gray)),

        # Composition / structure
        "symmetry":           _symmetry(gray),
        "diagonal_dominance": _diagonal_dominance(gray),
        "rule_of_thirds":     _rule_of_thirds(gray, FEATURE_THIRDS_RADIUS),
        "visual_complexity":  float(np.mean(cv2.Canny(gray.astype(np.uint8), FEATURE_CANNY_LOW, FEATURE_CANNY_HIGH) > 0)), # edge density
    }

    # Replace any non-finite value (e.g. kurtosis of a flat image) with 0
    return {name: (float(value) if np.isfinite(value) else 0.0) for name, value in features.items()}


def features(dataset: Path = FILTERED_DATASET_FILE, images: Path = FILTERED_IMGS_DIR):
    """
    Extracts the features for every image in `images` and
    merges them, by filename, into the dataset CSV (one column per feature).

    Args:
        dataset: Path to the dataset CSV the features are merged into (merge key: 'filename').
        images:  Directory of images.

    Returns:
        None.
    """

    if not images.is_dir():
        print(f"ERROR: images directory '{images}' not found.")
        return
    
    # Get image paths, excluding hidden files
    imgs = [path for path in images.iterdir() if not path.name.startswith(".")]
    if not imgs:
        print(f"ERROR: No valid images found in '{images}'.")
        return


    print(f"Extracting features from {len(imgs)} images in '{images}'...")
    rows = []
    for image in tqdm(imgs, desc="Features"):
        feats = extract_features(image)
        if feats is None:
            continue
        rows.append({"filename": image.stem, **feats})

    if not rows:
        print("ERROR: no features could be extracted.")
        return

    # Update dataset
    data = pd.DataFrame(rows)
    print(f"Extracted {len(data)} feature rows ({len(FEATURE_NAMES)} features each).")
    update_csv(dataset, data)


# =============================================================================
# (3) PLOT
# =============================================================================

def _feature_correlations(df: pd.DataFrame, target: str, feature_cols: list[str], method: str) -> pd.Series:
    """
    Correlation of each feature column with `target`, computed pairwise (NaNs dropped).

    Args:
        df:           Dataset containing the feature columns and the target column.
        target:       Name of the column to correlate every feature against.
        feature_cols: Names of the feature columns to correlate with the target.
        method:       Correlation method, e.g. "pearson" or "spearman".

    Returns:
        A Series indexed by feature name, holding each feature's correlation with the target.
    """

    return pd.Series({columns: df[columns].corr(df[target], method=method) for columns in feature_cols})


def plot(dataset: Path = FILTERED_DATASET_FILE, report: Path = IMAGE_FEATURES_FILE, plot: Path = IMAGE_FEATURES_PLOT_FILE,
         x_target: str = FEATURE_PLOT_X_TARGET, y_target: str = FEATURE_PLOT_Y_TARGET, method: str = FEATURE_CORR_METHOD):
    """
    Create a plot in whitch every visual feature is a single point whose coordinates are its correlation with
    the two outcome columns (x_target, y_target).
    A dashed regression line is fit through the points and a dotted identity line (y = x) is draw. for reference.
    Also writes a text report listing each feature's (x_corr, y_corr).

    Args:
        dataset:  Dataset CSV (must contain the feature columns and both target columns).
        report:   Output path for the text correlation report.
        plot:     Output path for the PNG figure.
        x_target: Column used for the X axis correlation (default crowdtangle_score).
        y_target: Column used for the Y axis correlation (default trufor_score).
        method:   Correlation method, "pearson" or "spearman".

    Returns:
        None.
    """

    df = load_csv(dataset)
    if df is None:
        return

    # Only the features actually present in the CSV
    feature_cols = [name for name in FEATURE_NAMES if name in df.columns]
    if not feature_cols:
        print("ERROR: no feature columns in the dataset.")
        return
    missing = [target for target in (x_target, y_target) if target not in df.columns]
    if missing:
        print(f"ERROR: missing target column(s) {missing}.")
        return


    print(f"Computing {method} correlations of {len(feature_cols)} features: '{x_target}' (X) and '{y_target}' (Y)...")
    corr_x = _feature_correlations(df, x_target, feature_cols, method)
    corr_y = _feature_correlations(df, y_target, feature_cols, method)


    # Eeport (sorted by |Y correlation|, strongest first)
    order = corr_y.abs().sort_values(ascending=False).index
    lines = [f"\t{name:<20} X={corr_x[name]:+.4f}  Y={corr_y[name]:+.4f}" for name in order]
    text = (
        f"{'=' * 77}\n"
        "IMAGE FEATURES\n"
        f"{'=' * 77}\n"
        f"Correlation method: {method}\n"
        f"X axis: correlation(feature, {x_target})\n"
        f"Y axis: correlation(feature, {y_target})\n"
        f"Samples: {len(df)}\n\n"
        "Features (sorted by |Y|):\n" + "\n".join(lines)
    )
    save_report(text, report)


    # Plot
    print("Plotting features...")
    # A constant feature (zero variance) has an undefined (NaN) correlation:
    # matplotlib cannot place such a point and np.polyfit would crash on it,
    # so keep only the features with finite correlations
    plot_cols = [name for name in feature_cols if np.isfinite(corr_x[name]) and np.isfinite(corr_y[name])]
    if not plot_cols:
        print("ERROR: no features have a defined correlation; nothing to plot.")
        return

    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h", "<", ">", "d", "H", "8", "o", "s", "^", "v"]
    colors  = plt.get_cmap("tab20")(np.linspace(0, 1, len(plot_cols)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(plot_cols):
        ax.scatter(corr_x[name], corr_y[name],
                   marker=markers[i % len(markers)], color=colors[i],
                   s=90, edgecolors="black", linewidths=0.5, label=name, zorder=3)

    xs = corr_x[plot_cols].to_numpy()
    ys = corr_y[plot_cols].to_numpy()
    lim = float(np.nanmax(np.abs(np.concatenate([xs, ys])))) * 1.2 or 0.1

    # Dotted identity line y = x (reference)
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="black", lw=1.2, zorder=1)
    # Dashed regression line through the feature points
    if len(xs) >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        ax.plot([-lim, lim], [slope * -lim + intercept, slope * lim + intercept],
                ls="--", color="firebrick", lw=2.5, zorder=2)

    # Solid axis lines through the origin + light grid, like the reference figure
    ax.axhline(0, color="gray", lw=1.2, zorder=0)
    ax.axvline(0, color="gray", lw=1.2, zorder=0)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(FEATURE_PLOT_X_LABEL)
    ax.set_ylabel(FEATURE_PLOT_Y_LABEL)
    ax.legend(title="Visual features", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, ncol=1, frameon=False)

    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to '{plot}'.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    (1) Extracts visual features for every filtered image and merges them into the dataset CSV,
    (2) plots (each feature's correlation with the engagement and AI-generated).

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Extract image features and plot it.")

    # Flags, many arguments come from config.py (the project backbone)
    parser.add_argument("--method", choices=("pearson", "spearman"), default=FEATURE_CORR_METHOD, help=f"Correlation method. Default: {FEATURE_CORR_METHOD}")
    args = parser.parse_args()

    # Extract featurs
    features()

    # Plot the features
    plot(method=args.method)

if __name__ == "__main__":
    main()