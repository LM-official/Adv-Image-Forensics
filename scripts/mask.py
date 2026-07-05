import argparse
from   config   import (FILTERED_IMGS_DIR, NPZ_DIR, MASKED_DIR,
                        MASK_THRESHOLD, MASK_CONFIDENCE_THRESHOLD,
                        MASK_INVERT_COMPOSITE, MASK_FLAT_GRAD_THRESH, MASK_INVERT_MARGIN,
                        MASK_INVERT_MIN_AREA, MASK_MORPH_KERNEL, MASK_MIN_AREA_RATIO, 
                        MASKED_TRUTH_DIR, MASK_TRUTH_THRESHOLD, MASK_CONFUSION_MATRIX_FILE)
import cv2
import numpy     as     np
from   pathlib   import Path
from   tqdm      import tqdm
from   utils     import load_npz, save_report


# =============================================================================
# (1) MASK
# =============================================================================

def _clean_mask(binary_mask: np.ndarray, kernel_size: int, min_area_ratio: float) -> np.ndarray:
    """
    Cleans a raw binary mask so the surviving region matches the real manipulated area instead of noise.

    TruFor's anomaly map is grainy and leaves small holes inside solid regions. This is produced strange masks.
    Three steps fix it:
    1. Morphological OPEN  (erode then dilate) erases the isolated specks / salt-and-pepper noise.
    2. Morphological CLOSE (dilate then erode) fills the small holes inside an otherwise solid region.
    3. Connected-component filtering drops every blob smaller than `min_area_ratio` of the image,
       removing the scattered fragments while keeping the main manipulated object(s).

    Args:
        binary_mask:    uint8 mask (0 or 255).
        kernel_size:    Side of the square structuring element for open/close (<= 1 disables morphology).
        min_area_ratio: Minimum blob area, as a fraction of the whole image, to keep (<= 0 disables filtering).

    Returns:
        The cleaned uint8 mask (0 or 255).
    """

    # Morphological denoising: open removes specks, close fills holes
    if kernel_size > 1:
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


    # Connected-component area filtering: keep only blobs that are large enough to be a real region
    if min_area_ratio > 0:
        # Label every separate blob (label 0 is the background)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if count > 1:
            min_area    = min_area_ratio * binary_mask.size
            binary_mask = np.zeros_like(binary_mask)
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    binary_mask[labels == i] = 255

    return binary_mask


def _is_inverted_composite(image: np.ndarray, binary_mask: np.ndarray, grad_thresh: float, margin: float, min_area: float) -> bool:
    """
    TruFor flags the region whose statistics differ from the image majority, assuming the majority is authentic.
    On a real photo composited over a flat colored banner, the banner is the majority,
    so TruFor flags the (textured) person instead of the (flat) synthetic overlay and the mask keeps the wrong part.
    Such images have a clear texture asymmetry: the flagged region is photographic (rich texture) while the background is flat.
    We measure that and, when the background is flatter than the flagged region by at least `margin`, report the mask as inverted (caller should flip it).

    Args:
        image:       Original BGR image (as read by cv2).
        binary_mask: uint8 mask (0 or 255) flagging the pixels TruFor considers altered.
        grad_thresh: Gradient magnitude below which a pixel counts as "flat" (no texture).
        margin:      Minimum (background flat-fraction - flagged-region flat-fraction) to call it inverted.
        min_area:    Minimum flagged fraction of the image required to consider inverting (guards against
                     flipping a tiny noise speck on a near-authentic image into a fully-colored mask).

    Returns:
        True if the mask looks polarity-inverted (textured region on a flat background), else False.
    """

    fg = binary_mask.astype(bool)
    # An empty or full mask has no foreground/background contrast to compare; a tiny flagged region is
    # just noise (TruFor found nothing real), so inverting it would wrongly keep almost the whole image
    if not fg.any() or fg.all() or fg.mean() < min_area:
        return False

    # Per-pixel texture: magnitude of the grayscale gradient; "flat" = little/no texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    flat = np.sqrt(gx * gx + gy * gy) < grad_thresh

    # Invert when the flagged region is photographic (low flat-fraction) but the background is a flat banner
    fg_flat = float(flat[fg].mean())
    bg_flat = float(flat[~fg].mean())
    return (bg_flat - fg_flat) >= margin


def mask(images: Path = FILTERED_IMGS_DIR, npz: Path = NPZ_DIR, masked: Path = MASKED_DIR,
         threshold: float = MASK_THRESHOLD, conf_threshold: float = MASK_CONFIDENCE_THRESHOLD,
         invert_composite: bool = MASK_INVERT_COMPOSITE, flat_grad_thresh: float = MASK_FLAT_GRAD_THRESH,
         invert_margin: float = MASK_INVERT_MARGIN, invert_min_area: float = MASK_INVERT_MIN_AREA,
         kernel_size: int = MASK_MORPH_KERNEL, min_area_ratio: float = MASK_MIN_AREA_RATIO):
    """
    Creates a masked image for every .npz file in `npz`, using the matching original image in `images`.
    Masked images are saved as PNG in `masked`.

    The mask keeps only the pixels that TruFor flags as both manipulated AND confidently localized:
    a pixel survives when its anomaly value is >= `threshold` and its confidence is >= `conf_threshold`.
    Using the confidence map avoids the strange masks produced by the anomaly map alone where low-confidence regions leak into the result.

    PNG is used because it is lossless AND, being mostly black (only the manipulated region is visible),
    it compresses well: this keeps the masks folder small on disk.

    On synthetic-majority (a real photo over a flat banner),
    TruFor flags the real person instead of the overlay; when `invert_composite` is set, such masks are
    detected (see `_is_inverted_composite`) and flipped so the overlay is kept instead of the subject.

    The raw mask is then cleaned (see `_clean_mask`) to drop scattered noise.

    Args:
        images:           Directory containing the original input images.
        npz:              Directory containing the TruFor .npz files with the anomaly and confidence maps.
        masked:           Output directory where the masked PNG images are saved.
        threshold:        Binarization threshold in [0.0, 1.0] applied to the anomaly map.
        conf_threshold:   Confidence threshold in [0.0, 1.0] applied to the confidence map.
        invert_composite: If True, flip the mask on detected synthetic-majority composites (polarity fix).
        flat_grad_thresh: Gradient magnitude below which a pixel counts as "flat" in the inversion test.
        invert_margin:    Min (background - foreground) flat-fraction gap that triggers an inversion.
        invert_min_area:  Min flagged fraction of the image required before an inversion is considered.
        kernel_size:      Side of the morphological open/close kernel used to denoise the mask.
        min_area_ratio:   Minimum area (fraction of the image) kept by the mask cleanup.

    Returns:
        None.
    """

    if not images.is_dir() or not npz.is_dir():
        print("ERROR: invalid input directories for creating masked images.")
        return


    # Maximum PNG compression (0 = none, 9 = max): smaller files at the cost of a bit of CPU
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    # Create the output directory if it doesn't exist
    masked.mkdir(parents=True, exist_ok = True)

    # Masking
    print("Creating masked images for all images...")
    for item in tqdm(npz.glob("*.npz"), desc="Masking", unit="img"):
        # TruFor appends .npz to the original filename, so .stem is the original image name (with extension)
        image_name = item.stem

        # Load image
        image = cv2.imread(images / image_name)
        if image is None:
            print("WARNING: image not found or corrupted at path.")
            continue

        # Load the anomaly map (where TruFor sees manipulation) and the confidence map (how reliable that is)
        anomaly_df = load_npz(item, {"map", "conf"})
        if anomaly_df is None:
            continue
        anomaly    = np.array(anomaly_df["map"].iloc[0])
        confidence = np.array(anomaly_df["conf"].iloc[0])
        # Resize both maps to match the image pixels (width, height)
        # cv2.resize expects (width, height), while numpy shape is (height, width)
        anomaly    = cv2.resize(anomaly,    (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarization: keep a pixel only if TruFor flagged it as altered AND is sure of that call.
        # 'conf' is TruFor's per-pixel reliability map (TCP): high = sure of its decision.
        # So requiring anomaly >= threshold AND confidence >= conf_threshold
        # keeps the confidently-altered pixels and deletes authentic pixels and the ones TruFor flagged but is unsure about.
        # cv2 requires the mask to be uint8 format (0 or 255) for bitwise operations
        binary_mask = ((anomaly >= threshold) & (confidence >= conf_threshold)).astype(np.uint8) * 255

        # Free the large full-resolution maps: drops their only references, so CPython reclaims them
        del anomaly, confidence

        # Fix TruFor's polarity on synthetic-majority composites: if the flagged region is photographic
        # and the background is a flat banner, TruFor flagged the real subject, so keep the overlay instead.
        if invert_composite and _is_inverted_composite(image, binary_mask, flat_grad_thresh, invert_margin, invert_min_area):
            binary_mask = cv2.bitwise_not(binary_mask)

        # Remove the scattered specks and fill the holes left by the raw threshold
        binary_mask = _clean_mask(binary_mask, kernel_size, min_area_ratio)
        # Save one mask per image for full coverage: when nothing survives, the PNG is all black,
        # which correctly means "TruFor found no confident manipulation here".

        # Apply the mask
        image = cv2.bitwise_and(image, image, mask=binary_mask)
        # Save as compressed PNG
        cv2.imwrite(masked / f"{Path(image_name).stem}.png", image, png_params)

    print("Masked images saved successfully.")


# =============================================================================
# (2) CONFUSION MATRIX
# =============================================================================

def confusion_matrix(masked: Path = MASKED_DIR, truth: Path = MASKED_TRUTH_DIR,
                     report: Path = MASK_CONFUSION_MATRIX_FILE, truth_threshold: int = MASK_TRUTH_THRESHOLD):
    """
    Computes the pixel-level confusion matrix of the generated masks against the ground truth,
    then saves accuracy, precision, recall, F1-score and IoU.

    Every pixel of every image is one sample:
    - prediction: a pixel is positive when the masked PNG keeps it (any non-zero channel);
    - truth:      a pixel is positive when the ground-truth mask paints it white (gray value > truth_threshold).

    Args:
        masked:          Directory containing the masked PNG images produced by `mask()`.
        truth:           Directory containing the ground-truth masks (white = manipulated, black = authentic).
        truth_threshold: Gray value above which a ground-truth pixel counts as manipulated (kills the noise).
        report:          Output file path where the performance report is written.

    Returns:
        None.
    """

    if not masked.is_dir():
        print("ERROR: invalid masked directory for computing the confusion matrix.")
        return

    # The ground truth is optional. Without it there is nothing to score against
    if not truth.is_dir():
        print("Skipping mask confusion matrix: no ground-truth directory found.")
        return


    # Compute performance
    print(f"Computing the pixel-level confusion matrix with truth threshold {truth_threshold}...")

    # Global pixel counters (every pixel of every image is one sample)
    # true positives  (manipulated pixel -> correctly kept)
    # true negatives  (authentic pixel -> correctly removed)
    # false positives (authentic pixel -> mistakenly kept)
    # false negatives (manipulated pixel -> mistakenly removed)
    tp = tn = fp = fn = 0

    # Per-image IoU, to spot which masks are wrong (the global counters alone hide them)
    ious = []
    for item in masked.iterdir():
        # Load masked image
        image = cv2.imread(item)
        if image is None:
            print("WARNING: masked image not found or corrupted.")
            continue

        # Load the matching ground-truth mask
        truth_mask = cv2.imread(truth / f"{item.stem}.jpg", cv2.IMREAD_GRAYSCALE)
        if truth_mask is None:
            print("WARNING: ground-truth mask not found or corrupted at path.")
            continue

        # Prediction: masked-out pixels are exactly 0, so a pixel is predicted manipulated when any channel is non-zero
        prediction = image.max(axis=2) > 0
        # Truth: white = manipulated; binarize to kill the noise
        label      = truth_mask > truth_threshold
        # Resize the truth to the prediction pixels if the does not match exactly
        # cv2.resize expects (width, height), while numpy shape is (height, width)
        if label.shape != prediction.shape:
            label = cv2.resize(label.astype(np.uint8), (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


        # Per-image pixel counters
        img_tp = int(( label &  prediction).sum())
        img_tn = int((~label & ~prediction).sum())
        img_fp = int((~label &  prediction).sum())
        img_fn = int(( label & ~prediction).sum())
        tp += img_tp
        tn += img_tn
        fp += img_fp
        fn += img_fn

        # IoU of the image
        # when both masks are empty (union = 0) prediction and truth agree on "nothing manipulated here", which is a perfect score
        union = img_tp + img_fp + img_fn
        ious.append((item.stem, img_tp / union if union > 0 else 1.0))


    total = tp + tn + fp + fn
    if total == 0:
        print("ERROR: no masked/ground-truth pairs found to score.")
        return


    """
    accuracy:
    Measures the overall correctness of the masks.
    It represents the ratio of correctly classified pixels (both true positives and true negatives) to the total number of pixels.
    It can be misleading here: on a mostly-authentic image the huge black background is easy to get right and inflates the score.
    """
    accuracy = (tp + tn) / total

    """
    precision:
    Measures the reliability of the kept pixels.
    'Out of all the pixels the mask kept, how many were actually manipulated?'
    A high precision indicates a low rate of false positives (authentic pixels leaking into the mask).
    """
    # tp + fp > 0 check to avoid division by zero when the masks kept no pixel at all
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    """
    recall (sensitivity):
    Measures the mask's ability to cover the whole manipulated region.
    'Out of all the truly manipulated pixels, how many did the mask keep?'
    A high recall indicates a low rate of false negatives (manipulated pixels wrongly removed).
    """
    # tp + fn > 0 check to avoid division by zero when the ground truth marks no pixel as manipulated
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    """
    f1-score:
    The harmonic mean of precision and recall.
    Since precision and recall often have an inverse relationship (improving one can worsen the other),
    the f1-score provides a single, balanced metric.
    It is widely considered the most robust and reliable indicator of a model's performance,
    especially when dealing with imbalanced data.
    """
    # precision + recall > 0 check to avoid division by zero when both precision and recall are zero
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    """
    IoU (Jaccard index):
    The standard segmentation metric: intersection / union of the predicted and true manipulated regions.
    Unlike accuracy it ignores the true negatives, so the (often huge) black background cannot inflate the score:
    only how well the two regions coincide counts.
    Reported both globally (all pixels pooled, so big images weigh more) and
    as the mean of the per-image values (every image weighs the same).
    """
    # tp + fp + fn > 0 check: when both prediction and truth are empty everywhere they perfectly agree
    iou      = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    mean_iou = sum(value for _, value in ious) / len(ious)

    print("Confusion matrix computed successfully.")

    # Report
    text = (
        f"{'=' * 77}\n"
        "MASK CONFUSION MATRIX\n"
        f"{'=' * 77}\n"
        f"Total images analyzed: {len(ious)}\n"
        f"Total pixels analyzed: {total}\n"
        f"Truth threshold: {truth_threshold}\n\n"
        f"Accuracy  :\t{accuracy:.2%}\t(percentage of correctly classified pixels)\n"
        f"Precision :\t{precision:.2%}\t(reliability of the kept pixels)\n"
        f"Recall    :\t{recall:.2%}\t(ability to keep all the manipulated pixels)\n"
        f"F1-score  :\t{f1_score:.4f}\t(final harmonic mean)\n"
        f"IoU       :\t{iou:.4f}\t(overlap / union of the manipulated regions, all pixels pooled)\n"
        f"Mean IoU  :\t{mean_iou:.4f}\t(average of the per-image IoU below)\n\n"
        "Confusion matrix (pixels):\n"
        f"True positives  :\t{tp}\n"
        f"True negatives  :\t{tn}\n"
        f"False positives :\t{fp}\t(authentic pixels mistakenly kept)\n"
        f"False negatives :\t{fn}\t(manipulated pixels mistakenly removed)\n"
    )

    # Per-image IoU, worst first: points straight at the masks to inspect by eye
    lines = ["\nPer-image IoU (worst first):"]
    for filename, value in sorted(ious, key=lambda pair: pair[1]):
        lines.append(f"  '{filename}', IoU: {value:.4f}")
    text += "\n".join(lines)

    # Save
    save_report(text, report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    (1) Mask the manipulated region of every image using its TruFor .npz output.
    (2) Compute the confusion matrix of the masks against the ground truth.

    Args:
        None (paths and the cleanup/inversion tuning come from config.py, the project backbone).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Mask the manipulated region of every image using its TruFor .npz output.")

    # Flags
    # The cleanup/inversion knobs stay in config.py
    parser.add_argument("--threshold",       type=float, default=MASK_THRESHOLD,            help=f"Anomaly binarization threshold (0.0 to 1.0). Default: {MASK_THRESHOLD}")
    parser.add_argument("--conf-threshold",  type=float, default=MASK_CONFIDENCE_THRESHOLD, help=f"Confidence threshold (0.0 to 1.0). Default: {MASK_CONFIDENCE_THRESHOLD}")
    parser.add_argument("--truth-threshold", type=float, default=MASK_TRUTH_THRESHOLD,      help=f"Ground-truth binarization threshold (0 to 255). Default: {MASK_TRUTH_THRESHOLD}")
    args = parser.parse_args()

    # Create the masked images
    mask(threshold=args.threshold, conf_threshold=args.conf_threshold)

    # Compute the pixel-level confusion matrix against the hand-made ground truth
    confusion_matrix(truth_threshold=args.truth_threshold)

if __name__ == "__main__":
    main()