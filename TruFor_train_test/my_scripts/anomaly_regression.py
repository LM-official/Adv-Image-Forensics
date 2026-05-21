import argparse
from   config                          import (IMG_DIR, NPZ_DIR, MASK_THRESHOLD, MASKED_DIR,
                                               BATCH_SIZE, MAX_DESCRIPTION_TOKENS, FILTERED_DATASET_FILE,
                                               MAX_NUMBER_OF_WORDS, SVD_COMPONENTS, ANOMALY_REGRESSION_FILE)
import cv2
import gc
import numpy                           as     np
import pandas                          as     pd
from   pathlib                         import Path
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.decomposition           import TruncatedSVD
import statsmodels.api                 as     sm
import torch
from   transformers                    import BlipProcessor, BlipForConditionalGeneration
from   utils                           import load_npz, update_dataset, load_csv, save_report


# Step 1: masking
def mask_images(images: Path, npz: Path, threshold: float, masked: Path) -> None:
    """
    Creates a masked image for every .npz file in npz_dir, using the matching original image in images.
    Masked images are saved as PNG in masked.

    PNG is used because it is lossless AND, being mostly black (only the manipulated
    region is visible), it compresses extremely well: this keeps the masks folder small on disk.
    """

    if not images.is_dir() or not npz.is_dir():
        print("ERROR: invalid input directories for creating masked images.")
        return


    # Maximum PNG compression (0 = none, 9 = max): smaller files at the cost of a bit of CPU
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    # Create the output directory if it doesn't exist
    masked.mkdir(parents=True, exist_ok = True)

    # Masking
    print(f"Creating masked images for all images in '{images}'...")
    for item in npz.glob("*.npz"):
        # TruFor appends .npz to the original filename, so .stem is the original image name (with extension)
        image_name = item.stem

        # Load image
        image = cv2.imread(images / image_name)
        if image is None:
            print(f"ERROR: '{image_name}' image not found or corrupted at path.")
            continue
        
        # Load anomaly map
        anomaly_df = load_npz(item, {"map"})
        if anomaly_df is None:
            continue
        anomaly = np.array(anomaly_df["map"].iloc[0])
        # Resize the map to match the image pixels (width, height)
        # cv2.resize expects (width, height), while numpy shape is (height, width)
        anomaly = cv2.resize(anomaly, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarization: cv2 requires the mask to be uint8 format (0 or 255) for bitwise operations
        binary_mask = (anomaly >= threshold).astype(np.uint8) * 255
        # Apply the mask using OpenCV's optimized native function
        image =  cv2.bitwise_and(image, image, mask=binary_mask)

        # Force deletion of the anomaly map and run garbage collection to free up memory immediately
        del anomaly
        gc.collect()

        if image is None:
            continue
        # Save as compressed PNG
        cv2.imwrite(masked / f"{Path(image_name).stem}.png", image, png_params)

    print(f"Masked images successfully saved to '{masked}'.")
    return None


# Step 2: describing anomaly
def load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    """
    Loads the Hugging Face BLIP processor and model for image captioning.
    """

    print("Loading Hugging Face model (BLIP)...")
    try:
        # Check for hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia GPUs) and set the device accordingly
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 for better accuracy
        torch_dtype = torch.float32
        print(f"Using device: {device} in {torch_dtype} precision.")

        # Processor is responsible for preparing the input data (images) in the format expected by the model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # Model is the actual neural network that generates the caption based on the processed input
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch_dtype)

        # Move the model to the selected device for faster inference
        model.to(device)

        print("BLIP model loaded successfully.")
        return processor, model, device

    except Exception as e:
        print(f"ERROR: loading BLIP model: {e}")
        return None


def _describe_images_batch(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, batch: list[Path], max_length: int) -> list[tuple[str, str]] | None:
    """
    Passes a batch of masked images to the BLIP model to get textual descriptions of the manipulated area.
    Returns a list of tuples containing the image filename stem and its generated description.
    """

    if not batch:
        print("ERROR: empty image batch for description.")
        return None


    try:
        loads, names = [], []
        # Load all images in the batch
        for image_path in batch:
            # Force the image to be read in color mode
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"ERROR: '{image_path}' image not found or corrupted.")
                continue
            # Hugging Face processors accept numpy arrays directly in RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loads.append(image)
            names.append(image_path.stem)
        if not loads:
            return None
        

        # Process the entire batch of images and move the input tensors to the model's device
        inputs = processor(images=loads, return_tensors="pt").to(device)
        # Generate the predictions
        out = model.generate(**inputs, max_new_tokens=max_length)

        # Decode the output tensors into a list of readable text strings
        batch_descriptions = processor.batch_decode(out, skip_special_tokens=True)
        return list(zip(names, batch_descriptions))

    except Exception as e:
        print(f"ERROR: generating description: {e}")
        return None


def describe_images(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, masked: Path, batch_size: int, max_length: int, dataset: Path) -> None:
    """
    Iterates over all masked images in masked_dir in chunks (batches) and generates textual descriptions.
    """

    if not masked.is_dir() or batch_size <= 0 or max_length <= 0:
        print(f"ERROR: '{masked}' is not a valid directory or batch size is not positive.")
        return None


    # Get image paths, excluding hidden files
    images = [path for path in masked.iterdir() if not path.name.startswith(".")]
    if not images:
        print(f"WARNING: No valid masked images found in '{masked}'.")
        return None
    len_images = len(images)

    # Generating descriptions
    print(f"Describing masked images in folder: '{masked}'...")
    descriptions = []
    # Process images in chunks of size = batch_size
    for i in range(0, len_images, batch_size):
        batch = images[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len_images + batch_size - 1) // batch_size}...", end="\r")

        # Generate the descriptions for the batch
        batch_results = _describe_images_batch(processor, model, device, batch, max_length)
        if not batch_results:
            print(f"WARNING: No descriptions generated for batch {i // batch_size + 1}.")
            continue

        descriptions.extend(batch_results)

    update_dataset(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
    return None


# Step 3: regression on the descriptions
def extract_text_features(descriptions: pd.Series, max_words: int, svd_components: int, use_svd: bool) -> pd.DataFrame | None:
    """
    Converts the text descriptions into a numerical feature matrix.

    TF-IDF (Term Frequency - Inverse Document Frequency) always runs: it turns each description string into a numeric vector (a row of word weights).
    SVD (Singular Value Decomposition) is optional (use_svd): when enabled it compresses the wide, sparse TF-IDF matrix into
    a few dense latent-concept columns, which avoids multicollinearity in the regression.
    When disabled the raw TF-IDF columns are used directly.
    """

    if descriptions is None or descriptions.empty or descriptions.nunique() <= 1 or max_words <= 0:
        print("ERROR: invalid input for text feature extraction.")
        return None


    print("Vectorizing text descriptions (TF-IDF)...")
    try:
        # TF-IDF
        # Ignore common english words and limit the vocabulary to the most frequent words
        # Extracts both single words (unigrams) and pairs of words (bigrams)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_words, ngram_range=(1, 2))
        # Build the weight matrix (rows = descriptions, columns = words)
        tfidf_matrix = vectorizer.fit_transform(descriptions)


        # SVD disabled: return the raw TF-IDF matrix as a dense DataFrame
        if not use_svd:
            print("SVD disabled: using raw TF-IDF features.")
            return pd.DataFrame(
                tfidf_matrix.toarray(),
                columns = [f"tfidf_{word}" for word in vectorizer.get_feature_names_out()],
                index   = descriptions.index,
            )


        # SVD enabled
        # The number of SVD components must be smaller than both matrix dimensions
        actual_components = min(svd_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if actual_components <= 0:
            print("ERROR: Not enough data to extract SVD features.")
            return None
        print(f"Applying SVD (reducing to {actual_components} components)...")
        # TruncatedSVD decomposes the TF-IDF matrix and keeps only the top components,
        # reducing dimensionality while preserving as much variance as possible (random_state for reproducibility)
        svd = TruncatedSVD(n_components=actual_components, random_state=42)
        svd_features = svd.fit_transform(tfidf_matrix)

        print("Extracted text features successfully.")
        return pd.DataFrame(
            svd_features,
            columns = [f"text_svd_{i + 1}" for i in range(actual_components)],
            index   = descriptions.index,
        )

    except Exception as e:
        print(f"ERROR: during text feature extraction: {e}")
        return None


def _compute_regression(X: pd.DataFrame, Y: pd.Series, target_metric: str, report: Path) -> None:
    """
    Fits an Ordinary Least Squares (OLS) regression to evaluate how well the combined visual
    (trufor_score) and textual features (X) predict the target metric (Y), and saves the report.
    """


    print(f"Computing regression to predict '{target_metric}'...")
    try:
        # In statsmodels we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
        X = sm.add_constant(X)

        # If the rank of X is less than its number of columns there is perfect multicollinearity:
        # The design matrix is singular and the OLS model cannot be fitted
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            print(f"ERROR: Multicollinearity detected. The design matrix is singular for '{target_metric}'. "
                  f"Try enabling SVD or lowering the number of TF-IDF words.")
            return None

        # Define and fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X).fit()

        # Report
        text = (
            "Anomaly Regression:\n"
            f"Target Metric: {target_metric}\n"
            f"Total valid samples: {len(Y)}\n"
            f"Feature columns: {', '.join(X.columns)}\n\n"
            f"{model.summary().as_text()}"
        )

        # Save
        save_report(text, report)

    except Exception as e:
        print(f"ERROR: during OLS regression calculation or saving: {e}")


def regression(dataset: Path, max_words: int, svd_components: int, use_svd: bool, report: Path) -> None:
    """
    Loads the dataset, builds the feature matrix from the anomaly descriptions and the trufor score,
    and runs an OLS regression predicting the crowdtangle score.
    """

    # Load only the columns needed for the regression (load_csv drops rows with missing values)
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return


    # Extract the numerical text features from the anomaly descriptions
    features = extract_text_features(df["anomaly_txt"], max_words, svd_components, use_svd)
    if features is None:
        return


    # Independent variables (X): the visual anomaly score combined with the textual features
    X = pd.concat([df["trufor_score"].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    # Dependent variable (Y): the metric we want to predict
    Y = df["crowdtangle_score"].reset_index(drop=True)

    # OLS regression
    _compute_regression(X, Y, "crowdtangle_score", report)
    return None


def main() -> None:
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Mask images, describe the anomaly and run a regression on the descriptions.")

    # Flags
    # Mask
    parser.add_argument("--skip-mask",       action="store_true",                                  help="Skip step 1 (masking).")
    parser.add_argument("--images",          type=Path,           default=IMG_DIR,                 help=f"Original images directory. Default: '{IMG_DIR}'")
    parser.add_argument("--npz",             type=Path,           default=NPZ_DIR,                 help=f"Directory containing the TruFor .npz files. Default: '{NPZ_DIR}'")
    parser.add_argument("--threshold",       type=float,          default=MASK_THRESHOLD,          help=f"Binarization threshold (0.0 to 1.0). Default: {MASK_THRESHOLD}")
    parser.add_argument("--masked",          type=Path,           default=MASKED_DIR,              help=f"Output directory for the masked images. Default: '{MASKED_DIR}'")
    
    # Describe
    parser.add_argument("--skip-describe",   action="store_true",                                  help="Skip step 2 (anomaly description).")
    parser.add_argument("--batch-size",      type=int,            default=BATCH_SIZE,              help=f"Number of images captioned simultaneously. Default: {BATCH_SIZE}")
    parser.add_argument("--max-length",      type=int,            default=MAX_DESCRIPTION_TOKENS,  help=f"Maximum number of tokens in the generated descriptions. Default: {MAX_DESCRIPTION_TOKENS}")
    parser.add_argument("--dataset",         type=Path,           default=FILTERED_DATASET_FILE,   help=f"Dataset csv file. Default: '{FILTERED_DATASET_FILE}'")

    # Regression
    parser.add_argument("--skip-regression", action="store_true",                                  help="Skip step 3 (regression).")
    parser.add_argument("--max-words",       type=int,            default=MAX_NUMBER_OF_WORDS,     help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--svd-components",  type=int,            default=SVD_COMPONENTS,          help=f"Number of SVD components. Default: {SVD_COMPONENTS}")
    # Switch: SVD on (default) / off. --no-svd uses the raw TF-IDF features instead.
    parser.add_argument("--no-svd",          dest="use_svd",      action="store_false",            help="Disable SVD: use the raw TF-IDF features in the regression.")
    parser.set_defaults(use_svd=True)
    parser.add_argument("--report",          type=Path,           default=ANOMALY_REGRESSION_FILE, help=f"Output file for the regression report. Default: '{ANOMALY_REGRESSION_FILE}'")

    args = parser.parse_args()


    # Step 1: Masking
    if not args.skip_mask:
        mask_images(args.images, args.npz, args.threshold, args.masked)


    # Step 2: Anomaly description
    if not args.skip_describe:
        blip = load_blip()
        if blip is not None:
            processor, model, device = blip
            describe_images(processor, model, device, args.masked, args.batch_size, args.max_length, args.dataset)


    # Step 3: Regression on the descriptions
    if not args.skip_regression:
        regression(args.dataset, args.max_words, args.svd_components, args.use_svd, args.report)

if __name__ == "__main__":
    main()