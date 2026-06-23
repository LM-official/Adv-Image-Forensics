import argparse
import subprocess
import sys
from   config                          import (SCRIPTS_DIR, FILTERED_DATASET_FILE,
                                               MAX_NUMBER_OF_WORDS, SVD_COMPONENTS, ANOMALY_REGRESSION_FILE)
import pandas                          as     pd
from   pathlib                         import Path
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.decomposition           import PCA, TruncatedSVD
import statsmodels.api                 as     sm
from   utils                           import load_csv, save_report


# Step 3: regression on the descriptions
def extract_text_features(descriptions: pd.Series, max_words: int, svd_components: int, use_svd: bool) -> pd.DataFrame | None:
    """
    Converts the text descriptions into a numerical feature matrix.

    TF-IDF (Term Frequency - Inverse Document Frequency) always runs: it turns each description string into a numeric vector (a row of word weights).
    SVD (Singular Value Decomposition) is optional (`use_svd`):
    when enabled it compresses the wide, sparse TF-IDF matrix into a few dense latent-concept columns, which avoids multicollinearity in the regression.
    When disabled the raw TF-IDF columns are used directly.

    Args:
        descriptions:   Series of textual descriptions, one per sample.
        max_words:      Maximum size of the TF-IDF vocabulary.
        svd_components: Target number of SVD components (only used when use_svd is True).
        use_svd:        If True apply TruncatedSVD on top of TF-IDF, otherwise return raw TF-IDF features.

    Returns:
        A DataFrame of numerical features indexed like descriptions, or None on invalid input or failure.
    """

    if descriptions is None or descriptions.empty or descriptions.nunique() <= 1 or max_words <= 0:
        print("ERROR: invalid input for text feature extraction.")
        return None


    print("Vectorizing text descriptions (TF-IDF)...")
    try:
        # TF-IDF
        # Ignore common english words and limit the vocabulary to the most frequent words
        # Extracts both single words (unigrams) and pairs of words (bigrams)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_words)
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
        actual_components= 60
        # TruncatedSVD decomposes the TF-IDF matrix and keeps only the top components,
        # reducing dimensionality while preserving as much variance as possible (random_state for reproducibility)
        svd = TruncatedSVD(n_components=actual_components, random_state=42)
        #svd = PCA(n_components=0.9, svd_solver="full", random_state=42)
        svd_features = svd.fit_transform(tfidf_matrix.toarray())
        #actual_components = svd.n_components_
        # PCA


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

    Args:
        X:             Design matrix with the independent variables (visual + textual features).
        Y:             Series with the dependent variable (target metric to predict).
        target_metric: Name of the target metric, used only in the report.
        report:        Output file path where the regression summary will be written.

    Returns:
        None.
    """

    print(f"Computing regression to predict '{target_metric}'...")
    try:
        # In statsmodels we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
        X = sm.add_constant(X)

        # If the rank of X is less than its number of columns there is perfect multicollinearity:
        # The design matrix is singular and the OLS model cannot be fitted
        #if np.linalg.matrix_rank(X.values) < X.shape[1]:
        #   print(f"ERROR: Multicollinearity detected. The design matrix is singular for '{target_metric}'. "
        #          f"Try enabling SVD or lowering the number of TF-IDF words.")
        #    return None

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
        return None

    except Exception as e:
        print(f"ERROR: during OLS regression calculation or saving: {e}")


def regression(dataset: Path, max_words: int, svd_components: int, use_svd: bool, report: Path) -> None:
    """
    Loads the dataset, builds the feature matrix from the anomaly descriptions and the trufor score,
    and runs an OLS regression predicting the crowdtangle score.

    Args:
        dataset:        Path to the dataset CSV (must contain trufor_score, anomaly_txt, crowdtangle_score).
        max_words:      Maximum size of the TF-IDF vocabulary.
        svd_components: Target number of SVD components (only used when use_svd is True).
        use_svd:        If True apply SVD on top of TF-IDF, otherwise use the raw TF-IDF features.
        report:         Output file path where the regression summary will be written.

    Returns:
        None.
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
    """
    Runs the three optional steps:
    1. masking (delegated to mask.py via subprocess),
    2. anomaly description (with/without the OCR words of the altered region merged in),
    3. regression on the descriptions.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Mask images, describe the anomaly and run a regression on the descriptions.")

    # Flags
    # Mask (delegated to mask.py via subprocess; mask.py sources its dirs and tuning knobs from config.py)
    parser.add_argument("--skip-mask",       action="store_true",                                  help="Skip step 1 (masking).")

    # Describe (delegated to describe.py via subprocess; describe.py sources its params from config.py)
    parser.add_argument("--skip-describe",   action="store_true",                                  help="Skip step 2 (anomaly description).")
    
    # Regression
    parser.add_argument("--skip-regression", action="store_true",                                  help="Skip step 3 (regression).")
    parser.add_argument("--max-words",       type=int,            default=MAX_NUMBER_OF_WORDS,     help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--svd-components",  type=int,            default=SVD_COMPONENTS,          help=f"Number of SVD components. Default: {SVD_COMPONENTS}")
    # Switch: SVD on (default) / off. --no-svd uses the raw TF-IDF features instead.
    parser.add_argument("--no-svd",          dest="use_svd",      action="store_false",            help="Disable SVD: use the raw TF-IDF features in the regression.")
    parser.set_defaults(use_svd=True)
    parser.add_argument("--report",          type=Path,           default=ANOMALY_REGRESSION_FILE, help=f"Output file for the regression report. Default: '{ANOMALY_REGRESSION_FILE}'")
    args = parser.parse_args()


    # Masking (mask.py reads images/npz/masked and its tuning knobs from config.py)
    if not args.skip_mask:
        command = [sys.executable, str(SCRIPTS_DIR / "mask.py")]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Masking failed: {e}")
            return None

    # Anomaly description (describe.py reads its params from config.py)
    if not args.skip_describe:
        command = [sys.executable, str(SCRIPTS_DIR / "describe.py")]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Description failed: {e}")
            return None

    '''# Step 3: Regression on the descriptions
    if not args.skip_regression:
        regression(args.dataset, args.max_words, args.svd_components, args.use_svd, args.report)'''

if __name__ == "__main__":
    main()