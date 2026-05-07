import  argparse
from    config                              import  TEMP_DIR_PATH, MERGED_METRICS_FILE_NAME, MAX_NUMBER_OF_WORDS, SVD_COMPONENTS, MULTIMODAL_REGRESSION_FILE_NAME
import  numpy                               as      np
from    pathlib                             import  Path
import  pandas                              as      pd
import  statsmodels.api                     as      sm
from    sklearn.feature_extraction.text     import  TfidfVectorizer
from    sklearn.decomposition               import  TruncatedSVD
import  sys
from    utils                               import  load_target_metrics


def extract_text_features(descriptions: pd.Series, features: int, svd_components: int) -> pd.DataFrame | None:
    """
    Extracts numerical features from text descriptions using a dimensionality reduction pipeline.
    Applies TF-IDF (Term Frequency - Inverse Document Frequency) to convert the text into a sparse weight matrix,
    keeping only the top "features" (most frequent words).
    Then uses truncated SVD (Singular Value Decomposition) to compress this matrix into a dense DataFrame of "svd_components" latent semantic features.
    It automatically scales down the requested components to prevent mathematical errors (e.g. on small datasets).

    Args:
        descriptions (pd.Series): A pandas Series containing the textual descriptions.
        features (int): The maximum number of top terms to consider for the TF-IDF vocabulary.
        svd_components (int): The maximum number of SVD components (latent concepts) to extract.

    Returns:
        pd.DataFrame | None: A DataFrame containing the extracted features as numerical columns, 
        or None if an error occurs during the extraction process.
    """

    print("Vectorizing text descriptions (TF-IDF)...")
    
    # LBYL = Look Before You Leap: check if the input descriptions are valid and if the requested number of features and SVD components are positive integers
    if (descriptions is None or descriptions.empty or descriptions.nunique() <= 1 or
        features <= 0 or svd_components <= 0):
        print("ERROR: invalid input for text feature extraction.")
        return


    # EAFP = Try to process the NLP features and catch mathematical/memory exceptions
    try:
        # TF-IDF
        # Ignore useless english words and limit the number features (most used word) to extract
        # Ngram_range=(1, 2) to extract both single words (unigrams) and pairs of words (bigrams)
        vectorizer = TfidfVectorizer(stop_words = "english", max_features = features, ngram_range = (1, 2))
        # Build the weight matrix (rows = descriptions, columns = words) where each value is the TF-IDF weight of the word in the description
        tfidf_matrix = vectorizer.fit_transform(descriptions)


        # SVD
        # LBYL = Look Before You Leap: limit the number of SVD components to less than the dimensions of the TF-IDF matrix
        actual_components = min(svd_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if actual_components <= 0:
            print("ERROR: Not enough data to extract SVD features.")
            return
        print(f"Applying SVD (reducing to {actual_components} components)...")
        # TruncatedSVD is a linear dimensionality reduction technique that decomposes the TF-IDF matrix into three matrices (U, Sigma, V^T)
        # And keeps only the top "actual_components" singular values and corresponding vectors,
        # Reducing the dimensionality of the text features while preserving as much variance as possible
        # Random_state is set for reproducibility
        svd = TruncatedSVD(n_components = actual_components, random_state = 42)
        # Apply SVD compression to the TF-IDF matrix, resulting in a new matrix where each row is a description represented by "actual_components" SVD features
        svd_features = svd.fit_transform(tfidf_matrix)


        # Create a DataFrame with the SVD features, naming the columns as "text_svd_1", "text_svd_2", ..., "text_svd_{actual_components}"
        svd_cols = [f"text_svd_{i+1}" for i in range(actual_components)]

        print(f"Extracted text features successfully.")
        return pd.DataFrame(svd_features, columns = svd_cols, index = descriptions.index)

    except Exception as e:
        print(f"ERROR: during text feature extraction: {e}")
        return


def compute_regression(X: pd.DataFrame, Y: pd.Series, target_metric: str, report_file_path: Path) -> None:
    """
    Computes a multiple linear regression model and saves the statistical report.
    Fits an Ordinary Least Squares (OLS) regression
    to evaluate how well the combined visual and textual features (X) predict the target human metric (Y).
    
    Args:
        X (pd.DataFrame): The independent variables (e.g. TruFor score and text SVD features).
        Y (pd.Series): The dependent variable to predict (the target human evaluation metric).
        target_metric (str): The predicted metric.
        report_file_path (Path): The file path where the generated text report will be saved.

    Returns:
        None
    """
    
    print(f"Computing regression to predict '{target_metric}'...")


    # EAFP = Try to fit the model and save the report
    try:
        # In statsmodels, we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
        X = sm.add_constant(X)

        # If the rank of the matrix X is less than the number of columns, there is perfect multicollinearity (some features are linear combinations of others),
        # Which makes the matrix singular and prevents the OLS model from being fitted.
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            print(f"ERROR: Multicollinearity detected. The design matrix is singular and cannot be inverted for '{target_metric}'.")
            return
        
        # Define and fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X).fit()


        # Report
        report = f"""Multimodal Regression Analysis
Target Metric: {target_metric}
Total valid samples: {len(Y)}\n
{model.summary().as_text()}"""
        
        # Save
        print(report)
        # Create output directory if it doesn"t exist
        report_file_path.parent.mkdir(parents = True, exist_ok = True)
        with open(report_file_path, mode = "w", encoding = "utf-8") as file:
            file.write(report)
        print(f"Regression calculated successfully.")

    except Exception as e:
        print(f"ERROR: during OLS regression calculation or saving: {e}")


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run Multimodal Regression using Visual and Textual features.")
    
    # Flags
    MERGED_METRICS_FILE_PATH  = TEMP_DIR_PATH / MERGED_METRICS_FILE_NAME
    parser.add_argument("--merged_metrics",     type = Path,    default  = MERGED_METRICS_FILE_PATH,    help = f"Merged metrics csv file path. Default: '{MERGED_METRICS_FILE_PATH}'")
    parser.add_argument("--metric",             type = str,     required = True,                        help = "Target metric to predict. Required.")
    parser.add_argument("--svd",                type = int,     default  = SVD_COMPONENTS,              help = f"Number of SVD (Singular Value Decomposition) components to extract from text. Default: {SVD_COMPONENTS}")
    parser.add_argument("--output",             type = Path,    default = TEMP_DIR_PATH,                help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")

    # Read the command line arguments
    args = parser.parse_args()


    # Load merged metrics
    metrics = load_target_metrics(args.merged_metrics, ["filename", "trufor_score", "manipulated_description", args.metric])
    if metrics is None or metrics.empty:
        sys.exit(1)


    # Extract text features
    features = extract_text_features(metrics["manipulated_description"], MAX_NUMBER_OF_WORDS, args.svd)
    if features is None:
        sys.exit(1)


    # Concatenate the visual anomaly score with the NLP dimensions
    # Independent variable (X): What we use to predict
    X = pd.concat([metrics["trufor_score"], features], axis = 1)
    # Dependent variable (y): What we want to predict
    Y = metrics[args.metric]


    # OLS regression
    REPORT_FILE_PATH = args.output / MULTIMODAL_REGRESSION_FILE_NAME.format(args.metric)
    compute_regression(X, Y, args.metric, REPORT_FILE_PATH)