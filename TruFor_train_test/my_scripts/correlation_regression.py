import  argparse
from    config              import  TEMP_DIR_PATH, MERGED_METRICS_FILE_NAME, CORRELATION_FILE_NAME, REGRESSION_FILE_NAME
import  pandas              as      pd
from    pathlib             import  Path
from    scipy.stats         import  pearsonr, spearmanr
import  statsmodels.api     as      sm
from    utils               import  load_target_metrics


def compute_correlations(metrics: pd.DataFrame, target_metric: str, correlation_file_path: Path) -> None:
    """
    Calculates and prints the Pearson and Spearman correlations between 
    the TruFor score and the specified target metric.
    """
    
    print(f"Calculating correlation between TruFor Score and '{target_metric}'...")

    if (metrics is None or metrics.empty or
        "trufor_score" not in metrics.columns or target_metric not in metrics.columns):
        print("ERROR: DataFrame is empty or missing required columns.")
        return

    # Drop NaN values 
    metrics = metrics.dropna(subset = ["trufor_score", target_metric])
    # Pearson requires at least 2 unique values in each variable to compute correlation, otherwise it will return NaN.
    if (metrics.empty or
        metrics["trufor_score"].nunique() <= 1 or metrics[target_metric].nunique() <= 1):
        print("ERROR: Not enough valid data points after dropping NaN values or insufficient variability in the data.")
        return
    
    """
    Pearson Correlation: Measures linearity. It asks, 'Do these data points form a perfectly straight line?'
    It calculates using exact numbers, making it very sensitive to extreme outliers.
    """
    # Pearson correlation (measures linear relationship)
    pearson_corr, p_value_p = pearsonr(metrics["trufor_score"], metrics[target_metric])
    
    """
    Spearman Correlation: Measures trend (monotonicity). It asks, 'As X goes up, does Y consistently go up (or down), regardless of the curve?'
    Instead of exact numbers, it looks at the rank of the values (1st, 2nd, 3rd...), making it highly resistant to outliers and non-linear shapes.
    """
    # Spearman correlation (measures monotonic relationship / rank-based)
    spearman_corr, p_value_s = spearmanr(metrics["trufor_score"], metrics[target_metric])
    

    # Report
    report = f"""Correlation analysis between TruFor Score and {target_metric}:
Number of images analyzed: {len(metrics)}
Pearson correlation: {pearson_corr:.3f} (p-value: {p_value_p:.3e})
Spearman correlation: {spearman_corr:.3f} (p-value: {p_value_s:.3e})"""

    # Save
    print(report)
    # Create output directory if it doesn"t exist
    correlation_file_path.parent.mkdir(parents = True, exist_ok = True)
    with open(correlation_file_path, mode = "w", encoding = "utf-8") as file:
        file.write(report)

    print(f"Correlation calculated successfully.")


def compute_regression(metrics: pd.DataFrame, target_metric: str, regression_file_path: Path) -> None:
    """
    Performs Ordinary Least Squares (OLS) regression using statsmodels.
    """

    print(f"Calculating regression between TruFor Score and '{target_metric}'...")

    if (metrics is None or metrics.empty or
        target_metric not in metrics.columns or "trufor_score" not in metrics.columns):
        print("ERROR: DataFrame is empty or missing required columns.")
        return
    
    # Drop NaN values
    metrics = metrics.dropna(subset = ["trufor_score", target_metric])
    if metrics.empty:
        print("ERROR: No valid data points after dropping NaN values.")
        return

    # Independent variable (X): What we use to predict
    X = metrics["trufor_score"]
    # In statsmodels, we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
    X = sm.add_constant(X)
    # Dependent variable (y): What we want to predict
    Y = metrics[target_metric]

    """
    Ordinary Least Squares (OLS) regression is a statistical method used to model the relationship between a dependent variable
    (Y, e.g., Realism/Engagement) and one independent variable (X, e.g., TruFor Score). 
    
    1. R-squared: the coefficient of determination. It measures the proportion of variance in the dependent variable that is predictable from the
       independent variable. Values range from 0 to 1 (0% to 100%). Explain how much of the 'variability' of human votes is justified by the TruFor vote.
       If it is 0.01, it means that TruFor explains only 1% of human behavior.
    2. Adj. R-squared: a version of R-squared adjusted for the number of predictors in the model.
    3. Coefficients (coef): represent the mean change in the dependent variable for one unit of change in the independent variable while
       holding other predictors constant. It tells you the 'slope'. If the TruFor coefficient is negative, it means that as the
       anomaly score increases (more 'fake'), the human grade decreases.
    4. P>|t| (p-value): tests the null hypothesis that the coefficient is equal to zero (no effect). A low p-value (< 0.05) indicates that
       the predictor is statistically significant.
    5. F-statistic: evaluates the overall significance of the model.
    """
    # Define and fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X).fit()
    

    # Report
    report = f"""Regression analysis between TruFor Score and {target_metric}:
{model.summary().as_text()}"""

    # Save
    print(report)
    # Create output directory if it doesn"t exist
    regression_file_path.parent.mkdir(parents = True, exist_ok = True)
    with open(regression_file_path, mode = "w", encoding = "utf-8") as file:
        file.write(report)

    print(f"Regression calculated successfully.")


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Compute performance metrics of TruFor results.")

    # Flags
    MERGED_METRICS_FILE_PATH = TEMP_DIR_PATH / MERGED_METRICS_FILE_NAME
    parser.add_argument("--merged_metrics",     type = Path,     default = MERGED_METRICS_FILE_PATH,        help = f"Merged metrics csv file path. Default: '{MERGED_METRICS_FILE_PATH}'")
    parser.add_argument("--metric",             type = str,      required = True,                           help = "Target metric to run correlation and regression. Required.")
    parser.add_argument("--output",             type = Path,     default = TEMP_DIR_PATH,                   help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")

    # Read the command line arguments
    args = parser.parse_args()


    # Load merged metrics
    metrics = load_target_metrics(args.merged_metrics, ["trufor_score", args.metric])


    # Correlation and regression
    CORRELATION_FILE_PATH   = args.output / CORRELATION_FILE_NAME.format(args.metric)
    REGRESSION_FILE_PATH    = args.output / REGRESSION_FILE_NAME.format(args.metric)
    compute_correlations(metrics, args.metric, CORRELATION_FILE_PATH)
    compute_regression(metrics, args.metric, REGRESSION_FILE_PATH)