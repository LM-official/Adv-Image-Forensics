import argparse
from   config          import FILTERED_DATASET_FILE, CORRELATION_FILE, REGRESSION_FILE
from   pathlib         import Path
from   scipy.stats     import pearsonr, spearmanr
import statsmodels.api as     sm
from   utils           import load_csv, save_report


def compute_correlation(dataset: Path, report: Path) -> None:
    """
    Calculates Pearson and Spearman correlations between TruFor score and Crowdtangle score.

    Args:
        dataset: Path to the dataset CSV (must contain trufor_score and crowdtangle_score).
        report:  Output file path where the correlation report will be written.

    Returns:
        None.
    """
    
    if not dataset.is_file():
        print(f"ERROR: Dataset '{dataset}' not found.")
        return None
    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return None
    # Pearson requires at least 2 unique values in each variable to compute correlation, otherwise it will return NaN.
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points after dropping NaN values or insufficient variability in the data.")
        return None

    
    # Correlation
    print(f"Calculating correlation between TruFor scores and Crowdtangle scores...")
    
    """
    Pearson Correlation: Measures linearity. It asks, 'Do these data points form a perfectly straight line?'
    It calculates using exact numbers, making it very sensitive to extreme outliers.
    """
    # Pearson correlation (measures linear relationship)
    pearson_corr, p_value_p = pearsonr(df["trufor_score"], df["crowdtangle_score"])
    
    """
    Spearman Correlation: Measures trend (monotonicity). It asks, 'As X goes up, does Y consistently go up (or down), regardless of the curve?'
    Instead of exact numbers, it looks at the rank of the values (1st, 2nd, 3rd...), making it highly resistant to outliers and non-linear shapes.
    """
    # Spearman correlation (measures monotonic relationship / rank-based)
    spearman_corr, p_value_s = spearmanr(df["trufor_score"], df["crowdtangle_score"])
    print(f"Correlation calculated successfully.")
    

    # Report
    text = (
        "Correlation analysis between TruFor scores and Crowdtangle scores:\n"
        f"Rows analyzed: {len(df)}\n"
        f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value_p:.3e})\n"
        f"Spearman correlation: {spearman_corr:.3f} (p-value: {p_value_s:.3e})"
    )

    # Save
    save_report(text, report)
    return None


def compute_regression(dataset: Path, report: Path) -> None:
    """
    Calculates Ordinary Least Squares (OLS) regression between TruFor score and Crowdtangle score.

    Args:
        dataset: Path to the dataset CSV (must contain trufor_score and crowdtangle_score).
        report:  Output file path where the regression report will be written.

    Returns:
        None.
    """
    
    if not dataset.is_file():
        print(f"ERROR: Dataset '{dataset}' not found.")
        return None
    # Load dataset
    df = load_csv(dataset, ["trufor_score", "crowdtangle_score"])
    if df is None:
        return None
    if df["trufor_score"].nunique() <= 1 or df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return None


    # Regression
    print("Calculating regression between TruFor Scores and Crowdtangle scores...")
    
    # Independent variable (X): What we use to predict
    X = df["trufor_score"]
    # In statsmodels, we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
    X = sm.add_constant(X)
    # Dependent variable (y): What we want to predict
    Y = df["crowdtangle_score"]

    """
    Ordinary Least Squares (OLS) regression is a statistical method used to model the relationship between a dependent variable
    (Y, e.g., CrowdTangle score) and one independent variable (X, e.g., TruFor Score).
    1. R-squared: the coefficient of determination.
       It measures the proportion of variance in the dependent variable that is predictable from the independent variable.
       Values range from 0 to 1 (0% to 100%). Explains how much of the 'variability' of the engagement score is justified by the TruFor score.
       If it is 0.01, it means that TruFor explains only 1% of the variance in engagement.
    2. Adj. R-squared: a version of R-squared adjusted for the number of predictors in the model.
    3. Coefficients (coef): represent the mean change in the dependent variable for one unit of change in the independent variable while
       holding other predictors constant.
       It tells you the 'slope'.
       If the TruFor coefficient is negative, it means that as the anomaly score increases (more 'fake'), the engagement score decreases.
    4. P>|t| (p-value): tests the null hypothesis that the coefficient is equal to zero (no effect).
       A low p-value (< 0.05) indicates that the predictor is statistically significant.
    5. F-statistic: evaluates the overall significance of the model.
    """
    # Define and fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(Y, X).fit()
    print(f"Regression calculated successfully.")
    

    # Report
    text = (
        "Regression analysis between TruFor scores and Crowdtangle scores:\n"
        f"{model.summary().as_text()}"
    )

    # Save
    save_report(text, report)
    return None


def main() -> None:
    """
    Runs correlation and OLS regression between TruFor scores and Crowdtangle scores.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Compute correlation and regression between TruFor scores and Crowdtangle scores.")

    # Flags
    parser.add_argument("--dataset",     type=Path, default=FILTERED_DATASET_FILE, help=f"Path to the dataset file. Default: '{FILTERED_DATASET_FILE}'")
    parser.add_argument("--correlation", type=Path, default=CORRELATION_FILE,      help=f"Path to text file to save the correlation report. Default: '{CORRELATION_FILE}'")
    parser.add_argument("--regression",  type=Path, default=REGRESSION_FILE,       help=f"Path to text file to save the regression report. Default: '{REGRESSION_FILE}'")
    args = parser.parse_args()


    # Correlation
    compute_correlation(args.dataset, args.correlation)
    
    # Regression
    compute_regression(args.dataset, args.regression)

if __name__ == "__main__":
    main()