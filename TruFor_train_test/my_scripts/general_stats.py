import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: render straight to file, no display needed
import matplotlib.pyplot as     plt
from   config            import FILTERED_DATASET_FILE, CORRELATION_FILE, REGRESSION_FILE, DISTRIBUTION_FILE, DISTRIBUTION_PLOT_FILE
from   pathlib           import Path
from   scipy.stats       import pearsonr, spearmanr
import statsmodels.api   as     sm
from   utils             import load_csv, save_report


def correlation(dataset: Path = FILTERED_DATASET_FILE, report: Path = CORRELATION_FILE) -> None:
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


def regression(dataset: Path = FILTERED_DATASET_FILE, report: Path = REGRESSION_FILE) -> None:
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


def distribution(dataset: Path = FILTERED_DATASET_FILE, report: Path = DISTRIBUTION_FILE, plot: Path = DISTRIBUTION_PLOT_FILE) -> None:
    """
    Analyzes the distribution of the Crowdtangle score: whether it is centered on zero and how many outliers it has.

    Args:
        dataset: Path to the dataset CSV (must contain crowdtangle_score).
        report:  Output file path where the distribution report will be written.
        plot:    Output image path where the histogram + boxplot figure will be saved.

    Returns:
        None.
    """
    if not dataset.is_file():
        print(f"ERROR: Dataset '{dataset}' not found.")
        return None
    
    # Load dataset
    df = load_csv(dataset, ["crowdtangle_score"])
    if df is None:
        return None
    if df["crowdtangle_score"].nunique() <= 1:
        print("ERROR: Not enough valid data points or insufficient variability in the data.")
        return None


    # Distribution
    print("Analyzing the distribution of Crowdtangle scores...")
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
    print("Distribution analyzed successfully.")


    # Report
    len_score, len_outliers = len(scores), len(outliers)
    text = (
        "Distribution analysis of Crowdtangle scores:\n"
        f"Rows analyzed: {len_score}\n"
        f"{scores.describe().to_string()}\n"
        f"Median: {scores.median():.3f} (mean: {scores.mean():.3f})\n"
        f"Skewness: {scores.skew():.3f}\n"
        f"Tukey fences: [{low_fence:.3f}, {high_fence:.3f}]\n"
        f"Outliers: {len_outliers} ({100 * len_outliers / len_score:.1f}%)"
    )

    # Save
    save_report(text, report)


    # Plot: histogram clipped to the fences (the bulk) + boxplot over the full range (the outliers)
    print("Plotting the distribution...")
    # One figure with two charts (1, 2). ax_hist to the left and ax_box to the right. size 10x4
    _, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histogram: clip to the fences so the outliers don't squash the bulk into a single bar
    scores.clip(low_fence, high_fence).hist(bins=40, ax=ax_hist)
    # Reference red line at zero to judge the centering
    ax_hist.axvline(0, color="r", ls="--")
    ax_hist.set_title("Histogram (clipped to fences)")

    # Boxplot: full range (no clip) so the outliers are visible as separate points
    ax_box.boxplot(scores, vert=False)
    # Reference red line at zero to judge the centering
    ax_box.axvline(0, color="r", ls="--")
    ax_box.set_title("Boxplot (full range)")

    # Prevent titles/labels from overlapping
    plt.tight_layout()
    # Create the output directories if they don't exist
    plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot, dpi=120)
    # Free the figure from memory
    plt.close()
    print(f"Plot saved to '{plot}'.")
    return None


def main() -> None:
    """
    Runs correlation and OLS regression between TruFor scores and Crowdtangle scores.
    Calculates the distribution of Crowdtangle scores and plots it.

    Args:
        None (paths come from config.py, the project backbone).

    Returns:
        None.
    """

    # Inputs and outputs default from config.py
    # Correlation
    correlation()

    # Regression
    regression()

    # Distribution
    distribution()

if __name__ == "__main__":
    main()