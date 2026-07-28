import argparse
from   classification_glm      import glm, _cv_pseudo_r2
from   classification_home     import prepare_dataset, pca
from   config                  import (FILTERED_DATASET_FILE, VISUAL_TEXT_CLASSIFICATION_FILE, VISUAL_SIGNIFICANCE_LEVELS,
                                       MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE, TFIDF_MIN_DF, TFIDF_MAX_DF,
                                       GLM_ALPHA, GLM_L1_WT, PCA_VARIANCE)
from   features                import FEATURE_NAMES
import numpy                   as     np
import pandas                  as     pd
from   pathlib                 import Path
from   scipy.stats             import chi2
from   sklearn.preprocessing   import StandardScaler
import statsmodels.api         as     sm
from   utils                   import save_report
import warnings


# =============================================================================
# (1) VISUAL FEATURES
# =============================================================================

def standardize(features: pd.DataFrame) -> pd.DataFrame | None:
    """
    Z-scores every visual feature (mean 0, standard deviation 1), keeping the column names and the index.

    The raw features live on wildly different scales: `sharpness` is a Laplacian variance in the thousands while `brightness` is a 0-1 fraction.
    Left as they are, their GLM coefficients are not comparable to each other,
    because each beta is expressed per unit of its own arbitrary scale
    (a "+1" of sharpness is a rounding error, a "+1" of brightness is the whole range).
    After z-scoring every beta reads as "the change in the log-odds per one standard deviation of that feature",
    so the coefficients can be ranked against each other and the largest one really is the strongest driver.

    Fitted on all rows, outside the cross-validation loop: the scaling only uses the features themselves and never the target,
    so it leaks nothing about the label (the same reasoning already applied to pca()).

    Args:
        features: DataFrame of raw visual features, one column per feature.

    Returns:
        A DataFrame of z-scored features with the same columns and index, or None on invalid input or failure.
    """

    if features is None or features.empty:
        print("ERROR: invalid input for the visual feature standardization.")
        return None

    print(f"Standardizing {features.shape[1]} visual features (z-score)...")
    try:
        # A constant column has zero variance: StandardScaler leaves it at 0 instead of dividing by zero,
        # so it simply carries no information rather than breaking the fit
        return pd.DataFrame(
            StandardScaler().fit_transform(features.values),
            columns = features.columns,
            index   = features.index,
        )

    except Exception as e:
        print(f"ERROR: during the visual feature standardization: {e}")
        return None


# =============================================================================
# (2) NESTED GLM COMPARISON
# =============================================================================

def _fit(X: pd.DataFrame, y: pd.Series) -> tuple | None:
    """
    Fits the plain (unregularised) binomial GLM once and returns the numbers the comparison table needs.

    glm() already refits this internally to build its formatted summary, but it only returns text.
    A binomial costs milliseconds, so refitting here is fine.

    Args:
        X: Design matrix without the intercept column, index-aligned with y.
        y: Binary target (1 = high engagement, 0 = low), index-aligned with X.

    Returns:
        A (deviance, df_resid, pseudo_r2) tuple,
        or None when the plain fit does not converge (quasi-complete separation) and its deviance is therefore meaningless.
    """

    # A separating fit overflows before .converged can be read, so silence the expected numerical noise
    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        try:
            fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
            # deviance is computed lazily and overflows on a separating fit
            if not (fit.converged and np.isfinite(fit.deviance)):
                return None
        except Exception:
            return None

    # McFadden pseudo-R2 from the deviance: deviance = -2*LL, so 1 - dev/null_dev = 1 - LL/LL_null
    return float(fit.deviance), int(fit.df_resid), float(1 - fit.deviance / fit.null_deviance)


def _likelihood_ratio(full: tuple | None, reduced: tuple | None) -> str:
    """
    Likelihood-ratio test of a nested pair: does the larger block explain significantly more than the smaller one?

    This is the statistically clean form of "do these extra columns add anything".
    The in-sample pseudo-R2 alone cannot answer it, because it mechanically rises with every column added even when those columns are noise.
    The test compares the drop in deviance against how much a chi-squared distribution says pure chance would give for that many extra parameters.

    Args:
        full:    (deviance, df_resid, pseudo_r2) of the larger block, or None if it did not converge.
        reduced: (deviance, df_resid, pseudo_r2) of the block it is nested in, or None if it did not converge.

    Returns:
        The formatted "<statistic> on <df> df" and p-value columns, or a placeholder when the test is not valid.
    """

    # A regularised or diverged fit has no valid likelihood, so the difference of deviances means nothing
    if full is None or reduced is None:
        return f"{'n/a':>18}  {'n/a':>8}"

    # Deviance = -2*LL, so the difference is the classic 2*(LL_full - LL_reduced) test statistic
    statistic = reduced[0] - full[0]
    # Degrees of freedom = how many parameters the larger block added
    degrees   = reduced[1] - full[1]
    if degrees <= 0:
        return f"{'n/a':>18}  {'n/a':>8}"

    # Survival function = P(chi2 >= statistic): how likely a gain this large is under "the extra columns do nothing"
    return f"{statistic:>9.1f} on {degrees:>2} df  {chi2.sf(statistic, degrees):>8.4f}"


def _significant_features(X: pd.DataFrame, y: pd.Series, levels: tuple = VISUAL_SIGNIFICANCE_LEVELS) -> str:
    """
    Digest of the visual features whose coefficient is significant, strongest effect first.

    The full GLM summary already lists every feature, but it is sorted by column order and padded with standard errors and confidence intervals.
    This pulls out the readable answer: which visual properties actually move engagement, in which direction, and how confidently.

    Args:
        X:      Design matrix of z-scored visual features (no intercept column), index-aligned with y.
        y:      Binary target (1 = high engagement, 0 = low), index-aligned with X.
        levels: (threshold, marker) pairs, strictest first, for the significance stars.

    Returns:
        The digest as text, opened by the significance legend generated from `levels`
        (a "(none)" line when nothing clears the loosest threshold).
    """

    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        try:
            fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
        except Exception as e:
            return f"    ERROR: {e}"

    # Loosest threshold: anything above it is indistinguishable from chance
    cutoff = max(threshold for threshold, _ in levels)
    coefs  = fit.params.drop("const")
    pvals  = fit.pvalues.drop("const")

    # Keep only what clears the cutoff, strongest absolute effect first
    keep  = pvals[pvals < cutoff].index
    order = coefs[keep].abs().sort_values(ascending=False).index

    rows = []
    for name in order:
        # Strictest matching level wins, so a p of .0004 is starred *** rather than *
        marker    = next(mark for threshold, mark in levels if pvals[name] < threshold)
        direction = "up" if coefs[name] > 0 else "down"
        rows.append(f"    {name:<22} {coefs[name]:+.4f} per SD   p={pvals[name]:.4f} {marker:<3}  (pushes engagement {direction})")

    # Legend built from the levels that produced the markers, so editing VISUAL_SIGNIFICANCE_LEVELS can never
    # leave the report explaining a notation it no longer uses. Loosest first, the order a reader scans them in
    legend = ", ".join(f"{mark} p<{f'{threshold:g}'.lstrip('0')}" for threshold, mark in reversed(levels))

    return (
        f"Significance: {legend}\n\n"
        + ("\n".join(rows) if rows else f"    (no feature reaches p < {cutoff:g})")
    )


def classify(dataset: Path = FILTERED_DATASET_FILE, report: Path = VISUAL_TEXT_CLASSIFICATION_FILE,
             max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
             min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF, alpha: float = GLM_ALPHA, l1_wt: float = GLM_L1_WT,
             pca_variance: float = PCA_VARIANCE):
    """
    Fits a nested chain of binomial GLMs and writes the comparison plus every coefficient summary to one report.

    Everything stays inside the same model family (binomial GLM, logit link) on the same posts, so the pseudo-R2 of
    each block is directly comparable and the whole story can be told with one table.
    The four blocks are nested:
      1. TruFor                  - the manipulation score alone (the baseline already in trufor_crowdtangle_glm.txt)
      2. TruFor + visual         - plus the whole-image features: is the image well made / social-friendly? (craft)
      3. TruFor + text           - plus the description of the altered region: what is in it? (content)
      4. TruFor + visual + text  - both, so each contribution can be attributed separately

    Each block reports the in-sample pseudo-R2, the out-of-sample (5-fold cross-validated) pseudo-R2 and a
    likelihood-ratio test against the block it is nested in. The cross-validated column is what keeps the
    comparison honest: the in-sample one always grows when columns are added, even when they carry no signal.

    The text enters as PCA components rather than raw TF-IDF: on the raw matrix the plain GLM may hits quasi-complete separation,
    which leaves no p-values and no valid likelihood-ratio test (that variant is already covered by classification_glm.py).

    Args:
        dataset:      Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score + the visual features).
        report:       Output file for the nested comparison and the coefficient summaries.
        max_words:    Maximum size of the TF-IDF vocabulary.
        dead_zone:    Drop posts with |crowdtangle_score| <= this before binarising. 0 disables the cut.
        min_df:       Keep only words appearing in at least this many documents.
        max_df:       Keep only words appearing in at most this fraction of documents.
        alpha:        Elastic-net strength for the GLM fallback when a plain fit does not converge; 0 disables it.
        l1_wt:        Mix of that penalty: 0 = ridge, 1 = lasso, in-between = elastic net.
        pca_variance: Fraction of variance the text PCA must retain.

    Returns:
        None.
    """

    # Clamp to 0 (kept for a consistent header; binarize_engagement clamps internally too)
    if dead_zone < 0:
        dead_zone = 0

    # Validate the regularization parameters
    if alpha < 0:
        print("ERROR: alpha must be >= 0 (0 disables the regularized fallback).")
        return
    if not 0 <= l1_wt <= 1:
        print("ERROR: l1_wt must be in [0, 1] (0 = ridge, 1 = lasso, in-between = elastic net).")
        return

    # max_words, min_df, max_df and pca_variance have delegated checks

    # Load + dead-zone cut + binarisation + TF-IDF,
    # with the visual features alongside so every block is fitted on the same posts as the sibling classification reports
    prep = prepare_dataset(dataset, dead_zone, max_words, min_df, max_df,
                           columns=("trufor_score", "anomaly_txt", "crowdtangle_score", *FEATURE_NAMES))
    if prep is None:
        return
    df, y, tfidf = prep
    len_y  = len(y)

    # Visual block: z-scored so the coefficients are per standard deviation and comparable to each other
    visual = standardize(df[list(FEATURE_NAMES)])
    if visual is None:
        return

    # Text block: PCA-compressed, the variant on which the plain GLM converges most and yields p-values
    text = pca(tfidf, pca_variance)
    if text is None:
        return


    # The nested chain: each block adds columns to an earlier one, which is what makes the likelihood-ratio tests valid
    # "parent" names the block the likelihood-ratio test compares against
    trufor = df["trufor_score"]
    blocks = (
        ("TruFor",                 trufor,                                             None),
        ("TruFor + text",          pd.concat([trufor, text],         axis=1),          "TruFor"),
        ("TruFor + visual",        pd.concat([trufor, visual],       axis=1),          "TruFor"),
        ("TruFor + text + visual", pd.concat([trufor, text, visual], axis=1),          "TruFor + text"),
    )

    # One pass per block: the formatted summary, the plain-fit numbers for the table, and the out-of-sample score
    fits, rows, sections = {}, [], []
    for title, X, parent in blocks:
        # The TruFor baseline is a single column, so it comes out of the dataset as a Series
        # while every other block is a concatenated DataFrame
        # Promote it, so the whole loop can treat the blocks identically:
        # a Series has no .shape[1] for the column count and would reach statsmodels without a column name
        X = X.to_frame() if isinstance(X, pd.Series) else X

        # Inferential summary (p-values when the plain fit converges, elastic-net fallback otherwise)
        sections.append(glm(title, X, y, alpha, l1_wt))

        # Table numbers: the plain fit for the deviance-based statistics, and the cross-validated fit for the honest score
        fits[title] = _fit(X, y)
        print(f"Cross-validating the binomial GLM ({title})...")
        cv_score = _cv_pseudo_r2(sm.add_constant(X), y, alpha=0, l1_wt=0)

        in_sample = f"{fits[title][2]:>12.4f}" if fits[title] is not None else f"{'n/a':>12}"
        cv        = f"{cv_score:>8.4f}"        if cv_score      is not None else f"{'n/a':>8}"
        ratio     = _likelihood_ratio(fits[title], fits[parent]) if parent else f"{'-':>18}  {'-':>8}"
        rows.append(f"  {title:<24} {X.shape[1]:>4}  {in_sample}  {cv}  {ratio}")

    # The readable payoff: which whole-image properties actually move engagement
    sections.append(
        "=== Significant visual features (binomial GLM, z-scored) ===\n"
        "Coefficients are per standard deviation, so they are comparable to each other.\n"
        + _significant_features(pd.concat([trufor, visual], axis=1), y)
    )


    # save report
    table  = (
        f"  {'block':<24} {'cols':>4}  {'in-sample R2':>12}  {'CV R2':>8}  {'LR vs parent':>18}  {'p':>8}\n"
        f"  {'-' * 83}\n"
        + "\n".join(rows)
    )
    header = (
        f"{'=' * 77}\n"
        "Anomaly Classification - TruFor + text + visual features (nested binomial GLM)\n"
        f"{'=' * 77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {len(df)} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"TF-IDF       : min_df={min_df}, max_df={max_df:g}, vocabulary {tfidf.shape[1]} words (cap {max_words})\n"
        f"Text         : {text.shape[1]} PCA components of the TF-IDF ({pca_variance:.0%} of the variance)\n\n"
        f"Visual       : {visual.shape[1]} whole-image features from features.py, z-scored\n"
        "What each block measures:\n"
        "  trufor_score    - how manipulated the image looks                 (whole image)\n"
        "  anomaly_txt     - CONTENT: what things or people are present      (the altered region only, describe.py)\n"
        "  visual features - CRAFT: is the image well made, social-friendly? (whole image, features.py)\n"
        "features.py runs before mask.py, so the visual features describe the whole image, while the text describes the manipulation itself.\n"
        "The combined model therefore separates whether engagement tracks WHAT was put into the image or HOW WELL the image is made.\n\n"
        "Scope: the visual block is the optional replication of the reference paper's visual-feature work,\n"
        "and is supplementary to the main 'does the alteration drive engagement' research question.\n\n"
        "All blocks are fitted on the same posts with the same model family (binomial GLM, logit link), so their pseudo-R2 are directly comparable.\n"
        "'in-sample R2' always grows when columns are added, even useless ones:\n"
        "'CV R2' (5-fold, out-of-sample) and the likelihood-ratio test are what say whether a block really adds information.\n"
        "'LR vs parent' tests each block against the smaller block it is nested in.\n\n"
        "Nested model comparison:\n"
        f"{table}\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Nested binomial GLM chain over the three sources of information the pipeline produces per post:
    1. the TruFor manipulation score,
    2. the whole-image visual features (craft) and
    3. the description of the altered region (content).
    Reports how much each one adds, with cross-validated scores and likelihood-ratio tests.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Fit a nested binomial GLM chain (TruFor, + visual features, + text) and report how much each block adds.")

    # Flags
    parser.add_argument("--max-words", type=int,   default=MAX_NUMBER_OF_WORDS,  help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
    parser.add_argument("--min-df",    type=int,   default=TFIDF_MIN_DF,         help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df",    type=float, default=TFIDF_MAX_DF,         help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    args = parser.parse_args()


    # Nested GLM comparison of the visual and textual blocks
    classify(max_words=args.max_words, dead_zone=args.dead_zone, min_df=args.min_df, max_df=args.max_df)

if __name__ == "__main__":
    main()