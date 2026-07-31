import argparse
from   classification_home     import tf_idf, pca, prepare_dataset
from   config                  import (FILTERED_DATASET_FILE, GLM_CLASSIFICATION_FILE, GRID_SEARCH_FILE,
                                       MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE, TFIDF_MIN_DF, TFIDF_MAX_DF,
                                       GLM_ALPHA, GLM_L1_WT, GRID_MIN_DF, GRID_MAX_DF, GRID_ALPHA, GRID_L1_WT,
                                       PCA_VARIANCE)
import itertools
import numpy                   as     np
import pandas                  as     pd
from   pathlib                 import Path
from   sklearn.metrics         import log_loss
from   sklearn.model_selection import KFold
import statsmodels.api         as     sm
from   tqdm                    import tqdm
from   utils                   import load_csv, save_report, binarize_engagement
import warnings


# =============================================================================
# (1) BINOMIAL GLM (inferential)
# =============================================================================

def glm(title: str, X: pd.DataFrame, y: pd.Series, alpha: float = GLM_ALPHA, l1_wt: float = GLM_L1_WT, top_n: int = 15) -> str:
    """
    Fits a binomial GLM with a logit link and returns its coefficient summary.

    statsmodels GLM is unregularised, so on the wide, sparse raw TF-IDF matrix it hits quasi-complete separation:
    the coefficients diverge and the standard errors / p-values become meaningless.
    Non-convergence or a non-finite log-likelihood fall back to an elastic-net regularised fit:
    the penalty shrinks the exploding coefficients toward zero (lasso even disables the useless ones entirely),
    so the fit is stable and the coefficients readable - but a regularised fit has no standard errors, hence no p-values.
    On the compact PCA features the plain GLM is well posed and the full summary is valid.

    Args:
        title: Human-readable name of the feature variant (e.g. "Raw TF-IDF").
        X:     Design matrix (visual + textual features), index-aligned with y.
        y:     Binary target (1 = high engagement, 0 = low), index-aligned with X.
        alpha: Elastic-net penalty strength for the fallback; 0 disables the fallback.
        l1_wt: Mix of the penalty: 0 = ridge, 1 = lasso, in-between = elastic net.
        top_n: How many coefficients to list on each side in the regularised fallback.

    Returns:
       The report section as text (an ERROR line on failure).
    """

    print(f"Fitting binomial GLM (logit link) ({title})...")
    try:
        # add_constant adds the intercept column; Binomial() defaults to the logit link
        # On separating raw TF-IDF the fit overflows/diverges before we detect it below,
        # so we silence the expected overflow / divide-by-zero / non-convergence noise around the fit
        X_const = sm.add_constant(X)
        model   = sm.GLM(y, X_const, family=sm.families.Binomial())
        with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            glm       = model.fit()
            # llf / deviance are computed lazily and overflow on a separating fit
            converged = glm.converged and np.isfinite(glm.llf)

        if converged:
            # McFadden pseudo-R2 from the deviance: deviance = -2*LL, so 1 - dev/null_dev = 1 - LL/LL_null
            pseudo_r2 = 1 - glm.deviance / glm.null_deviance

            print(f"Binomial GLM ({title}) fitted successfully.")
            return (
                f"=== {title} - binomial GLM (logit link) ===\n"
                f"In-sample McFadden pseudo-R2: {pseudo_r2:.4f}\n\n"
                f"{glm.summary().as_text()}"
            )

        # Fallback: elastic-net regularised fit
        if alpha > 0:
            with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore")
                reg = model.fit_regularized(method="elastic_net", alpha=alpha, L1_wt=l1_wt)

            if np.isfinite(reg.params).all():
                # fit_regularized returns bare params (no llf/deviance), so rebuild the deviance
                # from the predictions to keep the same McFadden pseudo-R2 as the plain fit
                mu        = model.predict(reg.params)
                pseudo_r2 = 1 - model.family.deviance(y, mu) / model.family.deviance(y, np.full(len(y), y.mean()))

                # Coefficient reading, like odds-ratio ranking of the classifier:
                # positive pushes toward high engagement, negative toward low
                # Lasso zeroes the useless ones
                coefs  = reg.params.drop("const")
                zeroed = int((coefs == 0).sum())
                alive  = coefs[coefs != 0].sort_values()
                up     = ", ".join(f"{name} ({value:+.3f})" for name, value in alive.tail(top_n)[::-1].items())
                down   = ", ".join(f"{name} ({value:+.3f})" for name, value in alive.head(top_n).items())

                print(f"Binomial GLM ({title}) fitted successfully (elastic-net regularised fallback).")
                return (
                    f"=== {title} - binomial GLM (logit link, elastic-net regularised) ===\n"
                    "The plain GLM did not converge (quasi-complete separation), so this is a regularised fit:\n"
                    "coefficients are shrunk toward zero and stable, but there are no standard errors / p-values.\n"
                    f"Penalty            : alpha={alpha:g}, L1_wt={l1_wt:g} (0 = ridge, 1 = lasso, in-between = elastic net)\n"
                    f"In-sample McFadden pseudo-R2: {pseudo_r2:.4f}\n"
                    f"Zeroed coefficients: {zeroed} of {len(coefs)}\n\n"
                    "Largest coefficients (positive -> pushes toward high engagement):\n"
                    f"  up:   {up}\n"
                    f"  down: {down}"
                )

        print(f"ERROR: the binomial GLM ({title}) did not converge, its coefficients and p-values are not interpretable.")
        return (
            f"=== {title} - binomial GLM (logit link) ===\n"
            "Skipped: the unregularised GLM did not converge (quasi-complete separation on the sparse TF-IDF features),\n"
            "so its coefficients and p-values are not interpretable"
            + (" (and the regularised fallback produced non-finite coefficients too).\n" if alpha > 0 else
               " (and the regularised fallback is disabled: alpha = 0).\n")
            + "Use the cross-validated logistic-regression metrics (classification_logistic_regression.py);\n"
            "the GLM inference is reliable on the PCA-compressed features."
        )

    except Exception as e:
        print(f"ERROR: during the binomial GLM ({title}): {e}")
        return f"=== {title} - binomial GLM (logit link) ===\nERROR: {e}"


def classify(dataset: Path = FILTERED_DATASET_FILE, report: Path = GLM_CLASSIFICATION_FILE,
             max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
             min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF, alpha: float = GLM_ALPHA, l1_wt: float = GLM_L1_WT,
             pca_variance: float = PCA_VARIANCE):
    """
    Fits the binomial GLM on the raw TF-IDF and on its PCA-compressed version, and writes both coefficient summaries to one report.
    On the raw TF-IDF the plain GLM does not converge, so a regularised fallback is used (no p-values);
    on the PCA features the plain GLM converges and the full inferential table (p-values) is valid.

    Args:
        dataset:      Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score).
        report:       Output file for the combined GLM summary.
        max_words:    Maximum size of the TF-IDF vocabulary.
        dead_zone:    Drop posts with |crowdtangle_score| <= this before binarising. 0 disables the cut.
        min_df:       Keep only words appearing in at least this many documents.
        max_df:       Keep only words appearing in at most this fraction of documents.
        alpha:        Elastic-net strength for the GLM fallback when the plain fit does not converge; 0 disables it.
        l1_wt:        Mix of that penalty: 0 = ridge, 1 = lasso, in-between = elastic net.
        pca_variance: Fraction of variance the PCA variant must retain.

    Returns:
        None.
    """

    print("Computing binomial GLM of anomaly descriptions...")

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

    # Load + dead-zone cut + binarisation + TF-IDF
    prep = prepare_dataset(dataset, dead_zone, max_words, min_df, max_df)
    if prep is None:
        return
    df, y, tfidf, n = prep

    # Two feature variants: the raw TF-IDF columns and their PCA-compressed version
    variants = [("Raw TF-IDF", tfidf)]
    compress_pca = pca(tfidf, pca_variance)
    if compress_pca is None:
        return
    variants.append(("TF-IDF + PCA", compress_pca))

    # Fit the GLM on each feature variant: the inferential summary with p-values when the plain fit converges,
    # otherwise the regularised fallback
    sections = []
    for title, features in variants:
        X = pd.concat([df["trufor_score"], features], axis=1)
        sections.append(glm(title, X, y, alpha, l1_wt))


    # save report
    len_y  = len(y)
    header = (
        f"{'=' *77}\n"
        "Anomaly Classification - Binomial GLM (inferential)\n"
        f"{'=' *77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {n} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"TF-IDF       : min_df={min_df}, max_df={max_df:g}, vocabulary {tfidf.shape[1]} words (cap {max_words})\n"
        "The GLM table is in-sample inference: on the raw TF-IDF a regularised fallback is used (no p-values),\n"
        "on the PCA-compressed features the plain GLM converges (valid p-values).\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)
    print("Binomial GLM of anomaly descriptions computed successfully.")


# =============================================================================
# (2) GRID SEARCH
# =============================================================================

def _cv_pseudo_r2(X_const: pd.DataFrame, y: pd.Series, alpha: float, l1_wt: float, folds: int = 5) -> float | None:
    """
    Scores one grid-search combination with the out-of-sample McFadden pseudo-R2 of the binomial GLM:
    manual K-fold cross-validation where each fold is fitted (plain if alpha = 0, elastic-net otherwise)
    on the training rows and predicts the held-out rows.

    Out-of-sample scoring is what makes combinations with different vocabulary sizes comparable:
    the in-sample pseudo-R2 always grows with more features, CV does not reward overfitting.

    Args:
        X_const: Design matrix with the intercept column already added, index-aligned with y.
        y:       Binary target (1 = high engagement, 0 = low), index-aligned with X_const.
        alpha:   Elastic-net penalty strength; 0 fits the plain (unregularised) GLM.
        l1_wt:   Mix of the penalty: 0 = ridge, 1 = lasso (ignored when alpha = 0).
        folds:   Number of cross-validation folds.

    Returns:
        The cross-validated McFadden pseudo-R2, or None when any fold fails to converge.
    """
    len_y = len(y)

    # One slot per post: every row ends up predicted exactly once, by the fold that did not train on it
    proba    = np.empty(len_y)
    # Splits the posts into folds groups. random_state fixes the split, so re-running gives the same score
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Warning silencing: a separating fold overflows before we detect it
    with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        # One pass per fold: train on the other folds, predict the rows held out here
        for train_idx, test_idx in splitter.split(X_const):
            model = sm.GLM(y.iloc[train_idx], X_const.iloc[train_idx], family=sm.families.Binomial())
            try:
                # Fitting modes: penalised when alpha > 0, plain otherwise
                if alpha > 0:
                    result = model.fit_regularized(method="elastic_net", alpha=alpha, L1_wt=l1_wt)
                    valid  = bool(np.isfinite(result.params).all())
                else:
                    result = model.fit()
                    valid  = result.converged and bool(np.isfinite(result.params).all())
                if not valid:
                    return None
                # predict with the fold's params on the held-out rows
                proba[test_idx] = model.predict(result.params, X_const.iloc[test_idx])
            except Exception:
                return None

    # A fold can return NaN/inf predictions without raising: the score would be meaningless
    if not np.isfinite(proba).all():
        return None

    # McFadden formula: 1 - logloss_model / logloss_null, where the null model always predicts the base rate
    # Clip so a hard 0/1 prediction stays finite
    proba        = np.clip(proba, 1e-15, 1 - 1e-15)
    base_rate    = float(y.mean())
    logloss_full = log_loss(y, proba, labels=[0, 1])
    logloss_null = log_loss(y, [base_rate] * len_y, labels=[0, 1])
    return 1 - logloss_full / logloss_null


def grid_search(dataset: Path = FILTERED_DATASET_FILE, report: Path = GRID_SEARCH_FILE,
                max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
                min_dfs: tuple = GRID_MIN_DF, max_dfs: tuple = GRID_MAX_DF, alphas: tuple = GRID_ALPHA, l1_wts: tuple = GRID_L1_WT):
    """
    Deterministic grid search over the two strategies against the GLM convergence problem:
    TF-IDF matrix conditioning (min_df x max_df) and elastic-net regularization (alpha x L1_wt), tried in every combination.

    Each combination is scored with the cross-validated McFadden pseudo-R2 of the binomial GLM (see _cv_pseudo_r2).
    The report also flags, per min_df/max_df pair, whether the plain (unregularised) GLM converges on all data:
    those rows keep full inference (p-values), so an alpha = 0 row that converges is preferable when its score is close to the best.

    Args:
        dataset:   Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score).
        report:    Output file for the ranked grid-search table.
        max_words: Maximum size of the TF-IDF vocabulary.
        dead_zone: Drop posts with |crowdtangle_score| <= this before binarising (see classify).
        min_dfs:   Grid of min_df values (absolute document counts).
        max_dfs:   Grid of max_df values (document fractions).
        alphas:    Grid of elastic-net penalty strengths (0 = plain GLM).
        l1_wts:    Grid of penalty mixes (0 = ridge, 1 = lasso); ignored when alpha = 0.

    Returns:
        None.
    """

    print(f"Grid search: min_df/max_df pairs x regularization grid...")
    
    # Clamp to 0
    if dead_zone < 0:
        dead_zone = 0

    # Every grid is checked for emptiness, because an empty one silently shrinks the search instead of failing:
    # with no alphas no row is ever scored, with no l1_wts only the alpha = 0 rows survive.
    # min_dfs/max_dfs: emptiness only. Their values stay delegated to tf_idf
    if not min_dfs or not max_dfs:
        print("ERROR: min_dfs and max_dfs must be non-empty grids.")
        return

    # Validate the TF-IDF regularization parameters
    if not alphas or any(alpha < 0 for alpha in alphas):
        print("ERROR: alphas must be a non-empty grid of values >= 0 (0 fits the plain GLM).")
        return
    if not l1_wts or any(not 0 <= w <= 1 for w in l1_wts):
        print("ERROR: l1_wts must be a non-empty grid of values in [0, 1] (0 = ridge, 1 = lasso, in-between = elastic net).")
        return

    # dataset, report and max_words have delegated checks


    # Load only the columns needed
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return

    # Shared dead-zone cut + binarisation
    total  = len(df)
    result = binarize_engagement(df, dead_zone)
    if result is None:
        return
    df, y  = result
    len_y = len(y)


    # The vectorization only depends on min_df/max_df, so build it once per pair and reuse it for every (alpha, L1_wt) of the inner loop
    pairs = list(itertools.product(min_dfs, max_dfs))
    rows  = []

    for min_df, max_df in tqdm(pairs, desc="Grid search", unit="pair"):
        tfidf = tf_idf(df["anomaly_txt"], max_words, min_df, max_df, verbose=False)

        if tfidf is None:
            # Impossible pair (empty vocabulary or max_df cutting below min_df): one skipped row covers it
            rows.append({"min_df": min_df, "max_df": max_df, "alpha": "-", "l1_wt": "-", "vocab": 0, "plain": "-", "score": None})
            continue

        # Design matrix for this pair: TruFor score + one column per surviving word, plus the intercept
        X_const = sm.add_constant(pd.concat([df["trufor_score"], tfidf], axis=1))
        vocab   = tfidf.shape[1]

        # Does the plain GLM converge on all data at this conditioning? (the p-value availability flag)
        with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            try:
                plain    = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
                plain_ok = "yes" if (plain.converged and np.isfinite(plain.llf)) else "no"
            except Exception:
                plain_ok = "no"

        # Score every regularization setting on this same matrix, one result row each
        for alpha in alphas:
            # Without a penalty l1_wt has no effect: evaluate alpha = 0 once
            for l1_wt in (l1_wts if alpha > 0 else (0.0,)):
                score = _cv_pseudo_r2(X_const, y, alpha, l1_wt)
                rows.append({"min_df": min_df, "max_df": max_df, "alpha": f"{alpha:g}", "l1_wt": f"{l1_wt:g}" if alpha > 0 else "-",
                             "vocab": vocab, "plain": plain_ok, "score": score})


    # Rank: scored combinations by CV pseudo-R2 (descending), failed ones at the bottom
    valid = []
    failed = []
    for row in rows:
        if row["score"] is not None:
            valid.append(row)
        else:
            failed.append(row)
    if not valid:
        print("ERROR: no grid-search combination converged.")
        return
    valid.sort(key=lambda row: row["score"], reverse=True)

    best      = valid[0]
    # Best combination that keeps full inference: no penalty and a converging plain GLM
    inference = next((row for row in valid if row["alpha"] == "0" and row["plain"] == "yes"), None)


    # Formats one result as an aligned table line
    def _row(row: dict) -> str:
        score = f"{row['score']:.4f}" if row["score"] is not None else "did not converge"
        return (f"{row['min_df']:>6}  {row['max_df']:>6.2f}  {row['alpha']:>6}  {row['l1_wt']:>5}  "
                f"{row['vocab']:>5}  {row['plain']:>9}  {score}")

    # Report: what was explored, the two winners, then the ranked table (failures listed last)
    header = f"{'min_df':>6}  {'max_df':>6}  {'alpha':>6}  {'L1_wt':>5}  {'vocab':>5}  {'plain GLM':>9}  CV pseudo-R2"
    text   = (
        f"{'=' * 77}\n"
        "GLM GRID SEARCH (TF-IDF conditioning x elastic-net regularization)\n"
        f"{'=' * 77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {total} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"Grid         : min_df {tuple(min_dfs)} x max_df {tuple(max_dfs)} x alpha {tuple(alphas)} x L1_wt {tuple(l1_wts)}"
        f" -> {len(rows)} combinations\n"
        "Scoring      : 5-fold cross-validated McFadden pseudo-R2 of the binomial GLM (out-of-sample)\n"
        "'plain GLM'  : yes = the unregularised fit converges on all data at that min_df/max_df, so p-values are available\n\n"
        f"Best combination : min_df={best['min_df']} max_df={best['max_df']:g} alpha={best['alpha']} L1_wt={best['l1_wt']}"
        f"  (CV pseudo-R2 {best['score']:.4f}, vocab {best['vocab']})\n"
        + (f"Best with p-values: min_df={inference['min_df']} max_df={inference['max_df']:g} alpha=0"
           f"  (CV pseudo-R2 {inference['score']:.4f}, vocab {inference['vocab']})\n" if inference is not None else
           "Best with p-values: none (the plain GLM never converged on this grid)\n")
        + "Prefer an alpha = 0 row whose score is close to the best: it keeps full inference (p-values).\n\n"
        f"{header}\n"
        f"{'-' * len(header)}\n"
        + "\n".join(_row(row) for row in valid)
        + (("\n\nDid not converge / impossible:\n" + "\n".join(_row(row) for row in failed)) if failed else "")
    )
    save_report(text, report)
    print("Grid search completed successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Fits the binomial GLM (logit link) on the raw and PCA-compressed TF-IDF features, with p-values where the plain fit converges
    and a regularised fallback otherwise.
    
    With --grid-search it instead explores every min_df x max_df x alpha x L1_wt combination to find the settings that make the GLM converge
    with the best out-of-sample fit.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Fit the binomial GLM on the anomaly descriptions (raw TF-IDF and TF-IDF + PCA), or grid-search the TF-IDF/regularization parameters.")

    # Flags
    parser.add_argument("--max-words", type=int,   default=MAX_NUMBER_OF_WORDS,  help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")

    # TF-IDF conditioning and GLM regularization (the two convergence strategies)
    parser.add_argument("--min-df", type=int,   default=TFIDF_MIN_DF, help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df", type=float, default=TFIDF_MAX_DF, help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    parser.add_argument("--alpha",  type=float, default=GLM_ALPHA,    help=f"Elastic-net strength for the GLM fallback when the plain fit does not converge; 0 disables it. Default: {GLM_ALPHA}")
    parser.add_argument("--l1-wt",  type=float, default=GLM_L1_WT,    help=f"Mix of that penalty: 0 = ridge, 1 = lasso. Default: {GLM_L1_WT}")

    # Grid search over the parameters
    parser.add_argument("--grid-search", action="store_true", help="Run the min_df x max_df x alpha x L1_wt grid search instead of the GLM analysis.")
    args = parser.parse_args()


    if args.grid_search:
        # Systematic exploration of the convergence strategies
        grid_search(max_words=args.max_words, dead_zone=args.dead_zone)
    else:
        # Binomial GLM analysis of the descriptions
        classify(max_words=args.max_words, dead_zone=args.dead_zone, min_df=args.min_df, max_df=args.max_df, alpha=args.alpha, l1_wt=args.l1_wt)

if __name__ == "__main__":
    main()