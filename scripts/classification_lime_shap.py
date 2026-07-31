import argparse
from   classification_home  import prepare_dataset
from   config               import (FILTERED_DATASET_FILE, EXPLANATION_FILE, MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE,
                                    TFIDF_MIN_DF, TFIDF_MAX_DF, EXPLAIN_TOP_N, EXPLAIN_N_REPEATS, EXPLAIN_N_SAMPLES)
from   lime.lime_tabular    import LimeTabularExplainer
import numpy                as     np
import pandas               as     pd
from   pathlib              import Path
import shap
from   sklearn.linear_model import LogisticRegression
from   sklearn.inspection   import permutation_importance
from   tqdm                 import tqdm
from   utils                import save_report
import warnings


# =============================================================================
# WORD-LEVEL INTERPRETATION
# =============================================================================

def _permutation(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, top_n: int, n_repeats: int) -> str:
    """
    Global word importance by permutation: shuffles each feature and measures the drop in ROC AUC.
    Model-agnostic and needs no convergence, so it works where the raw TF-IDF GLM fails.
    The spread over repeats is the uncertainty (the p-value substitute):
    a word whose mean importance exceeds twice its standard deviation is flagged robust (*).

    Permutation importance is a magnitude, so the direction (a word pushing engagement up or down)
    is read from the sign of the logistic coefficient.

    Args:
        model:     Fitted logistic model exposing predict_proba and coef_.
        X:         Design matrix (trufor_score + tfidf words), index-aligned with y.
        y:         Binary engagement target.
        top_n:     Words listed per side (up/down).
        n_repeats: Shuffles per feature.

    Returns:
        The report section as text (an ERROR line on failure).
    """

    print("Computing permutation importance...")
    try:
        result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=42)
        imp    = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std}, index=X.columns)
        # Direction from the coefficient sign (permutation importance itself is unsigned)
        signs  = pd.Series(model.coef_[0], index=X.columns)

        def _side(mask: pd.Series) -> str:
            """
            Renders one side of the ranking: the top_n words picked by `mask` (up / down), most important first (or "(none)")
            """

            side = imp[mask].sort_values("mean", ascending=False).head(top_n)
            rows = [f"    {name:<28} {row['mean']:.4f} +/- {row['std']:.4f}{'  *' if row['mean'] > 2 * row['std'] else ''}"
                    for name, row in side.iterrows()]
            return "\n".join(rows) if rows else "    (none)"

        return (
            "=== Permutation importance (global, model-agnostic) ===\n"
            f"Importance = drop in ROC AUC when the feature is shuffled (mean +/- std over {n_repeats} repeats);\n"
            "'*' = mean > 2*std (robust, the p-value substitute). Direction from the logistic coefficient sign.\n\n"
            "Words pushing engagement up:\n"
            f"{_side(signs >= 0)}\n\n"
            "Words pushing engagement down:\n"
            f"{_side(signs < 0)}"
        )

    except Exception as e:
        return f"=== Permutation importance (global, model-agnostic) ===\nERROR: {e}"


def _shap(model: LogisticRegression, X: pd.DataFrame, top_n: int) -> str:
    """
    Global word importance by SHAP on the linear logistic model (exact via LinearExplainer):
    importance = mean(|SHAP|) per feature, direction = sign of the coefficient.
    For a linear model this closely tracks the coefficient ranking; it is included for the standard, citable framing.

    Args:
        model: Fitted logistic model.
        X:     Design matrix (trufor_score + tfidf words).
        top_n: Words listed per side (up/down).

    Returns:
        The report section as text (an ERROR line on failure).
    """

    print("Computing SHAP values...")

    try:
        # Full background (all rows) instead of shap's default 100-sample subsample: exact feature means
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(X, max_samples=len(X)))
        sv        = np.asarray(explainer.shap_values(X))
        # A binary classifier may return (n, features) or (n, features, classes); take the positive class
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        # For a linear model the per-instance SHAP sign is coef*(x - E[x]), whose mean is ~0 (background noise)
        # so the direction is read from the coefficient sign
        # mean(|SHAP|) gives the magnitude (the coefficient weighted by how much the word actually varies)
        signs = pd.Series(model.coef_[0], index=X.columns)
        imp   = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

        def _side(mask: pd.Series) -> str:
            """
            Renders one side of the ranking: the top_n words picked by `mask` (up / down), most important first (or "(none)")
            """

            side = imp[mask].sort_values(ascending=False).head(top_n)
            rows = [f"    {name:<28} {value:.4f}" for name, value in side.items()]
            return "\n".join(rows) if rows else "    (none)"

        return (
            "=== SHAP (global; LinearExplainer) ===\n"
            "Importance = mean(|SHAP|) per feature (coefficient weighted by the word's spread);\n"
            "direction from the logistic coefficient sign.\n\n"
            "Words pushing engagement UP:\n"
            f"{_side(signs >= 0)}\n\n"
            "Words pushing engagement DOWN:\n"
            f"{_side(signs < 0)}"
        )

    except Exception as e:
        return f"=== SHAP (global) ===\nERROR: {e}"


def _lime(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, top_n: int, n_samples: int) -> str:
    """
    Local explanations: for a spread of individual posts (from lowest to highest predicted engagement),
    LIME perturbs the features and reports which words locally pushed that post toward high engagement (+) or low (-).
    Each post is shown with its filename and description for context.

    Args:
        model:     Fitted logistic model exposing predict_proba.
        X:         Design matrix (trufor_score + tfidf words).
        y:         Binary engagement target (for the actual label).
        df:        Frame aligned with X, carrying `filename` and `anomaly_txt` for context.
        top_n:     Words per local explanation.
        n_samples: Posts explained.

    Returns:
        The report section as text (an ERROR line on failure).
    """

    print("Computing LIME local explanations...")

    try:
        # discretize_continuous=False keeps the near-binary TF-IDF values continuous instead of quartile-binning them
        explainer = LimeTabularExplainer(X.values, feature_names=list(X.columns), class_names=["low", "high"],
                                         mode="classification", discretize_continuous=False, random_state=42)
        proba = model.predict_proba(X.values)[:, 1]
        # Evenly spaced posts from the lowest to the highest predicted engagement
        order = np.argsort(proba)
        picks = order[np.linspace(0, len(order) - 1, n_samples).astype(int)]

        blocks = []
        for i in tqdm(picks, desc="LIME", unit="post"):
            exp  = explainer.explain_instance(X.values[i], model.predict_proba, num_features=top_n, labels=(1,))
            drivers = ", ".join(f"{name} ({w:+.3f})" for name, w in exp.as_list(label=1))
            blocks.append(
                f"  post '{df.iloc[i]['filename']}'  (predicted P(high)={proba[i]:.2f}, actual={'high' if y.iloc[i] == 1 else 'low'})\n"
                f"    desc: {str(df.iloc[i]["anomaly_txt"])}\n"
                f"    local drivers: {drivers}"
            )

        return (
            "=== LIME (local, per-post) ===\n"
            f"{len(picks)} posts spanning lowest -> highest predicted engagement;\n"
            "a driver's sign is its local push toward high (+) or low (-) engagement for that post.\n\n"
            + "\n\n".join(blocks)
        )

    except Exception as e:
        return f"=== LIME (local, per-post) ===\nERROR: {e}"


def explain(dataset: Path = FILTERED_DATASET_FILE, report: Path = EXPLANATION_FILE,
            max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
            min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF,
            top_n: int = EXPLAIN_TOP_N, n_repeats: int = EXPLAIN_N_REPEATS, n_samples: int = EXPLAIN_N_SAMPLES):
    """
    Model-agnostic word-level interpretation of the logistic regression on the raw TF-IDF.

    The raw TF-IDF GLM does not converge (quasi-complete separation), so there are no word-level p-values.
    Instead of the model weights, this explains the well-performing predictive model directly, in terms of the
    actual words, with three complementary methods: permutation importance (global + uncertainty),
    SHAP (global) and LIME (local, per-post). Only the raw TF-IDF variant is used - PCA is excluded
    on purpose, since its components are not words (and the PCA GLM already gives valid p-values).

    Args:
        dataset:   Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score, filename).
        report:    Output file for the combined interpretation report.
        max_words: Maximum size of the TF-IDF vocabulary.
        dead_zone: Drop posts with |crowdtangle_score| <= this before binarising. 0 disables the cut.
        min_df:    Keep only words appearing in at least this many documents.
        max_df:    Keep only words appearing in at most this fraction of documents.
        top_n:     Words listed per side / per local explanation.
        n_repeats: Permutation-importance shuffles per feature.
        n_samples: Posts explained locally by LIME.

    Returns:
        None.
    """

    # Clamp to 0 (kept for a consistent header; binarize_engagement clamps internally too)
    if dead_zone < 0:
        dead_zone = 0

    if top_n < 1 or n_repeats < 1 or n_samples < 1:
        print("ERROR: invalid arguments.")
        return

    # dataset, report, max_words, min_df and max_df have delegated checks

    # Load + dead-zone cut + binarisation + TF-IDF (filename kept for the LIME per-post context)
    prep = prepare_dataset(dataset, dead_zone, max_words, min_df, max_df, columns=("trufor_score", "anomaly_txt", "crowdtangle_score", "filename"))
    if prep is None:
        return
    df, y, tfidf, n = prep
    len_y  = len(y)

    # Raw TF-IDF (the interpretable words); no PCA on purpose
    X = pd.concat([df["trufor_score"], tfidf], axis=1)

    # Fit the interpretable predictive model once on all data
    # (its out-of-sample skill is already established by the 5-fold CV in classification_logistic_regression.logistic_regression;
    # this fit is only for interpretation)
    print("Fitting logistic regression for post-hoc interpretation...")
    model = LogisticRegression(max_iter=1000).fit(X, y)

    # The helpers predict on numpy arrays (LIME) and SHAP subsamples its background
    # sklearn/shap flag these with benign warnings (the column order is preserved), so silence the noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        warnings.filterwarnings("ignore", message="Background dataset has")
        sections = [
            _permutation(model, X, y, top_n, n_repeats),
            _shap(model, X, top_n),
            _lime(model, X, y, df, top_n, n_samples),
        ]

    # save report
    header = (
        f"{'=' * 77}\n"
        "Post-hoc word-level interpretation\n"
        f"{'=' * 77}\n"
        "Explains the logistic regression on the Raw TF-IDF features (the words), bypassing the GLM convergence problem.\n"
        "Permutation importance & SHAP are global; LIME is local (per-post).\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {n} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"TF-IDF       : min_df={min_df}, max_df={max_df:g}, vocabulary {tfidf.shape[1]} words (cap {max_words})\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Word-level interpretation of the raw TF-IDF logistic model: permutation importance + SHAP (global) and LIME (local, per-post),
    naming which words drive engagement without needing the GLM to converge.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Post-hoc word-level interpretation (permutation importance + SHAP + LIME) of the raw TF-IDF logistic model.")

    # Flags
    parser.add_argument("--max-words", type=int,   default=MAX_NUMBER_OF_WORDS,  help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
    parser.add_argument("--min-df",    type=int,   default=TFIDF_MIN_DF,         help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df",    type=float, default=TFIDF_MAX_DF,         help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    args = parser.parse_args()


    explain(max_words=args.max_words, dead_zone=args.dead_zone, min_df=args.min_df, max_df=args.max_df)

if __name__ == "__main__":
    main()