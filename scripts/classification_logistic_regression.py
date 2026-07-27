import argparse
from   classification_home     import prepare_dataset, pca
from   config                  import (FILTERED_DATASET_FILE, LOGISTIC_CLASSIFICATION_FILE, MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE,
                                       TFIDF_MIN_DF, TFIDF_MAX_DF, PCA_VARIANCE)
import numpy                   as     np
import pandas                  as     pd
from   pathlib                 import Path
from   sklearn.linear_model    import LogisticRegression
from   sklearn.metrics         import roc_auc_score, confusion_matrix, log_loss
from   sklearn.model_selection import cross_val_predict
from   utils                   import save_report


# =============================================================================
# LOGISTIC CLASSIFICATION
# =============================================================================

def logistic_regression(title: str, X: pd.DataFrame, y: pd.Series, top_n: int = 15) -> str :
    """
    Fits a logistic regression classifier predicting the binary engagement label from the visual (trufor_score) and textual features (X),
    and returns its diagnostics as a labelled text section.

    Uses scikit-learn's LogisticRegression (L2-regularised, so it converges even on the wide, sparse raw TF-IDF matrix
    where an unregularised fit would hit perfect separation).
    Performance is measured with 5-fold cross-validation, so the numbers are out-of-sample and not inflated by overfitting 100+ features.

    Args:
        top_n: How many terms to list on each side of the odds-ratio ranking.
        X:     Design matrix (visual + textual features), index-aligned with y.
        y:     Binary target (1 = high engagement, 0 = low), index-aligned with X.
        title: Human-readable name of the feature variant (e.g. "Raw TF-IDF").

    Returns:
        The report section as text (an ERROR line on failure).
    """

    print(f"Computing logistic regression of {title}...")
    try:
        model = LogisticRegression(max_iter=1000)

        # Honest, out-of-sample performance: 5-fold cross-validated predicted probabilities
        # (In-sample metrics on 100+ features would overfit and look far better than the model really is)
        proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
        pred  = (proba >= 0.5).astype(int)

        # Accuracy must be read against the majority-class baseline ("always predict the bigger class"),
        # not against 50%: on an unbalanced split a useless model can still look accurate
        accuracy = float((pred == y).mean())
        baseline = float(max(y.mean(), 1 - y.mean()))
        auc      = roc_auc_score(y, proba)

        # McFadden pseudo-R2 = 1 - LL_model / LL_null. With the average log-loss L (LL = -n*L) the n cancels,
        # so it is 1 - logloss_model / logloss_null, where the null model always predicts the base rate
        base_rate    = float(y.mean())
        logloss_full = log_loss(y, proba, labels=[0, 1])
        logloss_null = log_loss(y, [base_rate] * len(y), labels=[0, 1])
        pseudo_r2    = 1 - logloss_full / logloss_null

        # Confusion matrix at the 0.5 cut-off
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        # Odds ratios = exp(coefficient), from a model fit on all the data (for interpretation, not scoring):
        # > 1 pushes a post toward the high-engagement class, < 1 toward low
        odds = pd.Series(np.exp(model.fit(X, y).coef_[0]), index=X.columns).sort_values()
        up   = ", ".join(f"{name} ({ratio:.2f})" for name, ratio in odds.tail(top_n)[::-1].items())
        down = ", ".join(f"{name} ({ratio:.2f})" for name, ratio in odds.head(top_n).items())

        return (
            f"=== {title} - logistic regression (cross-validated) ===\n"
            f"Features           : {X.shape[1]}\n"
            f"McFadden pseudo-R2 : {pseudo_r2:.4f}\n"
            f"Accuracy           : {accuracy:.1%}  (majority-class baseline: {baseline:.1%})\n"
            f"ROC AUC            : {auc:.3f}  (0.5 = no better than chance)\n"
            f"Confusion matrix   : TP={tp}  TN={tn}  FP={fp}  FN={fn}\n\n"
            "Largest odds ratios (>1 -> more likely high engagement):\n"
            f"  up:   {up}\n"
            f"  down: {down}"
        )

    except Exception as e:
        return f"=== {title} - logistic regression (cross-validated) ===\nERROR: {e}"


def classify(dataset: Path = FILTERED_DATASET_FILE, report: Path = LOGISTIC_CLASSIFICATION_FILE,
             dead_zone: float = ENGAGEMENT_DEAD_ZONE, max_words: int = MAX_NUMBER_OF_WORDS,
             min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF, pca_variance: float = PCA_VARIANCE):
    """
    Turns crowdtangle_score into a binary engagement label and fits a cross-validated logistic classifier twice:
    (1) on the raw TF-IDF features,
    (2) on their PCA compressed version.
    Writes both to one report.

    The continuous crowdtangle_score is bimodal and skewed, so the model is a binary high/low label instead of regressing.
    Posts in the low-density valley around 0 (|score| <= dead_zone) are dropped first,
    so the 0 cut-off separates two clearly-distinct clusters rather than slicing through ambiguous middle posts.
    The classifier then asks "does the manipulation description make a post more likely to overperform?".

    Args:
        dataset:      Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score).
        report:       Output file for the logistic classification summary.
        dead_zone:    Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising.
                      Negatives are clamped to 0. 0 disables the cut and keeps every post.
        max_words:    Maximum size of the TF-IDF vocabulary.
        min_df:       Keep only words appearing in at least this many documents.
        max_df:       Keep only words appearing in at most this fraction of documents.
        pca_variance: Fraction of variance the PCA variant must retain.

    Returns:
        None.
    """

    # Clamp to 0 (kept for a consistent header; binarize_engagement clamps internally too)
    if dead_zone < 0:
        dead_zone = 0

    # max_words, min_df, max_df and pca_variance have delegated checks

    # Load + dead-zone cut + binarisation + TF-IDF
    prep = prepare_dataset(dataset, dead_zone, max_words, min_df, max_df)
    if prep is None:
        return
    df, y, tfidf = prep

    # Two feature variants: the raw TF-IDF columns and their PCA-compressed version
    variants = [("Raw TF-IDF", tfidf)]
    compress_pca = pca(tfidf, pca_variance)
    if compress_pca is None:
        return
    variants.append(("TF-IDF + PCA", compress_pca))


    # Fit the classifier on each feature variant: out-of-sample predictive metrics + odds ratios
    sections = []
    for title, features in variants:
        X = pd.concat([df["trufor_score"], features], axis=1)
        sections.append(logistic_regression(title, X, y))


    # save report
    len_y = len(y)
    header = (
        f"{'=' *77}\n"
        "Anomaly Classification - Logistic Regression\n"
        f"{'=' *77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {len(df)} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"TF-IDF       : min_df={min_df}, max_df={max_df:g}, vocabulary {tfidf.shape[1]} words (cap {max_words})\n"
        "Logistic-regression metrics are 5-fold cross-validated (out-of-sample).\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Logistic classification of the anomaly descriptions by engagement.
    Binarises crowdtangle_score (after dropping the distribution's valley, unless --dead-zone 0 disables it)
    and fits a cross-validated logistic regression on the raw and PCA-compressed TF-IDF features.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Classify the anomaly descriptions by engagement with a cross-validated logistic regression (raw TF-IDF and TF-IDF + PCA).")

    # Flags
    parser.add_argument("--max-words", type=int,   default=MAX_NUMBER_OF_WORDS, help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
    parser.add_argument("--min-df",    type=int,   default=TFIDF_MIN_DF, help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df",    type=float, default=TFIDF_MAX_DF, help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    args = parser.parse_args()


    # Logistic classification of the descriptions
    classify(max_words=args.max_words, dead_zone=args.dead_zone, min_df=args.min_df, max_df=args.max_df)

if __name__ == "__main__":
    main()