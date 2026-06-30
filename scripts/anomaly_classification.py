import argparse
import warnings
from   config                          import (FILTERED_DATASET_FILE, MAX_NUMBER_OF_WORDS,
                                               PCA_VARIANCE, ENGAGEMENT_DEAD_ZONE, ANOMALY_CLASSIFICATION_FILE)
import numpy                           as     np
import pandas                          as     pd
from   pathlib                         import Path
from   sklearn.decomposition           import PCA
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.linear_model            import LogisticRegression
from   sklearn.metrics                 import roc_auc_score, confusion_matrix, log_loss
from   sklearn.model_selection         import cross_val_predict
import statsmodels.api                 as     sm
from   stop_words                      import get_stop_words
from   utils                           import load_csv, save_report, clean_text


# =============================================================================
# (1) TEXT FEATURES
# =============================================================================

# TF-IDF stop words: the descriptions mix English BLIP captions with English-Italian OCR text,
# so we union both languages from the `stop-words` package.
# Each word is run through clean_text (the same applied to the descriptions) so match the tokens
STOP_WORDS = sorted({
    word
    for lang in ("english", "italian")
    for word in clean_text(" ".join(get_stop_words(lang)), dedup_words=True).split()
    if len(word) >= 2
})


def extract_tfidf(descriptions: pd.Series, max_words: int) -> pd.DataFrame | None:
    """
    Vectorises the text descriptions into a TF-IDF feature matrix (one column per word).

    TF-IDF (Term Frequency - Inverse Document Frequency) turns each description string into a numeric vector (a row of word weights).

    Args:
        descriptions: Series of textual descriptions, one per sample.
        max_words:    Maximum size of the TF-IDF vocabulary.

    Returns:
        A dense DataFrame of TF-IDF weights (columns prefixed "tfidf_") indexed like descriptions,
        or None on invalid input or failure.
    """

    if descriptions is None or descriptions.empty or descriptions.nunique() <= 1:
        print("ERROR: invalid input for text feature extraction.")
        return None
    # max_words has delegated checks (try below)


    print("Vectorizing text descriptions (TF-IDF)...")
    try:
        # Ignore common English + Italian stop words, limit the vocabulary to the most frequent words,
        # and extract single words (unigrams). Rows = descriptions, columns = words.
        vectorizer   = TfidfVectorizer(stop_words=STOP_WORDS, max_features=max_words)
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns = [f"tfidf_{word}" for word in vectorizer.get_feature_names_out()],
            index   = descriptions.index,
        )

    except Exception as e:
        print(f"ERROR: during TF-IDF vectorization: {e}")
        return None


def compress_pca(tfidf: pd.DataFrame, pca_variance: float) -> pd.DataFrame | None:
    """
    Compresses an existing TF-IDF matrix into a few dense latent-concept columns with PCA (Principal Component Analysis),
    which avoids the multicollinearity of the wide, sparse TF-IDF in the classifier.
    Reuses the matrix from extract_tfidf instead of re-vectorising.

    Keeps the fewest components that together explain `pca_variance` of the variance
    (the knee / diminishing-returns point) instead of a guessed count.

    Args:
        tfidf:        TF-IDF feature matrix from extract_tfidf.
        pca_variance: Fraction of variance PCA must retain, which sets the component count.

    Returns:
        A DataFrame of PCA components (columns "text_pca_<i>") indexed like tfidf, or None on failure.
    """

    print(f"Applying PCA (retaining {pca_variance:.0%} of the variance)...")
    try:
        # svd_solver="full" is required for a fractional n_components; PCA centers the data,
        # which the dense TF-IDF DataFrame already is
        pca          = PCA(n_components=pca_variance, svd_solver="full")
        pca_features = pca.fit_transform(tfidf.values)
        print(f"PCA kept {pca.n_components_} components ({pca.explained_variance_ratio_.sum():.1%} of the variance).")

        return pd.DataFrame(
            pca_features,
            columns = [f"text_pca_{i + 1}" for i in range(pca.n_components_)],
            index   = tfidf.index,
        )

    except Exception as e:
        print(f"ERROR: during PCA compression: {e}")
        return None


# =============================================================================
# (2) CLASSIFICATION
# =============================================================================

def _classification_report(X: pd.DataFrame, y: pd.Series, title: str, top_n: int = 15) -> str | None:
    """
    Fits a logistic regression classifier predicting the binary engagement label from
    the visual (trufor_score) and textual features (X), and returns its diagnostics as a labelled text section.

    Uses scikit-learn's LogisticRegression (L2-regularised, so it converges even on the wide, sparse raw TF-IDF matrix
    where an unregularised fit would hit perfect separation).
    Performance is measured with 5-fold cross-validation, so the numbers are out-of-sample and not inflated by overfitting 100+ features.

    Args:
        X:     Design matrix (visual + textual features), index-aligned with y.
        y:     Binary target (1 = high engagement, 0 = low), index-aligned with X.
        title: Human-readable name of the feature variant (e.g. "Raw TF-IDF").
        top_n: How many terms to list on each side of the odds-ratio ranking.

    Returns:
        The report section as text, or None if the model could not be fitted.
    """

    print(f"Computing logistic classification ({title})...")
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
        # so it is 1 - logloss_model / logloss_null, where the null model always predicts the base rate.
        base_rate    = float(y.mean())
        logloss_full = log_loss(y, proba, labels=[0, 1])
        logloss_null = log_loss(y, [base_rate] * len(y), labels=[0, 1])
        pseudo_r2    = 1 - logloss_full / logloss_null

        # Confusion matrix at the 0.5 cut-off
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        # Odds ratios = exp(coefficient), from a model fit on all the data (for interpretation, not scoring):
        # > 1 pushes a post toward the high-engagement class, < 1 toward low.
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
        print(f"ERROR: during logistic classification ({title}): {e}")
        return None


def _glm_report(X: pd.DataFrame, y: pd.Series, title: str) -> str | None:
    """
    Fits a binomial GLM with a logit link (the inferential twin of the logistic classifier) and
    returns its coefficient summary.

    statsmodels GLM is unregularised, so on the wide, sparse raw TF-IDF matrix it hits quasi-complete separation:
    the coefficients diverge and the standard errors / p-values become meaningless.
    We detect that (non-convergence or a non-finite log-likelihood) and skip the table, pointing back to the
    regularised logistic metrics.
    On the compact PCA features the GLM is well posed and the summary is valid.

    Args:
        X:     Design matrix (visual + textual features), index-aligned with y.
        y:     Binary target (1 = high engagement, 0 = low), index-aligned with X.
        title: Human-readable name of the feature variant (e.g. "Raw TF-IDF").

    Returns:
        The report section as text, or None if the model could not be fitted at all.
    """

    print(f"Fitting binomial GLM (logit link) ({title})...")
    try:
        # add_constant adds the intercept column; Binomial() defaults to the logit link
        # On separating raw TF-IDF the fit overflows/diverges before we detect it below, so
        # we silence the expected overflow / divide-by-zero / non-convergence noise around the fit
        with warnings.catch_warnings(), np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            glm       = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
            # llf / deviance are computed lazily and overflow on a separating fit
            converged = glm.converged and np.isfinite(glm.llf)

        if not converged:
            return (
                f"=== {title} - binomial GLM (logit link) ===\n"
                "Skipped: the unregularised GLM did not converge (quasi-complete separation on the sparse TF-IDF features),\n"
                "so its coefficients and p-values are not interpretable.\n"
                "Use the cross-validated logistic-regression metrics;\n"
                "the GLM inference is reliable on the PCA-compressed features."
            )

        # McFadden pseudo-R2 from the deviance: deviance = -2*LL, so 1 - dev/null_dev = 1 - LL/LL_null.
        pseudo_r2 = 1 - glm.deviance / glm.null_deviance

        return (
            f"=== {title} - binomial GLM (logit link) ===\n"
            f"In-sample McFadden pseudo-R2: {pseudo_r2:.4f}\n\n"
            f"{glm.summary().as_text()}"
        )

    except Exception as e:
        print(f"ERROR: during binomial GLM ({title}): {e}")
        return None


def classify(dataset: Path = FILTERED_DATASET_FILE, report: Path = ANOMALY_CLASSIFICATION_FILE,
             max_words: int = MAX_NUMBER_OF_WORDS, pca_variance: float = PCA_VARIANCE,
             dead_zone: float = ENGAGEMENT_DEAD_ZONE):
    """
    Turns crowdtangle_score into a binary engagement label and fits a
    logistic classifier and inferential binomial-GLM twice:
    (1) on the raw TF-IDF features,
    (2) on their PCA compressed version.
    Write both to one report.

    The continuous crowdtangle_score is bimodal and skewed, so the model is a binary high/low label instead of regressing.
    Posts in the low-density valley around 0 (|score| <= dead_zone) are dropped first,
    so the 0 cut-off separates two clearly-distinct clusters rather than slicing through ambiguous middle posts.
    The classifier then asks "does the manipulation description make a post more likely to overperform?".

    Args:
        dataset:      Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score).
        report:       Output file for the combined classification summary.
        max_words:    Maximum size of the TF-IDF vocabulary.
        pca_variance: Fraction of variance the PCA variant must retain.
        dead_zone:    Drop posts with |crowdtangle_score| <= this (the distribution's valley) before
                      binarising. 0 disables the cut and keeps every post; negatives are clamped to 0.

    Returns:
        None.
    """

    # Negatives are meaningless: clamp them to 0. 0 disables the cut and keeps every post.
    if dead_zone < 0:
        dead_zone = 0

    # Load only the columns needed
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return
    # max_words and pca_variance have delegated checks


    # Drop the ambiguous middle: posts whose |score| <= dead_zone fall in the low-density valley
    # between the under- and over-performing modes of the bimodal distribution
    # dead_zone = 0 disables the cut, keeping every post
    score = df["crowdtangle_score"]
    total = len(df)
    if dead_zone > 0:
        keep = score.abs() > dead_zone
        df   = df[keep].reset_index(drop=True)
        if df.empty or df["crowdtangle_score"].nunique() <= 1:
            print(f"ERROR: no usable posts left after dropping the |score| <= {dead_zone:g} dead zone.")
            return
        score = df["crowdtangle_score"]

    trufor = df["trufor_score"]

    # Binarise the target: 1 = over-performed (score > 0), 0 = under-performed (score <= 0).
    # With the dead zone removed every remaining post sits clearly on one side of 0.
    y = (score > 0).astype(int)

    # Vectorise the descriptions once (TF-IDF), then reuse that matrix for both feature variants:
    # the raw TF-IDF columns and their PCA-compressed version.
    tfidf = extract_tfidf(df["anomaly_txt"], max_words)
    if tfidf is None:
        print("ERROR: no classification could be computed.")
        return

    variants = [("Raw TF-IDF", tfidf)]
    pca = compress_pca(tfidf, pca_variance)
    if pca is not None:
        variants.append(("TF-IDF + PCA", pca))

    # Fit the classifier on each feature variant
    sections = []
    for title, features in variants:
        X = pd.concat([trufor, features], axis=1)

        # (a) regularised logistic regression: out-of-sample predictive metrics + odds ratios
        section = _classification_report(X, y, title)
        if section is not None:
            sections.append(section)

        # (b) binomial GLM (logit link): the inferential summary with p-values (valid on PCA features)
        glm = _glm_report(X, y, title)
        if glm is not None:
            sections.append(glm)

    if not sections:
        print("ERROR: no classification could be computed.")
        return

    header = (
        f"{'=' *77}\n"
        "Anomaly Classification\n"
        f"{'=' *77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len(y)} of {total} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len(y)})\n"
        "Logistic-regression metrics are 5-fold cross-validated (out-of-sample); the GLM table is in-sample inference.\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Classifies the anomaly descriptions (produced upstream by describe.py) by engagement:
    binarises crowdtangle_score (after dropping the distribution's valley, unless --dead-zone 0 disables it)
    and fits a logistic regression / binomial GLM on the raw and PCA-compressed TF-IDF features.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Classify the anomaly descriptions by engagement with logistic regression / a binomial GLM.")

    # Flags (some parameters are read from config.py).
    # Both feature variants (raw TF-IDF and TF-IDF + PCA) are computed and reported.
    parser.add_argument("--max-words", type=int, default=MAX_NUMBER_OF_WORDS, help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
    args = parser.parse_args()

    # Logistic classification of the descriptions
    classify(max_words=args.max_words, dead_zone=args.dead_zone)

if __name__ == "__main__":
    main()