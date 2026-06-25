import argparse
import subprocess
import sys
from   config                          import (SCRIPTS_DIR, FILTERED_DATASET_FILE, MAX_NUMBER_OF_WORDS,
                                               PCA_VARIANCE, ANOMALY_REGRESSION_FILE, ANOMALY_CLASSIFICATION_FILE)
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


# TF-IDF stop words: the descriptions mix English BLIP captions with English-Italian OCR text,
# so we union both languages from the `stop-words` package.
# Each word is run through clean_text (the same applied to the descriptions) so match the tokens
STOP_WORDS = sorted({
    word
    for lang in ("english", "italian")
    for word in clean_text(" ".join(get_stop_words(lang)), dedup_words=True).split()
    if len(word) >= 2
})


def extract_text_features(descriptions: pd.Series, max_words: int, pca_variance: float, use_pca: bool) -> pd.DataFrame | None:
    """
    Converts the text descriptions into a numerical feature matrix.

    TF-IDF (Term Frequency - Inverse Document Frequency) always runs: it turns each description string into a numeric vector (a row of word weights).
    PCA (Principal Component Analysis) is optional (`use_pca`): when enabled it compresses the wide, sparse TF-IDF matrix
    into a few dense latent-concept columns (which avoids multicollinearity in the regression),
    keeping the fewest components that together explain `pca_variance` of the variance.
    When disabled the raw TF-IDF columns are used directly.

    Args:
        descriptions: Series of textual descriptions, one per sample.
        max_words:    Maximum size of the TF-IDF vocabulary.
        pca_variance: Fraction of variance PCA must retain, which sets the component count (only used when use_pca is True).
        use_pca:      If True apply PCA on top of TF-IDF, otherwise return raw TF-IDF features.

    Returns:
        A DataFrame of numerical features indexed like descriptions, or None on invalid input or failure.
    """

    if descriptions is None or descriptions.empty or descriptions.nunique() <= 1 or max_words <= 0:
        print("ERROR: invalid input for text feature extraction.")
        return None


    print("Vectorizing text descriptions (TF-IDF)...")
    try:
        # TF-IDF
        # Ignore common English + Italian stop words and limit the vocabulary to the most frequent words
        # Extracts single words (unigrams)
        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, max_features=max_words)
        # Build the weight matrix (rows = descriptions, columns = words)
        tfidf_matrix = vectorizer.fit_transform(descriptions)


        # PCA disabled: return the raw TF-IDF matrix as a dense DataFrame
        if not use_pca:
            print("PCA disabled: using raw TF-IDF features.")
            return pd.DataFrame(
                tfidf_matrix.toarray(),
                columns = [f"tfidf_{word}" for word in vectorizer.get_feature_names_out()],
                index   = descriptions.index,
            )


        # PCA enabled
        # Keep the fewest components that together explain `pca_variance` of the variance
        # (the knee / diminishing-returns point) instead of a guessed count.
        # svd_solver="full" is required for a fractional n_components, and PCA centers the data, hence the dense .toarray().
        print(f"Applying PCA (retaining {pca_variance:.0%} of the variance)...")
        pca          = PCA(n_components=pca_variance, svd_solver="full")
        pca_features = pca.fit_transform(tfidf_matrix.toarray())
        print(f"PCA kept {pca.n_components_} components ({pca.explained_variance_ratio_.sum():.1%} of the variance).")

        return pd.DataFrame(
            pca_features,
            columns = [f"text_pca_{i + 1}" for i in range(pca.n_components_)],
            index   = descriptions.index,
        )

    except Exception as e:
        print(f"ERROR: during text feature extraction: {e}")
        return None


def _regression_report(X: pd.DataFrame, Y: pd.Series, target_metric: str, title: str) -> str | None:
    """
    Fits an Ordinary Least Squares (OLS) regression measuring how well the combined visual
    (trufor_score) and textual features (X) predict the target metric (Y), and returns its
    summary as a labelled text section .

    Args:
        X:             Design matrix with the independent variables (visual + textual features).
        Y:             Series with the dependent variable (target metric to predict).
        target_metric: Name of the target metric, used only in the report.
        title:         Human-readable name of the feature variant (e.g. "Raw TF-IDF").

    Returns:
        The report section as text, or None if the model could not be fitted.
    """

    print(f"Computing regression to predict '{target_metric}' ({title})...")
    try:
        # In statsmodels we must explicitly add a constant to X to calculate the intercept (q in y = mx + q)
        X = sm.add_constant(X)

        # Define and fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(Y, X).fit()

        # Report section
        return (
            f"=== {title} ===\n"
            f"Target Metric: {target_metric}\n"
            f"Total valid samples: {len(Y)}\n"
            f"Feature columns: {', '.join(X.columns)}\n\n"
            f"{model.summary().as_text()}"
        )

    except Exception as e:
        print(f"ERROR: during OLS regression calculation ({title}): {e}")
        return None


def _term_intelligibility(tfidf: pd.DataFrame, Y: pd.Series, top_n: int = 15) -> str | None:
    """
    Two simple, human-readable checks on the raw TF-IDF terms:
    (1) In the regression: list the words with the largest OLS coefficients, so we can see at a glance whether
        the influential features are real words (meloni, elezioni, ...) and not OCR junk (th, xq, ...).
    (2) High vs low: split the posts at the median crowdtangle_score and list the words whose average TF-IDF
        weight differs most between the two halves, i.e. the words that best characterise each group.

    Args:
        tfidf: Raw TF-IDF feature matrix (columns prefixed "tfidf_"), index-aligned with Y.
        Y:     Target metric (crowdtangle_score), index-aligned with tfidf.
        top_n: How many terms to list on each side.

    Returns:
        The diagnostics section as text, or None on failure.
    """

    print("Computing TF-IDF term intelligibility (raw terms)...")
    try:
        # The TF-IDF columns are named "tfidf_<word>"; strip that prefix so the report shows the bare word
        terms = tfidf.rename(columns=lambda column: column.removeprefix("tfidf_"))

        # (1): are the words that drive the regression real, meaningful words?
        # Fit an OLS of the target on the word columns alone.
        # Every word then has a coefficient telling how much that word pushes the predicted crowdtangle_score up (positive) or down (negative);
        # a large magnitude means the word weighs heavily on the model.
        # Listing the heaviest words lets us eyeball them: real terms (meloni, elezioni, ...) => intelligible features;
        # junk fragments (th, xq, ...) => not.
        # sm.add_constant adds the intercept column (dropped before ranking)
        # sort_values() orders the coefficients ascending, so the most negative end up first and the most positive last
        coefs = sm.OLS(Y, sm.add_constant(terms)).fit().params.drop("const", errors="ignore").sort_values()
        # tail(top_n) = the most positive coefficients (reversed with [::-1] so the strongest reads first)
        pos   = ", ".join(f"{word} ({coef:+.3f})" for word, coef in coefs.tail(top_n)[::-1].items())
        # head(top_n) = the most negative ones.
        neg   = ", ".join(f"{word} ({coef:+.3f})" for word, coef in coefs.head(top_n).items())

        # (2): do the words separate over-under-performing posts?
        # Split the posts into two halves at the median crowdtangle_score: "high" (strictly above the median) and the rest ("~high", i.e. low)
        median = Y.median()
        high   = Y > median
        # For each word, average its TF-IDF weight within each half and subtract them.
        # This "gap" says which half leans on the word more: a positive gap => the word is heavier in high-engagement posts,
        # a negative gap => heavier in low-engagement posts.
        # sort_values() ascending puts the strongest low-engagement words first
        diff   = (terms[high].mean() - terms[~high].mean()).sort_values()
        # tail(top_n) = biggest positive gaps (words typical of high-engagement posts, reversed so strongest first)
        hi     = ", ".join(f"{word} ({gap:+.4f})" for word, gap in diff.tail(top_n)[::-1].items())
        # head(top_n) = biggest negative gaps (words typical of low-engagement posts)
        lo     = ", ".join(f"{word} ({gap:+.4f})" for word, gap in diff.head(top_n).items())

        return (
            "=== Raw TF-IDF term intelligibility ===\n"
            f"Vocabulary size: {terms.shape[1]} terms\n\n"
            "Largest OLS coefficients (effect on crowdtangle_score; an intelligibility check, not significance):\n"
            f"\tpositive: {pos}\n"
            f"\tnegative: {neg}\n\n"
            "Most discriminative terms, high vs low crowdtangle_score "
            f"(median split at {median:.2f}: {int(high.sum())} high / {int((~high).sum())} low; "
            "value = mean TF-IDF gap):\n"
            f"\thigh-engagement: {hi}\n"
            f"\tlow-engagement:  {lo}"
        )

    except Exception as e:
        print(f"ERROR: during TF-IDF term intelligibility: {e}")
        return None


def regression(dataset: Path = FILTERED_DATASET_FILE, max_words: int = MAX_NUMBER_OF_WORDS,
               pca_variance: float = PCA_VARIANCE, report: Path = ANOMALY_REGRESSION_FILE):
    """
    Loads the dataset and runs the OLS regression predicting the crowdtangle score twice:
    once on the raw TF-IDF features and once on their PCA-compressed version.

    Args:
        dataset:      Path to the dataset CSV (must contain trufor_score, anomaly_txt, crowdtangle_score).
        max_words:    Maximum size of the TF-IDF vocabulary.
        pca_variance: Fraction of variance the PCA variant must retain (sets its component count).
        report:       Output file path where the combined regression summary will be written.

    Returns:
        None.
    """

    # Load only the columns needed for the regression (load_csv drops rows with missing values)
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return

    # Dependent variable (Y): the metric we want to predict
    # trufor_score is shared by both variants
    Y      = df["crowdtangle_score"].reset_index(drop=True)
    trufor = df["trufor_score"].reset_index(drop=True)

    # Run the regression on both feature variants: raw TF-IDF and TF-IDF + PCA
    sections = []
    for title, use_pca in (("Raw TF-IDF", False), ("TF-IDF + PCA", True)):
        # Extract the numerical text features from the anomaly descriptions
        features = extract_text_features(df["anomaly_txt"], max_words, pca_variance, use_pca)
        if features is None:
            continue
        # Independent variables (X): the visual anomaly score combined with the textual features
        features = features.reset_index(drop=True)
        X        = pd.concat([trufor, features], axis=1)
        section  = _regression_report(X, Y, "crowdtangle_score", title)
        if section is not None:
            sections.append(section)

        # On the interpretable raw TF-IDF terms, add the term-intelligibility diagnostics
        if not use_pca:
            diagnostics = _term_intelligibility(features, Y)
            if diagnostics is not None:
                sections.append(diagnostics)

    if not sections:
        print("ERROR: no regression could be computed.")
        return

    # Save both variants in a single report
    save_report("Anomaly Regression\n\n" + "\n\n\n".join(sections), report)


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
            f"=== {title} ===\n"
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


def classify(dataset: Path = FILTERED_DATASET_FILE, max_words: int = MAX_NUMBER_OF_WORDS,
             pca_variance: float = PCA_VARIANCE, overperforming_cut: bool = False,
             report: Path = ANOMALY_CLASSIFICATION_FILE):
    """
    Loads the dataset, turns crowdtangle_score into a binary engagement label and
    fits a logistic regression classifier twice:
    (1) on the raw TF-IDF features,
    (2) on their PCA compressed version.
    Write both to one report.

    The continuous crowdtangle_score is wildly skewed, so a binary high/low label sidesteps the OLS assumptions that the raw score violates;
    the classifier asks "does the manipulation description make a post more likely to overperform?".

    Args:
        dataset:            Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score).
        max_words:          Maximum size of the TF-IDF vocabulary.
        pca_variance:       Fraction of variance the PCA variant must retain.
        overperforming_cut: If True label posts by crowdtangle_score > 0 (over-performed), else split at the median.
        report:             Output file for the combined classification summary.

    Returns:
        None.
    """

    # Load only the columns needed (load_csv drops rows with missing values)
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return

    score  = df["crowdtangle_score"].reset_index(drop=True)
    trufor = df["trufor_score"].reset_index(drop=True)

    # Binarise the target into high (1) / low (0) engagement: either at 0 (CrowdTangle's overperform line)
    # or at the median (a guaranteed-balanced split)
    threshold = 0.0 if overperforming_cut else score.median()
    target    = "crowdtangle_score > 0 (overperformed)" if overperforming_cut else f"crowdtangle_score > median ({threshold:.2f})"
    y         = (score > threshold).astype(int)

    # Fit the classifier on both feature variants: raw TF-IDF and TF-IDF + PCA
    sections = []
    for title, use_pca in (("Raw TF-IDF", False), ("TF-IDF + PCA", True)):
        features = extract_text_features(df["anomaly_txt"], max_words, pca_variance, use_pca)
        if features is None:
            continue
        X       = pd.concat([trufor, features.reset_index(drop=True)], axis=1)
        section = _classification_report(X, y, title)
        if section is not None:
            sections.append(section)

    if not sections:
        print("ERROR: no classification could be computed.")
        return

    header = (
        "Anomaly Classification\n"
        f"Target label : {target}\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len(y)})\n"
        "Metrics are 5-fold cross-validated (out-of-sample).\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


def main():
    """
    Runs the optional steps:
    1. masking (delegated to mask.py via subprocess),
    2. anomaly description (with/without the OCR words of the altered region merged in),
    3. regression on the descriptions,
    4. logistic classification of the descriptions.

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
    
    # Regression (some parameters are read from config.py).
    # Both feature variants (raw TF-IDF and TF-IDF + PCA) are computed and reported.
    parser.add_argument("--skip-regression",     action="store_true",                          help="Skip step 3 (OLS regression).")
    parser.add_argument("--max-words",           type=int,    default=MAX_NUMBER_OF_WORDS,     help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    
    # Classification
    parser.add_argument("--skip-classification", action="store_true",                          help="Skip step 3b (logistic classification).")
    parser.add_argument("--overperforming-cut",  action="store_true",                          help="Binarise engagement at crowdtangle_score > 0 instead of the median.")
    args = parser.parse_args()


    # Masking
    if not args.skip_mask:
        command = [sys.executable, str(SCRIPTS_DIR / "mask.py")]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Masking failed: {e}")
            return

    # Anomaly description
    if not args.skip_describe:
        command = [sys.executable, str(SCRIPTS_DIR / "describe.py")]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Description failed: {e}")
            return

    # Regression on the descriptions
    if not args.skip_regression:
        regression(max_words=args.max_words)

    # Logistic classification of the descriptions
    if not args.skip_classification:
        classify(max_words=args.max_words, overperforming_cut=args.overperforming_cut)

if __name__ == "__main__":
    main()