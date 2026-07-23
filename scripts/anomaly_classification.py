import argparse
import itertools
import warnings
from   config                          import (FILTERED_DATASET_FILE, ANOMALY_CLASSIFICATION_FILE, MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE,
                                               TFIDF_MIN_DF, TFIDF_MAX_DF, GLM_ALPHA, GLM_L1_WT,
                                               GRID_SEARCH_FILE, GRID_MIN_DF, GRID_MAX_DF, GRID_ALPHA, GRID_L1_WT,
                                               PCA_VARIANCE)
import numpy                           as     np
import pandas                          as     pd
from   pathlib                         import Path
from   sklearn.decomposition           import PCA
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.linear_model            import LogisticRegression
from   sklearn.metrics                 import roc_auc_score, confusion_matrix, log_loss
from   sklearn.model_selection         import cross_val_predict, KFold
import statsmodels.api                 as     sm
from   stop_words                      import get_stop_words
from   tqdm                            import tqdm
from   utils                           import load_csv, save_report, clean_text, binarize_engagement


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


def tf_idf(descriptions: pd.Series, max_words: int, min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF, verbose: bool = True) -> pd.DataFrame | None:
    """
    Vectorises the text descriptions into a TF-IDF feature matrix (one column per word).

    TF-IDF (Term Frequency - Inverse Document Frequency) turns each description string into a numeric vector (a row of word weights).

    min_df/max_df trim the document-frequency tails: words too rare to generalise (easy to overfit on)
    and words so common they are not discriminative. 
    Cutting them is one of the two strategies to make the unregularised GLM converge on the raw TF-IDF.

    Args:
        descriptions: Series of textual descriptions, one per sample.
        max_words:    Maximum size of the TF-IDF vocabulary.
        min_df:       Keep only words appearing in at least this many documents (absolute count).
        max_df:       Keep only words appearing in at most this fraction of documents.
        verbose:      Print the "Vectorizing..." progress line.

    Returns:
        A dense DataFrame of TF-IDF weights (columns prefixed "tfidf_") indexed like descriptions,
        or None on invalid input or failure.
    """

    if descriptions is None or descriptions.empty or descriptions.nunique() <= 1:
        print("ERROR: invalid input for text feature extraction.")
        return None
    
    # max_words, min_df and max_df have delegated checks


    if verbose:
        print("Vectorizing text descriptions (TF-IDF)...")
    try:
        # Ignore common English + Italian stop words, trim the document-frequency tails (min_df/max_df),
        # limit the vocabulary to the most frequent words, and extract single words (unigrams)
        # Rows = descriptions, columns = words
        vectorizer   = TfidfVectorizer(stop_words=STOP_WORDS, max_features=max_words, min_df=min_df, max_df=max_df)
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns = [f"tfidf_{word}" for word in vectorizer.get_feature_names_out()],
            index   = descriptions.index,
        )

    except Exception as e:
        print(f"ERROR: during TF-IDF vectorization: {e}")
        return None


def pca(tfidf: pd.DataFrame, pca_variance: float) -> pd.DataFrame | None:
    """
    Compresses an existing TF-IDF matrix into a few dense latent-concept columns with PCA (Principal Component Analysis),
    which avoids the multicollinearity of the wide, sparse TF-IDF in the classifier.
    Reuses the matrix from tf_idf instead of re-vectorising.

    Keeps the fewest components that together explain `pca_variance` of the variance
    (the knee / diminishing-returns point) instead of a guessed count.

    Args:
        tfidf:        TF-IDF feature matrix from tf_idf.
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

def logistic_regression(X: pd.DataFrame, y: pd.Series, title: str, top_n: int = 15) -> str | None:
    """
    Fits a logistic regression classifier predicting the binary engagement label from the visual (trufor_score) and textual features (X),
    and returns its diagnostics as a labelled text section.

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

    print(f"Computing logistic regression ({title})...")
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
        print(f"ERROR: during logistic regression ({title}): {e}")
        return None


def glm(X: pd.DataFrame, y: pd.Series, title: str, alpha: float = GLM_ALPHA, l1_wt: float = GLM_L1_WT, top_n: int = 15) -> str | None:
    """
    Fits a binomial GLM with a logit link (the inferential twin of the logistic classifier) and
    returns its coefficient summary.

    statsmodels GLM is unregularised, so on the wide, sparse raw TF-IDF matrix it hits quasi-complete separation:
    the coefficients diverge and the standard errors / p-values become meaningless.
    Non-convergence or a non-finite log-likelihood fall back to an elastic-net regularised fit:
    the penalty shrinks the exploding coefficients toward zero (lasso even disables the useless ones entirely),
    so the fit is stable and the coefficients readable - but a regularised fit has no standard errors, hence no p-values.
    On the compact PCA features the plain GLM is well posed and the full summary is valid.

    Args:
        X:     Design matrix (visual + textual features), index-aligned with y.
        y:     Binary target (1 = high engagement, 0 = low), index-aligned with X.
        title: Human-readable name of the feature variant (e.g. "Raw TF-IDF").
        alpha: Elastic-net penalty strength for the fallback; 0 disables the fallback.
        l1_wt: Mix of the penalty: 0 = ridge, 1 = lasso, in-between = elastic net.
        top_n: How many coefficients to list on each side in the regularised fallback.

    Returns:
        The report section as text, or None if the model could not be fitted at all.
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

        return (
            f"=== {title} - binomial GLM (logit link) ===\n"
            "Skipped: the unregularised GLM did not converge (quasi-complete separation on the sparse TF-IDF features),\n"
            "so its coefficients and p-values are not interpretable"
            + (" (and the regularised fallback produced non-finite coefficients too).\n" if alpha > 0 else
               " (and the regularised fallback is disabled: alpha = 0).\n")
            + "Use the cross-validated logistic-regression metrics;\n"
            "the GLM inference is reliable on the PCA-compressed features."
        )

    except Exception as e:
        print(f"ERROR: during binomial GLM ({title}): {e}")
        return None


def classify(dataset: Path = FILTERED_DATASET_FILE, report: Path = ANOMALY_CLASSIFICATION_FILE,
             max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
             min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF, alpha: float = GLM_ALPHA, l1_wt: float = GLM_L1_WT,
             pca_variance: float = PCA_VARIANCE):
    """
    Turns crowdtangle_score into a binary engagement label and fits a logistic classifier and inferential binomial-GLM twice:
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
        dead_zone:    Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising.
                      Negatives are clamped to 0. 0 disables the cut and keeps every post.
        min_df:       Keep only words appearing in at least this many documents (see tf_idf).
        max_df:       Keep only words appearing in at most this fraction of documents.
        alpha:        Elastic-net strength for the GLM fallback when the plain fit does not converge; 0 disables it.
        l1_wt:        Mix of that penalty: 0 = ridge, 1 = lasso, in-between = elastic net.
        pca_variance: Fraction of variance the PCA variant must retain.

    Returns:
        None.
    """

    # Load only the columns needed
    df = load_csv(dataset, ["trufor_score", "anomaly_txt", "crowdtangle_score"])
    if df is None:
        return

    # Clamp to 0
    if dead_zone < 0:
        dead_zone = 0

    # Validate the TF-IDF regularization parameters
    if alpha < 0:
        print("ERROR: alpha must be >= 0 (0 disables the regularized fallback).")
        return
    if not 0 <= l1_wt <= 1:
        print("ERROR: l1_wt must be in [0, 1] (0 = ridge, 1 = lasso, in-between = elastic net).")
        return

    # max_words, min_df, max_df and pca_variance have delegated checks


    # Shared dead-zone cut + binarisation
    result = binarize_engagement(df, dead_zone)
    if result is None:
        return
    df, y  = result


    # Vectorise the descriptions once (TF-IDF), then reuse that matrix for both feature variants:
    # the raw TF-IDF columns and their PCA-compressed version
    tfidf = tf_idf(df["anomaly_txt"], max_words, min_df, max_df)
    if tfidf is None:
        print("ERROR: during TF-IDF vectorization.")
        return
    variants = [("Raw TF-IDF", tfidf)]

    # PCA
    compress_pca = pca(tfidf, pca_variance)
    if compress_pca is None:
        print("ERROR: during PCA compression.")
        return
    variants.append(("TF-IDF + PCA", compress_pca))


    # Fit the classifier on each feature variant
    sections = []
    for title, features in variants:
        X = pd.concat([df["trufor_score"], features], axis=1)

        # (a) regularised logistic regression: out-of-sample predictive metrics + odds ratios
        section1 = logistic_regression(X, y, title)
        if section1 is not None:
            sections.append(section1)

        # (b) binomial GLM (logit link): the inferential summary with p-values when the plain fit converges,
        # otherwise the regularised fallback
        section2 = glm(X, y, title, alpha, l1_wt)
        if section2 is not None:
            sections.append(section2)

    if not sections:
        print("ERROR: no classification could be computed.")
        return


    # save report
    len_y = len(y)
    header = (
        f"{'=' *77}\n"
        "Anomaly Classification\n"
        f"{'=' *77}\n"
        f"Target label : crowdtangle_score > 0 (over-performed) vs < 0 (under-performed)\n"
        f"Dead zone    : |score| <= {dead_zone:g} (kept {len_y} of {len(df)} posts)\n"
        f"Class balance: {int(y.sum())} high / {int((y == 0).sum())} low  (out of {len_y})\n"
        f"TF-IDF       : min_df={min_df}, max_df={max_df:g}, vocabulary {tfidf.shape[1]} words (cap {max_words})\n"
        "Logistic-regression metrics are 5-fold cross-validated (out-of-sample); the GLM table is in-sample inference.\n\n\n"
    )
    save_report(header + "\n\n\n".join(sections), report)


# =============================================================================
# (3) GRID SEARCH
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
                min_dfs: tuple = GRID_MIN_DF, max_dfs: tuple = GRID_MAX_DF,
                alphas: tuple = GRID_ALPHA, l1_wts: tuple = GRID_L1_WT):
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
    len_pairs = len(pairs)
    rows  = []

    print(f"Grid search: {len_pairs} min_df/max_df pairs x regularization grid...")
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Classifies the anomaly descriptions (produced upstream by describe.py) by engagement:
    binarises crowdtangle_score (after dropping the distribution's valley, unless --dead-zone 0 disables it)
    and fits a logistic regression / binomial GLM on the raw and PCA-compressed TF-IDF features.

    With --grid-search it instead explores every min_df x max_df x alpha x L1_wt combination
    to find the settings that make the GLM converge with the best out-of-sample fit.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Classify the anomaly descriptions by engagement with logistic regression / a binomial GLM, or grid-search the TF-IDF/regularization parameters.")

    # Flags (some parameters are read from config.py).
    # Both feature variants (raw TF-IDF and TF-IDF + PCA) are computed and reported.
    parser.add_argument("--max-words", type=int, default=MAX_NUMBER_OF_WORDS, help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")

    # TF-IDF conditioning and GLM regularization (the two convergence strategies)
    parser.add_argument("--min-df", type=int,   default=TFIDF_MIN_DF, help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df", type=float, default=TFIDF_MAX_DF, help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    parser.add_argument("--alpha",  type=float, default=GLM_ALPHA,    help=f"Elastic-net strength for the GLM fallback when the plain fit does not converge; 0 disables it. Default: {GLM_ALPHA}")
    parser.add_argument("--l1-wt",  type=float, default=GLM_L1_WT,    help=f"Mix of that penalty: 0 = ridge, 1 = lasso. Default: {GLM_L1_WT}")

    # Grid search over the parameters above (grids come from config.py)
    parser.add_argument("--grid-search", action="store_true", help="Run the min_df x max_df x alpha x L1_wt grid search instead of the classification.")
    args = parser.parse_args()


    if args.grid_search:
        # Systematic exploration of the convergence strategies
        grid_search(max_words=args.max_words, dead_zone=args.dead_zone)
        return


    # Logistic classification of the descriptions
    classify(max_words=args.max_words, dead_zone=args.dead_zone, min_df=args.min_df, max_df=args.max_df, alpha=args.alpha, l1_wt=args.l1_wt)

if __name__ == "__main__":
    main()