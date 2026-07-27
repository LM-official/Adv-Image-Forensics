import argparse
from   config                          import (SCRIPTS_DIR, MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE, TFIDF_MIN_DF, TFIDF_MAX_DF)
import pandas                          as     pd
from   pathlib                         import Path
from   sklearn.decomposition           import PCA
from   sklearn.feature_extraction.text import TfidfVectorizer
from   stop_words                      import get_stop_words
import subprocess
import sys
from   utils                           import load_csv, clean_text, binarize_engagement


# =============================================================================
# TEXT FEATURES (shared by the sibling classification_*.py modules)
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


def prepare_dataset(dataset: Path, dead_zone: float, max_words: int, min_df: int, max_df: float,
            columns: tuple = ("trufor_score", "anomaly_txt", "crowdtangle_score")) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None:
    """
    Shared data preparation for the other classification modules:
    loads the needed columns, drops the dead zone and binarises the engagement label and vectorises the descriptions with TF-IDF.
    binarize_engagement resets the index, so the returned df / y / tfidf stay
    position-aligned (callers can add PCA on top of the tfidf and concatenate trufor_score).

    Args:
        dataset:   Path to the dataset CSV.
        dead_zone: Drop posts with |crowdtangle_score| <= this before binarising (0 disables the cut).
        max_words: Maximum size of the TF-IDF vocabulary.
        min_df:    Keep only words appearing in at least this many documents.
        max_df:    Keep only words appearing in at most this fraction of documents.
        columns:   Columns to load (add "filename" when the caller needs per-post context).

    Returns:
        A (df, y, tfidf) tuple, or None on failure.
    """

    # Load only the columns needed
    df = load_csv(dataset, list(columns))
    if df is None:
        return None

    # Shared dead-zone cut + binarisation (binarize_engagement clamps a negative dead_zone to 0)
    result = binarize_engagement(df, dead_zone)
    if result is None:
        return None
    df, y = result

    # Vectorise the descriptions (TF-IDF)
    tfidf = tf_idf(df["anomaly_txt"], max_words, min_df, max_df)
    if tfidf is None:
        return None

    return df, y, tfidf


# =============================================================================
# MAIN
# =============================================================================

# The classification modules, each is a standalone script that writes its own report.
# They import the shared text features (tf_idf / pca / prepare_dataset) from this module.
# They run one after another, so each step gets the whole machine power.
CLASSIFICATION_SCRIPTS = (
    "classification_logistic_regression.py",  # cross-validated logistic classification (LOGISTIC_CLASSIFICATION_FILE)
    # The grid search is a separate  mode (classification_glm.py --grid-search) and is not part of this default run.
    "classification_glm.py",         # inferential binomial GLM, raw + PCA     (GLM_CLASSIFICATION_FILE)
)

def _run_script(script: str, forwarded: list[str]) -> int:
    """
    Runs one classification_*.py as a child process, forwarding the shared TF-IDF flags.
    Each script is fully independent: it runs in its own process, so a crash in one never stops the others.

    Args:
        script:    Filename of the sibling script.
        forwarded: Command-line flags to pass through (already stringified).

    Returns:
        The child's exit code, or 1 if the child could not be launched (0 = success).
    """

    cmd = [
        # Current Python interpreter (.venv)
        sys.executable,
        # File
        str(SCRIPTS_DIR / script),
        # Flags
        *forwarded
    ]

    print(f"\n{'=' * 77}\n{script}\n{'=' * 77}", flush=True)

    try:
        return subprocess.run(cmd).returncode
    except Exception as e:
        # Launching failed (not a crash inside the script)
        print(f"ERROR: could not run {script}: {e}")
        return 1


def main():
    """
    Last pipeline phase entry point: runs the whole classification phase on the anomaly descriptions by launching
    the three sibling scripts as subprocesses, one after another so each gets the full machine. Each writes its
    own report:
      1. classification_logistic_regression.py -> the cross-validated logistic classification (LOGISTIC_CLASSIFICATION_FILE)
      2. classification_glm.py        -> the inferential binomial GLM, raw + PCA       (GLM_CLASSIFICATION_FILE)
      3. classification_lime_shap.py  -> the post-hoc word-level interpretation        (EXPLANATION_FILE)

    This module also exports the shared text features (STOP_WORDS / tf_idf / pca / prepare_dataset) those script import.
    The GLM grid search is a separate exploratory mode (classification_glm.py --grid-search), not run here.

    Args:
        None (arguments are read from the command line via argparse).

    Returns:
        None. Exits non-zero if any script fails.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run the full anomaly-classification phase (logistic classification, binomial GLM, post-hoc LIME/SHAP) as subprocesses, one after another.")

    # Flags shared by the three scripts, forwarded to each
    parser.add_argument("--max-words", type=int,   default=MAX_NUMBER_OF_WORDS, help=f"Maximum number of TF-IDF words. Default: {MAX_NUMBER_OF_WORDS}")
    parser.add_argument("--dead-zone", type=float, default=ENGAGEMENT_DEAD_ZONE, help=f"Drop posts with |crowdtangle_score| <= this (the distribution's valley) before binarising; 0 disables the cut. Default: {ENGAGEMENT_DEAD_ZONE}")
    parser.add_argument("--min-df",    type=int,   default=TFIDF_MIN_DF, help=f"Keep only words appearing in at least this many descriptions. Default: {TFIDF_MIN_DF}")
    parser.add_argument("--max-df",    type=float, default=TFIDF_MAX_DF, help=f"Keep only words appearing in at most this fraction of descriptions. Default: {TFIDF_MAX_DF}")
    args = parser.parse_args()


    len_scripts = len(CLASSIFICATION_SCRIPTS)
    print(f"Running {len_scripts} classification modules one after another...")

    # Flags passed through to every child script
    forwarded = ["--max-words", str(args.max_words), "--dead-zone", str(args.dead_zone), "--min-df", str(args.min_df), "--max-df", str(args.max_df)]
    # Run the next script after the previous script end
    # The scripts are independent (separate processes, separate reports), so a crash in one never stops the others
    failed = []
    for script in CLASSIFICATION_SCRIPTS:
        if _run_script(script, forwarded) != 0:
            failed.append(script)
    len_failed = len(failed)
    print(f"\n{len_scripts - len_failed} of {len_scripts} classification scripts completed successfully.")
    if failed:
        print(f"{len_failed} of {len_scripts} classification scripts failed: {', '.join(failed)}")

if __name__ == "__main__":
    main()