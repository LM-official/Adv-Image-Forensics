import argparse
from   classification_home  import prepare_dataset
from   config               import (FILTERED_DATASET_FILE, EXPLANATION_FILE, MAX_NUMBER_OF_WORDS, ENGAGEMENT_DEAD_ZONE,
                                    TFIDF_MIN_DF, TFIDF_MAX_DF, EXPLAIN_TOP_N, EXPLAIN_N_REPEATS, EXPLAIN_N_SAMPLES,
                                    EXPLANATION_PLOT_FILE, EXPLAIN_PLOT_TOP_N, EXPLAIN_PLOT_UP_COLOR, EXPLAIN_PLOT_DOWN_COLOR,
                                    EXPLANATION_BAR_PLOT_FILE, EXPLAIN_BAR_TOP_N)
from   lime.lime_tabular    import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg") # Non-interactive backend: render straight to file, no display needed
import matplotlib.pyplot    as     plt
from   matplotlib.ticker    import FuncFormatter
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
# (1) WORD-LEVEL INTERPRETATION
# =============================================================================

def permutation(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, top_n: int, n_repeats: int) -> str:
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

        print("Permutation importance computed successfully.")
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
        print(f"ERROR: during permutation importance: {e}.")
        return f"=== Permutation importance (global, model-agnostic) ===\nERROR: {e}"


def shap_(model: LogisticRegression, X: pd.DataFrame, top_n: int) -> tuple[str, pd.Series | None]:
    """
    Global word importance by SHAP on the linear logistic model (exact via LinearExplainer):
    importance = mean(|SHAP|) per feature, direction = sign of the coefficient.
    For a linear model this closely tracks the coefficient ranking; it is included for the standard, citable framing.

    Args:
        model: Fitted logistic model.
        X:     Design matrix (trufor_score + tfidf words).
        top_n: Words listed per side (up/down).

    Returns:
        A (text, importances) tuple: the report section (an ERROR line on failure)
        and the mean(|SHAP|) per feature, reused for the bubble and bar charts (None on failure).
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

        print("SHAP values computed successfully.")
        return (
            "=== SHAP (global; LinearExplainer) ===\n"
            "Importance = mean(|SHAP|) per feature (coefficient weighted by the word's spread);\n"
            "direction from the logistic coefficient sign.\n\n"
            "Words pushing engagement UP:\n"
            f"{_side(signs >= 0)}\n\n"
            "Words pushing engagement DOWN:\n"
            f"{_side(signs < 0)}"
        ), imp

    except Exception as e:
        print(f"ERROR: during SHAP computation: {e}.")
        return f"=== SHAP (global) ===\nERROR: {e}", None


def lime(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, top_n: int, n_samples: int) -> str:
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

        print("LIME local explanations computed successfully.")
        return (
            "=== LIME (local, per-post) ===\n"
            f"{len(picks)} posts spanning lowest -> highest predicted engagement;\n"
            "a driver's sign is its local push toward high (+) or low (-) engagement for that post.\n\n"
            + "\n\n".join(blocks)
        )

    except Exception as e:
        print(f"ERROR: during LIME computation: {e}.")
        return f"=== LIME (local, per-post) ===\nERROR: {e}"


# =============================================================================
# (2) PLOTS
# =============================================================================

def plot(plot: Path, model: LogisticRegression, X: pd.DataFrame, tfidf: pd.DataFrame, influence: pd.Series, top_n: int):
    """
    Bubble chart of the whole vocabulary: what the three ranked text lists cannot show, namely a strong but rare word against a weak but common one.

    Each word is one bubble: the horizontal position is the logistic coefficient (which way and how hard it moves engagement, 0 = no effect),
    the vertical position is how many posts actually contain that word, and the area is the mean(|SHAP|).
    Only the most influential words per side are labelled.

    `trufor_score` is dropped: it is not a word and it is present in every post, so "how frequently it occurs" has no meaning for it.

    Args:
        plot:      Output figure path (PDF by default).
        model:     Fitted logistic model exposing coef_.
        X:         Design matrix the model was fitted on (gives the coefficient order).
        tfidf:     TF-IDF matrix alone (the word columns), used for the document frequencies.
        influence: mean(|SHAP|) per feature from shap_, used as the bubble area.
        top_n:     Words labelled per side (up/down).

    Returns:
        None (writes the figure; prints an ERROR line and returns on failure).
    """

    print("Plotting the word effects...")

    if influence is None:
        print("ERROR: no SHAP importances available; skipping the word plot.")
        return

    FIG_WIDTH  = 11
    FONT_SIZE  = 7.5
    PLOT_WIDTH_POINTS = FIG_WIDTH * 72 * 0.75

    try:
        words = list(tfidf.columns)
        coefs = pd.Series(model.coef_[0], index=X.columns)

        # One row per word: effect (x), how many posts contain it (y), aggregate influence (area)
        data = pd.DataFrame({
            "coef":      coefs.reindex(words).astype(float),
            # A word is "in" a post when its TF-IDF weight is non-zero
            "freq":      (tfidf > 0).sum() / len(tfidf) * 100,
            "influence": influence.reindex(words).astype(float),
        }, index=words)
        # A non-finite coefficient or a word present in no post cannot be placed on the axes
        data = data[np.isfinite(data["coef"]) & np.isfinite(data["influence"]) & (data["freq"] > 0)]

        if data.empty:
            print("ERROR: no plottable words; skipping the word plot.")
            return

        data["label"] = [name.removeprefix("tfidf_") for name in data.index]

        # Scale mean(|SHAP|) into a readable marker area: the least influential word must stay visible,
        # the most influential must not swallow its neighbours
        span  = float(data["influence"].max() - data["influence"].min())
        area  = 20 + (data["influence"] - data["influence"].min()) / span * 380 if span > 0 else pd.Series(120.0, index=data.index)
        up    = data["coef"] >= 0

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 7))
        for mask, color, name in ((up, EXPLAIN_PLOT_UP_COLOR, "Positivo"), (~up, EXPLAIN_PLOT_DOWN_COLOR, "Negativo")):
            ax.scatter(data.loc[mask, "coef"], data.loc[mask, "freq"], s=area[mask], color=color,
                       alpha=0.75, edgecolors="white", linewidths=0.8, label=name, zorder=3)

        # Reference line at "no effect" + recessive grid
        ax.axvline(0, color="gray", lw=1.2, zorder=1)
        ax.grid(True, color="0.9", lw=0.6, zorder=0)

        # Log scale: min_df puts the floor near 1% and packs most of the vocabulary below 5%,
        # which a linear axis would pile onto the bottom edge
        ax.set_yscale("log")
        percent = FuncFormatter(lambda value, _: f"{value:g}%")
        ax.yaxis.set_major_formatter(percent)
        ax.yaxis.set_minor_formatter(percent)
        ax.tick_params(axis="y", which="minor", labelsize=7)

        # Label only the most influential words per side, offset clear of their own bubble
        labelled = pd.concat([data[up].nlargest(top_n, "influence"), data[~up].nlargest(top_n, "influence")])
        labelled = labelled.sort_values("influence", ascending=False)

        # Normalised positions, so "too close" means the same thing on the linear x and the logarithmic y
        x_min, x_max = data["coef"].min(), data["coef"].max()
        y_min, y_max = np.log10(data["freq"].min()), np.log10(data["freq"].max())
        placed       = []

        for name, row in labelled.iterrows():
            norm_x = (row["coef"] - x_min) / (x_max - x_min)
            norm_y = (np.log10(row["freq"]) - y_min) / (y_max - y_min)
            # Half the label width as a fraction of the axes, from the character count at this font size
            half_w = len(row["label"]) * FONT_SIZE * 0.6 / 2 / PLOT_WIDTH_POINTS
            above  = not any(abs(norm_x - other_x) < (half_w + other_w) and abs(norm_y - other_y) < 0.035
                             for other_x, other_y, other_w, other_above in placed if other_above)
            placed.append((norm_x, norm_y, half_w, above))

            # Clear the bubble itself: its radius grows with the marker area
            offset = np.sqrt(area[name]) / 2 + 4
            ax.annotate(row["label"], (row["coef"], row["freq"]), textcoords="offset points",
                        xytext=(0, offset if above else -offset - FONT_SIZE), ha="center", fontsize=FONT_SIZE, zorder=4)

        ax.set_xlabel("Coefficiente logistico (log-odds)")
        ax.set_ylabel("Descrizioni che contengono la parola")
        ax.set_title("Coefficienti delle parole e frequenza nelle descrizioni")

        # Both legends outside the axes: the corners of the plot carry data
        colors_legend = ax.legend(title="Segno del\ncoefficiente", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
        ax.add_artist(colors_legend)
        sizes  = [data["influence"].max(), data["influence"].median(), data["influence"].min()]
        handles = [ax.scatter([], [], s=20 + (value - data["influence"].min()) / span * 380 if span > 0 else 120,
                              color="0.6", edgecolors="white", linewidths=0.8, label=f"{value:.4f}")
                   for value in sizes]
        ax.legend(handles=handles, title="Media |SHAP|\n(log-odds)", loc="upper left",
                  bbox_to_anchor=(1.02, 0.72), frameon=False, fontsize=8, labelspacing=1.8, borderpad=1.0)

        plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot, dpi=120, bbox_inches="tight",
                    bbox_extra_artists=(*fig.get_default_bbox_extra_artists(), colors_legend))
        plt.close(fig)
        print("Plot saved successfully.")

    except Exception as e:
        print(f"ERROR: during the word plot: {e}")
        return


def plot_bars(plot_file: Path, model: LogisticRegression, X: pd.DataFrame, tfidf: pd.DataFrame,
              influence: pd.Series | None, top_n: int = EXPLAIN_BAR_TOP_N):
    """
    Save the top SHAP word importances in two panels, grouped by coefficient sign.

    Bar length is mean(|SHAP|), not the coefficient or a probability change.
    Only TF-IDF columns are ranked; both panels use the same horizontal scale.
    Missing SHAP importances or invalid rankings skip the figure.
    """
    
    print("Plotting the SHAP word ranking...")
    if influence is None or top_n < 1:
        print("ERROR: no SHAP importances or invalid top_n; skipping the bar chart.")
        return

    fig = None
    try:
        coefs = pd.Series(model.coef_[0], index=X.columns)
        data = pd.DataFrame({"coef": coefs.reindex(tfidf.columns),
                             "influence": influence.reindex(tfidf.columns)})
        data = data[np.isfinite(data["coef"]) & np.isfinite(data["influence"])
                    & (data["influence"] >= 0) & (data["coef"] != 0)]
        sides = [data[data["coef"] > 0].nlargest(top_n, "influence"),
                 data[data["coef"] < 0].nlargest(top_n, "influence")]
        if data.empty:
            print("ERROR: no plottable words; skipping the bar chart.")
            return

        height = max(4, 1.6 + 0.36 * max(len(side) for side in sides))
        fig, axes = plt.subplots(1, 2, figsize=(11, height), sharex=True)
        limit = max(float(data["influence"].max()) * 1.25, 0.001)
        for ax, side, color, title in zip(
            axes, sides, (EXPLAIN_PLOT_UP_COLOR, EXPLAIN_PLOT_DOWN_COLOR),
            ("Coefficiente positivo", "Coefficiente negativo"),
        ):
            positions = np.arange(len(side))
            bars = ax.barh(positions, side["influence"], color=color, alpha=0.75, height=0.7)
            ax.set_yticks(positions, [name.removeprefix("tfidf_") for name in side.index])
            ax.bar_label(bars, labels=[f"{value:.4f}".replace(".", ",") for value in side["influence"]],
                         padding=4, fontsize=8)
            ax.set_ylim(max(len(side), 1) - 0.5, -0.5)
            ax.set_xlim(0, limit)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Media |SHAP| (log-odds)")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".replace(".", ",")))
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="0.92")
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            if side.empty:
                ax.text(0.5, 0.5, "Nessuna parola", transform=ax.transAxes, ha="center", color="0.4")

        fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=3)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file, bbox_inches="tight")
        print("SHAP bar chart saved successfully.")
    except Exception as e:
        print(f"ERROR: during the SHAP bar chart: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# =============================================================================
# EXECUTOR
# =============================================================================

def explain(dataset: Path = FILTERED_DATASET_FILE, report: Path = EXPLANATION_FILE, plot_file: Path = EXPLANATION_PLOT_FILE,
            max_words: int = MAX_NUMBER_OF_WORDS, dead_zone: float = ENGAGEMENT_DEAD_ZONE,
            min_df: int = TFIDF_MIN_DF, max_df: float = TFIDF_MAX_DF,
            top_n: int = EXPLAIN_TOP_N, n_repeats: int = EXPLAIN_N_REPEATS, n_samples: int = EXPLAIN_N_SAMPLES,
            plot_top_n: int = EXPLAIN_PLOT_TOP_N, bar_plot_file: Path = EXPLANATION_BAR_PLOT_FILE,
            bar_top_n: int = EXPLAIN_BAR_TOP_N):
    """
    Model-agnostic word-level interpretation of the logistic regression on the raw TF-IDF.

    The raw TF-IDF GLM does not converge (quasi-complete separation), so there are no word-level p-values.
    Instead of the model weights, this explains the well-performing predictive model directly, in terms of the
    actual words, with three complementary methods: permutation importance (global + uncertainty),
    SHAP (global) and LIME (local, per-post). Only the raw TF-IDF variant is used - PCA is excluded
    on purpose, since its components are not words (and the PCA GLM already gives valid p-values).

    Args:
        dataset:    Path to the dataset CSV (trufor_score, anomaly_txt, crowdtangle_score, filename).
        report:     Output file for the combined interpretation report.
        plot_file:  Output figure for the word bubble chart, PDF by default (not named `plot`: that is the plotting function).
        max_words:  Maximum size of the TF-IDF vocabulary.
        dead_zone:  Drop posts with |crowdtangle_score| <= this before binarising. 0 disables the cut.
        min_df:     Keep only words appearing in at least this many documents.
        max_df:     Keep only words appearing in at most this fraction of documents.
        top_n:      Words listed per side / per local explanation.
        n_repeats:  Permutation-importance shuffles per feature.
        n_samples:  Posts explained locally by LIME.
        plot_top_n: Words labelled per side in the bubble chart.
        bar_plot_file: Output PDF for the two-panel SHAP word ranking.
        bar_top_n:  Words shown per coefficient sign in the bar chart.

    Returns:
        None.
    """

    print("Explaining the logistic model word by word (permutation importance, SHAP, LIME)...")

    # Clamp to 0 (kept for a consistent header; binarize_engagement clamps internally too)
    if dead_zone < 0:
        dead_zone = 0

    if top_n < 1 or n_repeats < 1 or n_samples < 1 or plot_top_n < 1 or bar_top_n < 1:
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
        permutation_section     = permutation(model, X, y, top_n, n_repeats)
        # SHAP also hands back its per-feature importances, reused as the bubble area of the word plot
        shap_section, influence = shap_(model, X, top_n)
        sections = [
            permutation_section,
            shap_section,
            lime(model, X, y, df, top_n, n_samples),
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

    # The same words as a figure: direction and strength against how often they actually occur
    plot(plot_file, model, X, tfidf, influence, plot_top_n)
    plot_bars(bar_plot_file, model, X, tfidf, influence, bar_top_n)
    print("Word-level interpretation completed successfully.")


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