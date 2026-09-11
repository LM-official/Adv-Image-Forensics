from pathlib import Path


# =============================================================================
# PRIMARY DIRECTORIES
# =============================================================================

# Get the current working directory
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPTS_DIR.parent
TRUFOR_DIR  = ROOT_DIR / "TruFor" / "TruFor_train_test"
TEMP_DIR    = SCRIPTS_DIR / "temp"


# =============================================================================
# DATASET
# =============================================================================

DATASET_DIR           = ROOT_DIR / "dataset"
DATASET_FILE          = DATASET_DIR / "crowdtangle.jsonl"
DATASET_IMGS_DIR      = DATASET_DIR / "crowdtangle_media"


# =============================================================================
# FILTER DATASET
# =============================================================================

FILTERED_DATASET_FILE  = TEMP_DIR / "dataset_filtered.csv"
FILTERED_IMGS_DIR      = TEMP_DIR / "images"
# This must match the char_limit used at CrowdTangle download time, otherwise every URL->file lookup silently misses.
CHAR_LIMIT             = 64
DISTRIBUTION_FILE      = TEMP_DIR / "dataset_distribution.txt"
PLOT_MAIN_COLOR = "#822433"
PLOT_SECOND_COLOR = "#676767ff"
DISTRIBUTION_PLOT_FILE = TEMP_DIR / "engagement"


# =============================================================================
# TRUFOR
# =============================================================================

TRUFOR_RUN        = TRUFOR_DIR / "test.py"
NPZ_DIR           = TEMP_DIR / "npz"
# Number of images to process per batch
TRUFOR_BATCH_SIZE = 50
# Wait time (seconds) after each batch
TRUFOR_WAIT       = 180
# Downscale images above this pixel count before inference (maps are upsampled back, so .npz arrays always match the original image size)
# Full resolution on multi-megapixel images is RAM expensive; 0 disables the cap
TRUFOR_MAX_PIXELS = 6_000_000
NPZ_COLS          = frozenset(("map", "imgsize", "score", "conf"))

# Confusion matrix
TRUTH_FILE                   = TEMP_DIR / "trufor_truth.csv"
TRUFOR_CONFUSION_MATRIX_FILE = TEMP_DIR / "trufor_confusion_matrix.txt"
TRUFOR_THRESHOLD             = 0.25

# Correlation, regression and GLM Trufor score - Crowdtangle score
CORRELATION_FILE = TEMP_DIR / "trufor_crowdtangle_correlation.txt"
REGRESSION_FILE  = TEMP_DIR / "trufor_crowdtangle_regression.txt"
GLM_FILE         = TEMP_DIR / "trufor_crowdtangle_glm.txt"


# =============================================================================
# IMAGE FEATURES
# =============================================================================

# Base output paths: features.py appends _pearson or _spearman before the extension
IMAGE_FEATURES_FILE      = TEMP_DIR / "images_features.txt"
IMAGE_FEATURES_PLOT_FILE = TEMP_DIR / "images_features.pdf"

FEATURE_PLOT_X_LABEL  = "Correlazione con il punteggio di engagement"
FEATURE_PLOT_X_TARGET = "crowdtangle_score"
FEATURE_PLOT_Y_LABEL  = "correlazione con il punteggio di TruFor"
FEATURE_PLOT_Y_TARGET = "trufor_score"
# Correlations computed in this order after a single feature extraction
FEATURE_CORR_METHODS  = ("pearson", "spearman")

# Feature tunables
# Long side an image is resized to before extraction; 0 = no resize
FEATURE_RESIZE_LONG_SIDE = 0
# Canny edge thresholds for visual_complexity
FEATURE_CANNY_LOW        = 100
FEATURE_CANNY_HIGH       = 200
# rule_of_thirds: radius around each power-point as a fraction of the image diagonal
FEATURE_THIRDS_RADIUS    = 0.10


# =============================================================================
# MASK
# =============================================================================

MASKED_DIR                = TEMP_DIR / "masked"
# Altered threshold
MASK_THRESHOLD            = 0.5
# Drops unsure pixels
MASK_CONFIDENCE_THRESHOLD = 0.30
# Mask inversion: fix TruFor polarity on majority composites,
# invert when (background flat-fraction - flagged-region flat-fraction) >= MASK_INVERT_MARGIN
MASK_INVERT_COMPOSITE     = True
# Gradient magnitude below this a pixel is flat (no texture)
MASK_FLAT_GRAD_THRESH     = 12
# Invert when the flagged region is photographic while the background is a flat banner by at least this much
MASK_INVERT_MARGIN        = 0.25
# Invert when the flagged region is at least this fraction of the image
MASK_INVERT_MIN_AREA      = 0.03
# Removes noise (erases specks, fills holes, smooths edges); <= 1 disables it
MASK_MORPH_KERNEL         = 9
# Drop area smaller than this fraction of the image area; <= 0 disables it
MASK_MIN_AREA_RATIO       = 0.002

# Confusion matrix
# Optional extra: computed only when a hand-made ground truth is present, the pipeline does not need it
# Ground-truth: white = manipulated, black = authentic
MASKED_TRUTH_DIR           = TEMP_DIR / "masked_truth"
MASK_CONFUSION_MATRIX_FILE = TEMP_DIR / "mask_confusion_matrix.txt"
# Gray value above which a ground-truth pixel counts as manipulated
MASK_TRUTH_THRESHOLD       = 127


# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# BLIP
# =============================================================================

DESCRIPTION_REPORT_FILE        = TEMP_DIR / "description_report.txt"
DESCRIPTION_TEMP_FILE          = TEMP_DIR / "description_temp.csv"
DESCRIPTION_MODEL              = "Salesforce/blip-image-captioning-large"
DESCRIPTION_BATCH_SIZE         = 30
# Wait time (seconds) after each batch
DESCRIPTION_WAIT               = 25
# Length of the BLIP caption
MIN_DESCRIPTION_TOKENS         = 5
MAX_DESCRIPTION_TOKENS         = 40
# Beam search (keep best of X generation)
DESCRIPTION_NUM_BEAMS          = 5
# Rerun produce the same captions
DESCRIPTION_SEED               = 42
# Off = deterministic beam search (generic but faithful captions)
# On = nucleus sampling for extra diversity at some fidelity cost
DESCRIPTION_DO_SAMPLE          = False
# Randomness, higher = more random
DESCRIPTION_TEMPERATURE        = 0.7
DESCRIPTION_TOP_P              = 0.9
# Sample only among the top-K candidate tokens; 0 disables
DESCRIPTION_TOP_K              = 50
# Anti-repetition
DESCRIPTION_REPETITION_PENALTY = 1.3
DESCRIPTION_NO_REPEAT_NGRAM    = 2
# Crop masked image to non-black
DESCRIPTION_CROP               = True
# if the non-black pixel fraction is <= this, skip BLIP and write EMPTY_MASK_LABEL
EMPTY_MASK_PIXEL_RATIO         = 0.02
# Must not be an English or Italian stop word: classification_home.STOP_WORDS is handed to the TF-IDF vectorizer
# otherwise the sentinel will be stripped and every mask-less post became an all-zero row - indistinguishable from a caption of pure stop words
# mask lable must survive clean_text and carries the "TruFor found nothing here" case into the model
EMPTY_MASK_LABEL               = "nomask"
# Drop duplicate consecutive words
DESCRIPTION_DEDUP_WORDS        = True
# Number of most frequent words shown in the report
DESCRIPTION_TOP_WORDS          = 10

# =============================================================================
# OCR
# =============================================================================

OCR_ENABLED        = True
OCR_GPU            = True
# Language packs
OCR_LANGS          = ("it", "en")
# Drop words with confidence below this
OCR_MIN_CONFIDENCE = 0.65
# Magnification before detection; > 1 helps small print
OCR_MAG_RATIO      = 1.5
# Drop words with less letters than this
OCR_MIN_WORD_LEN   = 3
# Keep at most this many words per image, preferring the most confident ones; 0 disables the cap
OCR_MAX_WORDS      = 20
# Margin pixels kept around the non-black areas when crops the masked image
OCR_CROP_PADDING   = 10
# Restrict recognition to a character set; None = no restriction
OCR_ALLOWLIST      = None


# =============================================================================
# (1) CLASSIFICATION HOME
# =============================================================================

# Posts whose |crowdtangle_score| <= this fall in the low-density valley between the under-over-performing modes of the bimodal score distribution
# They are dropped before binarising at 0, so the cut-off does not slice through a cluster of ambiguous posts
# Set to 0 to disable the cut and keep every post
ENGAGEMENT_DEAD_ZONE = 0.5
MAX_NUMBER_OF_WORDS = 100
# Drop words appearing in fewer than MIN documents (absolute count) or more than MAX fraction of documents
# (non-discriminative head/tail of the frequency curve)
# Tuned with --grid-search: min_df=10 keeps the ~100-word core vocabulary (CV pseudo-R2 peak); max_df is inert here
TFIDF_MIN_DF        = 5
TFIDF_MAX_DF        = 0.4

# =============================================================================
# (2) LOGISTIC REGRESSION
# =============================================================================
LOGISTIC_CLASSIFICATION_FILE = TEMP_DIR / "classification_logistic_regression.txt"

# =============================================================================
# (3) GLM
# =============================================================================

# GLM analysis of the classification (raw + PCA); distinct from GLM_FILE (trufor.py TruFor-vs-CrowdTangle GLM)
GLM_CLASSIFICATION_FILE     = TEMP_DIR / "classification_glm.txt"
# Elastic-net fallback for the GLM when the plain fit does not converge
# (quasi-complete separation on the raw TF-IDF): alpha = penalty strength (0 disables the fallback)
# L1_wt: 0 = ridge, 1 = lasso, in-between = elastic net
# TF-IDF values are small, so useful alphas and lasso are tiny
# Tuned with --grid-search: alpha=0.0007 ridge (L1_wt=0) is the CV pseudo-R2 peak
GLM_ALPHA           = 0.0007
GLM_L1_WT           = 0.0
# PCA on the TF-IDF matrix keeps the fewest components that together explain at least
# this fraction of the variance (the knee / diminishing-returns point)
PCA_VARIANCE        = 0.9

# Grid search over the two convergence strategies (matrix conditioning x regularization)
GRID_SEARCH_FILE    = TEMP_DIR / "classification_glm_grid_search.txt"
GRID_MIN_DF         = (2, 5, 10)
GRID_MAX_DF         = (0.4, 0.5, 0.6, 0.7, 0.8)
GRID_ALPHA          = (0.0, 0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.01)
# L1_wt: 0 = ridge, 1 = lasso, in-between = elastic net
GRID_L1_WT          = (0.0, 0.5, 1.0)

# =============================================================================
# (4) LIME / SHAP
# =============================================================================

# Post-hoc word-level interpretation (--explain): explains the logistic regression on the raw TF-IDF (the words) model-agnostically,
# bypassing the GLM convergence problem permutation importance (global) + SHAP (global, signed) + LIME (local, per-post)
EXPLANATION_FILE    = TEMP_DIR / "classification_explanations.txt"
# Words listed per side (up/down) in the global rankings
EXPLAIN_TOP_N       = 20
# Permutation-importance shuffles per feature: the spread over repeats is the uncertainty (p-value substitute)
EXPLAIN_N_REPEATS   = 30
# Individual posts explained locally by LIME
EXPLAIN_N_SAMPLES   = 5

# Bubble chart of the same words: x = coefficient (direction + strength), y = how many posts contain the word, area = mean|SHAP|
EXPLANATION_PLOT_FILE   = TEMP_DIR / "classification_explanations.pdf"
# Words labelled per side; the rest stay unlabelled dots
EXPLAIN_PLOT_TOP_N      = 15
# Diverging pair: raises vs lowers engagement. The direction is also carried by the x position, so it never depends on colour alone
EXPLAIN_PLOT_UP_COLOR   = "#132a7fff"
EXPLAIN_PLOT_DOWN_COLOR = "#822433"
# SHAP ranking: two panels, one per coefficient sign, using the bubble-chart colours
EXPLANATION_BAR_PLOT_FILE = TEMP_DIR / "classification_explanations_bars.pdf"
EXPLAIN_BAR_TOP_N         = 8

# =============================================================================
# (5) VISUAL + TEXT
# =============================================================================

# Nested binomial GLM comparison on:
# 1. TruFor alone,
# 2. + the visual features (whole-image craft),
# 3. + the TF-IDF description of the altered region compressed by PCA
VISUAL_TEXT_CLASSIFICATION_FILE = TEMP_DIR / "classification_visual_text.txt"
# Same in-sample and cross-validated scores as the nested-comparison report
VISUAL_TEXT_PLOT_FILE      = TEMP_DIR / "classification_model_comparison.pdf"
VISUAL_TEXT_PLOT_FIT_COLOR = PLOT_SECOND_COLOR
VISUAL_TEXT_PLOT_CV_COLOR  = PLOT_MAIN_COLOR
# Significance markers used in the visual-feature digest,
# read against the Benjamini-Hochberg q-values (FDR < .05 / .01 / .001), not the raw p-values
VISUAL_SIGNIFICANCE_LEVELS      = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
