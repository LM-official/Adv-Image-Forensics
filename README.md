<div align="center">

# 🔍 AI Forensics in Advertising

**A TruFor-based pipeline for detecting image manipulation in social-media advertising, written in Python.**
It scores how likely each image is manipulated, describes what was altered, and relates manipulation to engagement.

![Language](https://img.shields.io/badge/Language-Python%203.14-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-TruFor-orange?style=flat-square)
![Captioning](https://img.shields.io/badge/Captioning-BLIP%20%2B%20EasyOCR-9cf?style=flat-square)
![License](https://img.shields.io/badge/License-MIT%20%2B%20research-lightgrey?style=flat-square)

</div>

---

## 📋 What it does

Reads a CrowdTangle-style export of social-media posts and produces a forensic analysis of their images:

- **🧹 Photo-only filtering** — keeps posts that carry an engagement score and consist only of photos that resolve to real files on disk, then reports the engagement-score distribution.
- **🧠 Manipulation scoring** — runs **TruFor** over every kept image and records a global manipulation score in `[0, 1]`.
- **🎯 Detection performance** — evaluates TruFor as a binary detector against a manually labelled ground truth (confusion matrix + accuracy/precision/recall/F1).
- **📈 Engagement statistics** — correlates (Pearson/Spearman) and OLS-regresses the manipulation signal against the CrowdTangle overperforming score.
- **🖼️ Visual features** — extracts ~19 (no-ML) image features per photo and plots how each correlates with engagement and with the manipulation score.
- **🎭 What was manipulated** — masks the altered region, captions it with **BLIP** and reads its text with **EasyOCR**.
- **🔗 Description → engagement** — turns the description into TF-IDF features (raw and PCA-compressed) and tests whether they predict whether a post over- or under-performs, with a logistic-regression / binomial-GLM classifier.
- **🔄 Dataset-agnostic** — point the `DATASET_*` paths in `config.py` at any CrowdTangle-style export to switch sources.

> 🗣️ The growing analysis table lives in `scripts/temp/filtered_dataset.csv`, while every human-readable report is written next to it as a separate `*.txt`/`*.png`: data and reports stay apart.

---

## 🗂️ Project structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/                         # this project's pipeline + configuration
│   ├── config.py                    # single source of truth (paths + tunable knobs)
│   ├── filter.py                    # stage 1 — keep photo-only scored posts + score distribution
│   ├── trufor.py                    # stage 2 — run TruFor, detector confusion matrix, correlation + regression
│   ├── features.py                  # stage 3 — visual features + plot
│   ├── mask.py                      # stage 4 — mask the altered region
│   ├── describe.py                  # stage 5 — caption (BLIP) + read (OCR) the altered region
│   ├── anomaly_classification.py    # stage 6 — TF-IDF logistic / binomial-GLM engagement classifier
│   ├── utils.py                     # shared CSV / .npz / image / text helpers
│   └── temp/                        # all generated outputs land here
├── TruFor/                          # vendored TruFor (read-only dependency, GRIP-UNINA)
└── dataset/                         # input data (crowdtangle.jsonl + crowdtangle_media/)
```

> 🛡️ **TruFor is a read-only dependency:** the pipeline never writes into `TruFor/`. It hands TruFor's `test.py` an `--input`/`--output` pair under `scripts/temp/` (`temp/images` → `temp/npz`), so every intermediate lives there.

---

## ⚙️ Setup

Developed on macOS (Apple Silicon / MPS); also runs on CUDA GPUs or CPU (device auto-selected CUDA → MPS → CPU).

**1. Clone**

```sh
git clone https://github.com/MercuriLorenzo/AI-forense-in-advertising.git
```

**2. TruFor weights** — not in the repo (gitignored). Download [`TruFor_weights.zip`](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip), unzip, and place the checkpoint at `TruFor/TruFor_train_test/weights/trufor_ph3/best.pth.tar`. The `weights_only=False` fix upstream TruFor requires is already applied in this vendored copy.

**3. Python environment**

```sh
cd AI-forense-in-advertising
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Dataset** — not in the repo (gitignored). Drop your export in `dataset/`: e.g. `crowdtangle.jsonl` (one JSON post per line) and `crowdtangle_media/` (the downloaded files).

---

## 🚀 Usage

Run the six stages in order from `scripts/` (all paths come from `config.py`):

```sh
source .venv/bin/activate
cd scripts

python filter.py                  # stage 1 — filter + engagement-score distribution
python trufor.py --generate-truth # stage 2 — TruFor scores, detector metrics, correlation + regression
python features.py                # stage 3 — visual features + plot
python mask.py                    # stage 4 — mask the altered region
python describe.py                # stage 5 — BLIP caption + OCR of the altered region
python anomaly_classification.py  # stage 6 — TF-IDF logistic / binomial-GLM classifier
```

`--generate-truth` (first run only) writes the empty `trufor_truth.csv` label template and pauses so you can fill the `label` column (`0` = real, `1` = AI); later runs reuse it.

The most useful per-stage flags:

| Option | Stage | Description |
| :----- | :---- | :---------- |
| `--batch-size`, `--wait` | `trufor.py` | Images per TruFor invocation and cool-down seconds between batches. |
| `--generate-truth`, `--threshold` | `trufor.py` | Write the `trufor_truth.csv` label template (then label 0/1); decision threshold above which an image counts as altered (default `0.5`). |
| `--method` | `features.py` | `pearson` or `spearman` correlation for the feature plot. |
| `--threshold`, `--conf-threshold` | `mask.py` | Anomaly and confidence cut-offs a pixel must clear to be kept in the mask. |
| `--batch-size`, `--do-sample`, `--no-ocr` | `describe.py` | Images per batch; sampling instead of beam search; caption-only (skip OCR). |
| `--max-words`, `--dead-zone` | `anomaly_classification.py` | TF-IDF vocabulary cap; drop the score valley (`\|score\| ≤ x`) before binarising at 0 (`0` disables the cut). |

> ⚠️ Media is matched to posts by reproducing `sha256(url)[:CHAR_LIMIT]`. `CHAR_LIMIT` (in `config.py`, default `64`) must equal the truncation used at download time, or every URL→file lookup silently misses.

---

## 🔬 How it works

A six-stage pipeline; each stage appends a column to the analysis table or writes a report (see [Architecture](#-architecture)):

1. **`filter.py`** — keeps the photo-only scored posts and symlinks their images into `scripts/temp/images/` (TruFor's input dir), then reports the `crowdtangle_score` distribution (median/skew/outliers + histogram & boxplot).
2. **`trufor.py`** — runs TruFor's `test.py` over those images in batches (**resumable**: one `.npz` per image, existing ones skipped; multi-megapixel images downscaled then upsampled back), records the global `trufor_score`, and — against a manually labelled `trufor_truth.csv` — writes the **detector confusion matrix** (accuracy/precision/recall/F1) plus the **Pearson/Spearman correlation** and an **OLS regression** of `crowdtangle_score ~ trufor_score`.
3. **`features.py`** — extracts ~19 (no-ML) visual features per image (aspect ratio, brightness, contrast, sharpness, blur, colorfulness, warmth, entropy, symmetry, rule of thirds, depth of field, visual complexity, …) with OpenCV/NumPy/SciPy/scikit-image, merges them into the table, and renders a feature scatter: each feature plotted by its correlation with engagement (`crowdtangle_score`, X) and with the AI-manipulation score (`trufor_score`, Y), with a fitted regression line and a `y = x` reference.
4. **`mask.py`** — binarizes each TruFor map into a mask of the altered region (flagged and confidently localized pixels), denoises it and fixes polarity on synthetic-majority composites, saving one loseless compressed PNG per image.
5. **`describe.py`** — captions the masked region with **BLIP** and merges the **EasyOCR** words read off it into the `anomaly_txt` column.
6. **`anomaly_classification.py`** — models `anomaly_txt`: a TF-IDF representation (English + Italian stop-words removed), used both raw (one column per word) and PCA-compressed (enough components to keep 90% of the variance). It drops the posts in the bimodal score's low-density valley (`|crowdtangle_score| ≤ ENGAGEMENT_DEAD_ZONE`), binarises the rest (over- vs under-performed), and fits — on the raw words and on the PCA components — a 5-fold cross-validated **logistic regression** (out-of-sample metrics + odds ratios) alongside an inferential **binomial GLM with a logit link** (coefficient table with p-values). *Finding on this dataset: manipulation **intensity** (`trufor_score`) doesn't predict engagement, but the manipulated-region **description** carries a weak-but-real signal (cross-validated AUC ≈ 0.64).*

---

## ✅ How the forensic verdicts are decided

The calls are governed by a handful of thresholds in [`config.py`](scripts/config.py):

- **Detection** — an image is manipulated when `trufor_score > TRUFOR_THRESHOLD` (default `0.5`; set per run with `trufor.py --threshold`).
- **Mask** — a pixel is kept only when both flagged (`anomaly ≥ MASK_THRESHOLD`) and confidently localized (`confidence ≥ MASK_CONFIDENCE_THRESHOLD`); the raw mask is then denoised (morphological open/close + small-blob removal).
- **Polarity** — on synthetic-majority composites (a real subject on a flat banner) TruFor's mask can invert; texture asymmetry detects this and flips it so the overlay is kept, not the subject.
- **Honest empties** — a near-black mask is not captioned (the model would hallucinate) and gets the fixed label `empty`.
- **OCR text** — captioning can't read, so OCR supplies the words: low-confidence and 1–2-letter recognitions are dropped, the rest de-duplicated and capped most-confident-first.
- **Engagement** — a post over-performs when `crowdtangle_score > 0`; the bimodal distribution's low-density valley (`|crowdtangle_score| ≤ ENGAGEMENT_DEAD_ZONE`) is dropped first so the cut-off doesn't slice through ambiguous posts.

---

## 🧱 Architecture

The run revolves around one growing table, `scripts/temp/filtered_dataset.csv`:

| Column | Added by | Meaning |
| :----- | :------- | :------ |
| `id` | `filter.py` | Post identifier. |
| `crowdtangle_score` | `filter.py` | CrowdTangle overperforming score (the engagement signal). |
| `filename` | `filter.py` | Media file stem (the URL hash); the merge key for every stage. |
| `trufor_score` | `trufor.py` | TruFor global manipulation score in `[0, 1]`. |
| `aspect_ratio`, `brightness`, … | `features.py` | ~19 visual features, one column each (see `FEATURE_NAMES`). |
| `anomaly_txt` | `describe.py` | Text description of the altered region (BLIP caption + OCR words). |

Configuration is the backbone: **[`config.py`](scripts/config.py)** (imported by every Python script) holds every path and tunable knob — edit it, not the scripts. Shared building blocks (CSV / `.npz` / image / text helpers) live in [`utils.py`](scripts/utils.py).

---

## ⚠️ Notes & limitations

- TruFor weights and the dataset are **not** in the repo (gitignored).
- The notion of "manipulated" is heuristic and tuned for this advertising dataset: the mask-cleanup and polarity knobs cover the observed cases, not every composite.

---

## 👥 Authors

Built and designed by:
- **[LM-official](https://github.com/LM-official?tab=repositories)**.

Powered by **[TruFor](https://grip-unina.github.io/TruFor/)** (GRIP-UNINA, CVPR 2023) — see [`TruFor/README.md`](TruFor/README.md).

---

## 📄 License

This project's own code (under `scripts/`) is released under the **MIT License** — see [LICENSE](LICENSE). **TruFor** (vendored under `TruFor/`) is third-party software by GRIP-UNINA under its own **non-commercial research license**; the [LICENSE](LICENSE) file reproduces both. Your use of this repository must comply with each.