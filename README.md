<div align="center">

# 🔍 AI Forensics in Advertising

**A TruFor-based pipeline for detecting image manipulation in social-media advertising, written in Python.**
It scores how likely each image is manipulated, describes *what* was altered,
and relates manipulation to engagement.

![Language](https://img.shields.io/badge/Language-Python%203.14-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-TruFor%20(CVPR%202023)-orange?style=flat-square)
![Captioning](https://img.shields.io/badge/Captioning-BLIP%20%2B%20EasyOCR-9cf?style=flat-square)
![License](https://img.shields.io/badge/License-MIT%20%2B%20research-lightgrey?style=flat-square)

</div>

---

## 📋 What it does

Reads a CrowdTangle-style export of social-media posts and produces a forensic analysis of their images:

- **🧹 Photo-only filtering** — keeps posts that carry an engagement score and consist only of photos that resolve to real files on disk.
- **🧠 Manipulation scoring** — runs **TruFor** over every kept image and records a global manipulation score in `[0, 1]`.
- **🎯 Detection performance** — evaluates TruFor as a binary detector against a manually labelled ground truth.
- **📈 Engagement statistics** — correlates and regresses the manipulation signal against the CrowdTangle overperforming score.
- **🎭 What was manipulated** — masks the altered region, captions it with **BLIP** and reads its text with **EasyOCR**.
- **🔗 Description ↔ engagement** — turns the description into TF-IDF features (raw and PCA-compressed) and tests whether they predict the engagement score, with an OLS regression and a logistic high/low classifier.
- **🔄 Dataset-agnostic** — point the `DATASET_*` paths in `config.py` at any CrowdTangle-style export to switch sources.

> 🗣️ The growing analysis table lives in `scripts/temp/filtered_dataset.csv`, while every human-readable report is written next to it as a separate `*.txt`/`*.png`: data and reports stay apart.

---

## 🗂️ Project structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/                    # this project's pipeline + configuration
│   ├── run.sh                  # entry point: guided y/n walk through the pipeline
│   ├── cleaner.sh              # interactive cleanup of generated artifacts
│   ├── config.py / config.sh   # mirrored single source of truth (paths + knobs)
│   ├── filter.py               # stage 1 — keep photo-only scored posts
│   ├── trufor.py               # stage 2 — run TruFor in batches
│   ├── performance.py          # stage 3 — detector metrics vs. ground truth
│   ├── general_stats.py        # stage 4 — correlation / regression / distribution
│   ├── anomaly_regression.py   # stage 5 — mask → describe → TF-IDF regression + logistic classification
│   ├── mask.py / describe.py / ocr.py   # masking, captioning and OCR helpers
│   ├── utils.py                # shared CSV / .npz / image / text helpers
│   └── temp/                   # all generated outputs land here
├── TruFor/                     # vendored TruFor (read-only dependency, GRIP-UNINA)
└── dataset/                    # input data + pipeline-generated images/maps
```

> 🛡️ **TruFor is a read-only dependency:** the pipeline never writes into `TruFor/`. It hands `--input dataset/images` / `--output dataset/output` to TruFor's `test.py`, so all results live under `dataset/` and every intermediate under `scripts/temp/`.

---

## ⚙️ Setup

Developed on macOS (Apple Silicon / MPS); also runs on CUDA GPUs or CPU (device auto-selected CUDA → MPS → CPU).

**1. Clone**

```sh
git clone https://github.com/MercuriLorenzo/AI-forense-in-advertising.git
cd AI-forense-in-advertising
```

**2. TruFor weights** — not in the repo (gitignored). Download [`TruFor_weights.zip`](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip), unzip, and place the checkpoint at `TruFor/TruFor_train_test/weights/trufor_ph3/best.pth.tar`. The `weights_only=False` fix upstream TruFor requires is **already applied** in this vendored copy.

**3. Python environment**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Dataset** — not in the repo (gitignored). Drop your export in `dataset/`: `crowdtangle.jsonl` (one JSON post per line) and `crowdtangle_media/` (the downloaded files). BLIP and EasyOCR weights download automatically on first use.

---

## 🚀 Usage

Two shell orchestrators drive the pipeline; both prompt `y/n` before every step:

| Command | Action |
| :------ | :----- |
| `./run.sh` | ▶️ Activates the venv and walks the whole pipeline stage by stage. |
| `./cleaner.sh` | 🧹 Interactively deletes generated artifacts (CSV, symlinks, reports, masks, caches — and, if you confirm, the expensive `.npz` maps). |

```sh
source .venv/bin/activate
cd scripts
./run.sh
```

Or run a single stage directly (all paths come from `config.py`):

```sh
python filter.py
python trufor.py
python performance.py --generate-truth
python general_stats.py
python anomaly_regression.py                       # mask + describe + regression + classification
```

The most useful per-stage flags:

| Option | Stage | Description |
| :----- | :---- | :---------- |
| `--batch-size`, `--wait` | `trufor.py` / `describe.py` | Images per batch and cool-down seconds between batches. |
| `--generate-truth` | `performance.py` | Write the `truth.csv` label template, then stop. |
| `--threshold` | `performance.py` | Score above which an image counts as manipulated (default `0.5`). |
| `--no-ocr`, `--do-sample` | `describe.py` | Caption-only mode / sampling instead of beam search. |
| `--skip-mask`, `--skip-describe`, `--skip-regression`, `--skip-classification` | `anomaly_regression.py` | Skip any of the stage-5 sub-steps. |
| `--max-words`, `--overperforming-cut` | `anomaly_regression.py` | TF-IDF vocabulary cap; binarise engagement at `>0` instead of the median (classifier). |

> ⚠️ **A note on filenames.** Media is matched to posts by reproducing `sha256(url)[:CHAR_LIMIT]`. `CHAR_LIMIT` (in `config.py`, default `64`) **must equal the truncation used at download time**, or every URL→file lookup silently misses.

---

## 🧪 Example

After stages 1–3 on a labelled subset, `performance.py` writes `scripts/temp/performance.txt` (values illustrative):

```
TruFor performance report
Total files analyzed: 1180
Threshold: 0.50

Accuracy  :	86.10%	(percentage of correct predictions)
Precision :	81.40%	(reliability when it claims 'It is AI')
Recall    :	74.20%	(sensitivity, ability to detect all AI images)
F1-score  :	0.7763	(final harmonic mean)

Confusion matrix:
True positives  :	214
True negatives  :	802
False positives :	49	(real mistaken for AI)
False negatives :	74	(AI mistaken for real)
```

Stage 5 then describes the altered region, so an `anomaly_txt` cell reads e.g. `a man in a suit giorgia meloni elezioni` — a BLIP caption (the subject) merged with the OCR words read off the banner.

---

## 🔬 How it works

A five-stage pipeline, orchestrated by [`run.sh`](scripts/run.sh); each stage appends a column to the analysis table (see [Architecture](#-architecture)):

1. **`filter.py`** — keeps the photo-only scored posts and symlinks their images into `dataset/images/` (TruFor's input dir).
2. **`trufor.py`** — runs TruFor's `test.py` over those images in batches. **Resumable** (one `.npz` per image; existing ones skipped); multi-megapixel images are downscaled, then their maps upsampled back.
3. **`performance.py`** — scores TruFor against the manual `truth.csv` and reports metrics + the false-positive/negative lists.
4. **`general_stats.py`** — Pearson/Spearman correlation, an OLS regression of `crowdtangle_score ~ trufor_score`, and the distribution of the engagement score (+ histogram/boxplot).
5. **`anomaly_regression.py`** — runs `mask.py` then `describe.py`, then models `anomaly_txt`: a TF-IDF representation (English + Italian stop-words removed), used both raw (one column per word) and PCA-compressed (enough components to keep 90% of the variance), fed to **(a)** an OLS regression on the continuous `crowdtangle_score` and **(b)** a 5-fold cross-validated logistic classifier of high/low engagement — plus a term-intelligibility check (top coefficients and the words that best separate the two engagement halves). *Finding on this dataset: manipulation **intensity** (`trufor_score`) doesn't predict engagement, but the manipulated-region **description** carries a weak-but-real signal (cross-validated AUC ≈ 0.63).*

---

## ✅ How the forensic verdicts are decided

The calls are governed by a handful of thresholds in [`config.py`](scripts/config.py):

- **Detection** — an image is *manipulated* when `trufor_score > PERFORMANCE_THRESHOLD` (default `0.5`).
- **Mask** — a pixel is kept only when **both** flagged (`anomaly ≥ MASK_THRESHOLD`) **and** confidently localized (`confidence ≥ MASK_CONFIDENCE_THRESHOLD`); the raw mask is then denoised.
- **Polarity** — on synthetic-majority composites (a real subject on a flat banner) TruFor's mask can invert; texture asymmetry detects this and flips it so the overlay is kept, not the subject.
- **Honest empties** — a near-black mask is *not* captioned (the model would hallucinate) and gets the fixed label `empty`.
- **OCR text** — captioning can't read, so OCR supplies the words: low-confidence and 1–2-letter recognitions are dropped, the rest de-duplicated and capped most-confident-first.

---

## 🧱 Architecture

The run revolves around one growing table, `scripts/temp/filtered_dataset.csv`:

| Column | Added by | Meaning |
| :----- | :------- | :------ |
| `id` | `filter.py` | Post identifier. |
| `crowdtangle_score` | `filter.py` | CrowdTangle overperforming score (the engagement signal). |
| `filename` | `filter.py` | Media file stem (the URL hash); the merge key for every stage. |
| `trufor_score` | `trufor.py` | TruFor global manipulation score in `[0, 1]`. |
| `anomaly_txt` | `describe.py` | Text description of the altered region (BLIP caption + OCR words). |

Configuration is the backbone: **[`config.py`](scripts/config.py)** (imported by every Python script) is mirrored by **[`config.sh`](scripts/config.sh)** (same paths plus the `ask_execute` / `ask_remove` helpers used by the orchestrators). Edit these, not the scripts, and keep the two in sync. Shared building blocks live in [`utils.py`](scripts/utils.py).

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