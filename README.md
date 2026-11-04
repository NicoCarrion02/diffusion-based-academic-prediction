# Prediction of Student Academic Performance Using Diffusion-Based Models

**Master's Thesis in Data Science**  
**Author:** Nicolas Carrion  
**Institution:** Universidad San Francisco de Quito (USFQ) — Master's in Data Science  
**Target Submission & Evaluation:** user `mtgca` (Private Repository)

---

## 📌 Abstract & Research Problem

Academic performance prediction on large-scale international educational assessments (such as the OECD Programme for International Student Assessment, **PISA 2022**) faces two structural challenges:
1. **High Dimensionality & Mixed Survey Modalities:** The raw PISA student questionnaire contains $1,278$ heterogeneous variables ($1,070$ analytical predictors after leakage removal), spanning continuous IRT socio-economic and attitudinal scales, discrete ordinal Likert items, binary indicators, and unordered nominal codes (`ISCO-08` parental occupations, national programme tracks).
2. **Class Imbalance at the Proficiency Threshold:** Following the OECD PISA mathematics framework, students scoring below Level 2 ($< 420.07$ points on `PV1MATH`) lack baseline mathematical proficiency. In the Argentina 2022 cohort ($N = 12,111$ students), **$66.90\%$** ($8,102$) fall below Level 2 (**Class 0: Low-Performing / At Academic Risk**), whereas only **$33.10\%$** ($4,009$) reach Level 2 or above (**Class 1: Proficient / High-Performing**)—an approximate $2:1$ class imbalance.

This repository implements an **idempotent, leakage-free tabular generative and machine learning pipeline** faithful to **Official `TabRep-Flow`** (Si et al., `arXiv:2504.04798`) to synthesize realistic minority-class (`Class 1`) student profiles in native survey space, balance training cohorts $1:1$, and evaluate **Random Forest**, **XGBoost**, and the tabular foundation model **`TabPFNv3`** under a **Nested Cross-Validation** protocol with **Nadeau–Bengio variance-corrected inference**, **Exact McNemar & stratified bootstrap tests**, and **SHAP (`TreeExplainer`)** attributions.

---

## 🏗️ Architecture & Directory Structure

```text
├── configs/
│   ├── config.yaml               # Data paths, country filter (ARG), Level 2 threshold (420.07), seeds
│   ├── features.yaml             # Intra-split feature selection (k=100), real-cohort target encoding
│   └── models.yaml               # Official TabRep-Flow, classifier search spaces, Nested CV & selection
├── data/                         # Strictly ignored by git (downloaded/generated idempotently)
│   ├── raw/                      # Raw OECD STU_QQQ_SAS.zip, CY08MSP_STU_QQQ.SAS7BDAT & FORMAT.SAS
│   ├── interim/                  # Compressed Parquet (pisa2022.parquet) & SAS metadata dictionary JSON
│   └── processed/                # Stratified 80/20 splits, TabRep balanced split, & Nested CV cache
├── Dockerfile                    # CUDA 12.1 runtime container for university GPU servers
├── docker-compose.yml            # One-command GPU execution with volume mounts & token passthrough
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip dependencies (PyTorch, scikit-learn, XGBoost, Optuna, TabPFN, SHAP)
├── .env.example                  # Template for optional TabPFNv3 authentication (TABPFN_TOKEN / HF_TOKEN)
├── notebooks/
│   └── workflow_review.ipynb     # Executed end-to-end walkthrough of all 8 stages with rendered figures
├── outputs/
│   ├── figures/                  # TabRep density audit, ROC/PR curves, Confusion Matrices, SHAP plots
│   └── tables/                   # Nested CV selection, tuned hyperparams JSON, benchmark & stat CSVs
├── src/
│   ├── data/
│   │   ├── downloader.py         # Idempotent OECD PISA 2022 downloader with integrity verification
│   │   ├── converter.py          # SAS7BDAT to compressed Parquet converter
│   │   ├── metadata_explorer.py  # Authoritative SAS PROC FORMAT parser & 1,278-variable taxonomy
│   │   └── loader.py             # Filter ARG, PV1MATH binarization, 208-column leakage purge, 80/20 split
│   ├── features/
│   │   ├── cleaner.py            # Format-declared missing-code sanitizer & schema categorizer
│   │   ├── encoders.py           # Categorical encoders (Target, Frequency, One-Hot, CatConverter PSK)
│   │   ├── selector.py           # Hybrid MI + ExtraTrees + Domain Anchors & label-free anchor modes
│   │   └── preprocessor.py       # Leakage-free NativeImputer & real-cohort FeatureEncoderScaler
│   ├── models/
│   │   ├── diffusion/
│   │   │   ├── tabrep_adapter.py # Official TabRep-Flow (NQT + 2D CatConverter PSK + 4-layer ReLU MLP)
│   │   │   ├── balancer.py       # Minority-class (Class 1) synthesis & 1:1 dataset balancer
│   │   │   └── diagnostics.py    # Distribution audit (KS, 1-Wasserstein, OOB, Correlation MAE, Copy rate)
│   │   └── classifiers/
│   │       ├── random_forest.py  # Random Forest builder & trainer
│   │       ├── baseline_comparisons.py # CPU-hist XGBoost builder & trainer
│   │       └── tabpfn_classifier.py    # TabPFNv3 wrapper + single-pass raw-logit calibration helpers
│   ├── evaluation/
│   │   ├── metrics.py            # ROC-AUC, PR-AUC, Balanced Acc, Specificity, F1, Recall, Brier
│   │   ├── plots.py              # Publication-ready ROC, PR, Confusion Matrix & Gini importance plots
│   │   ├── tuning.py             # Nested CV Optuna TPE tuner & manual/automatic champion selector
│   │   ├── statistical_tests.py  # Nadeau-Bengio CV t-test, Exact McNemar & Stratified Bootstrap CIs
│   │   └── explainability.py     # SHAP TreeExplainer beeswarm, global bar & contrastive waterfalls
│   └── pipeline.py               # Master idempotent orchestrator
├── tests/
│   └── test_workflow.py          # Cold-clone unit & integration test suite (13 tests via pytest)
└── run.py                        # Unified CLI entrypoint supporting auto & manual selection modes
```

---

## 🔬 Methodological Design & Fidelity

### 1. Official `TabRep-Flow` vs. Explicit PISA-2022 Adaptations
- **Original TabRep Components (Preserved Verbatim from Si et al., `arXiv:2504.04798`):**
  - Numerical representation via `sklearn.preprocessing.QuantileTransformer(output_distribution="normal")` and exact `.inverse_transform()`.
  - Categorical representation via **2D `CatConverter` Phase-Shift Keying (PSK)** $(\sin(2\pi k / K_j), \cos(2\pi k / K_j))$ and `atan2` angular decoding (`convert_psk` / `psk2cat`).
  - Velocity backbone **`TabRepFlowMLP`**: `SinusoidalTimeEmbedding` ($d_t = 1024$) + 4-layer feed-forward ReLU MLP `[1024, 2048, 2048, 1024]` (`dropout = 0.0`, no `LayerNorm`), trained under linear Conditional Flow Matching with `Adam` (`lr = 1e-3`, `weight_decay = 5e-4`), `ReduceLROnPlateau`, and 50-step second-order Heun ODE integration.
- **Explicit Dataset-Specific PISA Adaptations:**
  1. `snap_discrete_levels=True` (configurable toggle): Post-inversion nearest-level snapping for the $44$ discrete ordinal/binary questionnaire scales so generated responses remain valid integer Likert/indicator levels.
  2. `"MISSING"` nominal category token preserving structured item non-response on routing-skipped fields (`OCOD1–3`, `PROGN`).
  3. Minority-only (`Class 1`, $N_1 = 3,207$) training to synthesize $N_0 - N_1 = 3,274$ profiles ($1:1$ balance: $6,481$ vs. $6,481$).

### 2. Leakage-Free Nested Cross-Validation & Statistical Testing
- **Intra-Split Preprocessing & Real-Cohort Target Encoding:** Every inner and outer CV split independently fits `FeatureSelector` ($k = 100$) $\to$ `NativeImputer` $\to$ `TabRep-Flow` $\to$ **real-cohort `FeatureEncoderScaler`** (`fit(X_tr_real, y_tr_real)`), preventing synthetic `y = 1` prior shift ($0.331 \to 0.500$) and rare-category label leakage.
- **Nested Model Selection (`test.parquet` Strictly Vaulted):** Hyperparameters $\hat{\theta}_k$ are tuned via Optuna Multivariate TPE on inner splits inside $\mathcal{D}_{\text{outer\_tr}}^{(k)}$, while candidate ranking (`--mode auto` or `--mode manual`) uses untouched outer validation folds $\mathcal{D}_{\text{outer\_val}}^{(k)}$.
- **Valid Statistical Inference:** Correlated $K$-fold CV comparisons use the **Nadeau & Bengio (2003) variance-corrected resampled $t$-test** ($\text{SE}_{\text{NB}} = \sqrt{(1/J + n_{\text{val}}/n_{\text{tr}})s_d^2}$), while independent held-out test predictions ($N_{\text{test}} = 2,423$) are evaluated via **Exact McNemar binomial tests** and **paired stratified bootstrap $95\%$ CIs** ($2,000$ resamples).

---

## 🚀 Quickstart & Reproduction

All pipeline stages are **idempotent**: existing intermediate artifacts are reused automatically unless `--force` (or an explicit CLI selection override) is passed.

### Option 1: Docker on University GPU Server (Recommended)
```bash
git clone https://github.com/NicoCarrion02/diffusion-based-academic-prediction.git
cd diffusion-based-academic-prediction

# Optional: configure TABPFN_TOKEN or HF_TOKEN in .env for TabPFNv3 evaluation
cp .env.example .env

# Build and execute the full pipeline with GPU acceleration
docker compose up --build
```

### Option 2: Conda / Local Workstation
```bash
conda env create -f environment.yml
conda activate pisa-diffusion

# Verify cold-clone automated test suite (passes without raw SAS/Parquet caches)
pytest tests/ -v

# Execute complete end-to-end pipeline (Automatic Nested CV selection)
python run.py --step all
```

### 🎛️ Automatic vs. Manual Model Selection Modes
```bash
# Automatic mode (default): Nested CV tuning & ranking across candidates
python run.py --step benchmark --mode auto --metric f1_high_performing --force

# Manual mode: tune & evaluate a specific model and regime independently
python run.py --step benchmark --mode manual --model xgboost --regime both --metric f1_high_performing
python run.py --step benchmark --mode manual --model random_forest --regime balanced --n-trials 20
python run.py --step benchmark --mode manual --model tabpfn --regime both

# Fast baseline evaluation without inner hyperparameter search
python run.py --step benchmark --no-tune --force
```

### 🔑 Optional: `TabPFNv3` Foundation Model Activation (`.env`)
The pipeline and notebook run out-of-the-box with **Random Forest** and **XGBoost**. To also evaluate **`TabPFNv3`** (`TabPFNV3p5`):
1. Register a free academic account at https://ux.priorlabs.ai and copy your token from https://ux.priorlabs.ai/account (or authorize gated weights via `HF_TOKEN`).
2. Copy `.env.example` to `.env` and set `TABPFN_TOKEN=...` (or `HF_TOKEN=...`).
3. If neither token is set, the pipeline logs a clear warning and executes Random Forest + XGBoost cleanly without failing.

---

## 📊 Experimental Results ($k = 100$)

### 1. Independent Vaulted Test Set Benchmark ($N_{\text{test}} = 2,423$)
Evaluated once on the held-out Argentina cohort ($1,621$ Class 0 [`66.90%`], $802$ Class 1 [`33.10%`]) using real-cohort target encoding and Nested CV-selected hyperparameters:

| Model & Training Regime | ROC-AUC | PR-AUC | Balanced Acc | Specificity | High-Perf F1 | High-Perf Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Unbalanced)** | `0.8433` | `0.7580` | `74.24%` | **89.76%** | `0.6546` | `58.73%` | `0.1465` |
| **Random Forest (TabRep Balanced)** | `0.8342` | `0.7397` | `74.91%` | `82.60%` | `0.6642` | **67.21%** | `0.1546` |
| **XGBoost (Unbalanced)** *(Tree CV Champion)* | `0.8678` | `0.7932` | `76.84%` | `88.34%` | `0.6917` | `65.34%` | `0.1348` |
| **XGBoost (TabRep Balanced)** | `0.8652` | `0.7832` | `76.63%` | `86.18%` | `0.6880` | `67.08%` | `0.1383` |
| **TabPFNv3 (Unbalanced)** | **0.8733** | **0.7936** | `77.12%` | `89.02%` | `0.6959` | `65.21%` | **0.1324** |
| **TabPFNv3 (TabRep Balanced)** | `0.8712` | `0.7919` | **77.67%** | `89.51%` | **0.7040** | `65.84%` | `0.1330` |

### 2. 5-Fold Nested Cross-Validation Inference (Nadeau–Bengio Corrected, `train.parquet` Only)
Evaluated strictly on untouched outer validation folds ($n_{\text{val}} / n_{\text{tr}} = 1,937 / 7,751$, accounting for $75\%$ training-set overlap):

| Model | Metric | Mean Unbalanced | Mean Balanced | Mean Diff | 95% NB CI | Folds Improved | $t_{\text{NB}}$ | $p_{\text{NB}}$ | Naive $p$ *(Ref)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Recall (High-Perf) | `57.84%` | `66.79%` | `+8.95%` | `[+4.35%, +13.55%]` | `5/5` | `5.41` | **0.0057** | `0.0013` |
| **Random Forest** | F1 (High-Perf) | `0.6570` | `0.6754` | `+0.0185` | `[-0.0170, +0.0539]` | `4/5` | `1.44` | `0.2220` | `0.0961` |
| **Random Forest** | Balanced Accuracy | `74.41%` | `75.73%` | `+1.31%` | `[-1.19%, +3.81%]` | `4/5` | `1.46` | `0.2190` | `0.0943` |
| **XGBoost** | Recall (High-Perf) | `64.26%` | `66.07%` | `+1.81%` | `[-0.31%, +3.93%]` | `5/5` | `2.37` | `0.0770` | `0.0238` |
| **XGBoost** | F1 (High-Perf) | `0.6911` | `0.6872` | `-0.0038` | `[-0.0185, +0.0109]` | `2/5` | `-0.72` | `0.5112` | `0.3408` |
| **XGBoost** | Balanced Accuracy | `76.76%` | `76.56%` | `-0.21%` | `[-1.27%, +0.86%]` | `2/5` | `-0.54` | `0.6179` | `0.4635` |

### 3. Held-Out Test Inference (Paired Stratified Bootstrap & Exact McNemar, $N_{\text{test}} = 2,423$)

| Model | Metric | Unbalanced | TabRep Balanced | Difference | 95% Bootstrap CI | Exact McNemar $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Recall (High-Perf) | `58.73%` | `67.21%` | **`+8.48%`** | `[+6.23%, +10.85%]` | **$1.40 \times 10^{-12}$** *(Class 1)* |
| **Random Forest** | F1 (High-Perf) | `0.6546` | `0.6642` | `+0.0096` | `[-0.0085, +0.0292]` | `0.0022` *(Full test)* |
| **XGBoost** | Recall (High-Perf) | `65.34%` | `67.08%` | `+1.75%` | `[-0.37%, +3.74%]` | `0.1143` *(Class 1)* |
| **TabPFNv3** | F1 (High-Perf) | `0.6959` | **`0.7040`** | `+0.0081` | `[-0.0065, +0.0226]` | `0.2546` *(Full test)* |
| **TabPFNv3** | Balanced Accuracy | `77.12%` | **`77.67%`** | `+0.56%` | `[-0.47%, +1.62%]` | `0.2546` *(Full test)* |

### Key Empirical Findings
1. **Substantial Minority Recall Recovery in Bagging (`Random Forest`):** Official `TabRep-Flow` $1:1$ balancing boosts minority proficiency recall by **`+8.48 pp` on the vaulted test set** ($p_{\text{McNemar}} = 1.40 \times 10^{-12}$, rescuing $68$ net proficient students from false-negative misclassification) and **`+8.95 pp` across 5-fold Nested CV** ($p_{\text{NB}} = 0.0057$), while raising both High-Performing F1 (`0.6546 -> 0.6642`) and Balanced Accuracy (`74.24% -> 74.91%`).
2. **Synergy with Tabular Foundation Models (`TabPFNv3`):** Supplying the `1:1` `TabRep-Flow` balanced training cohort as in-context conditioning to `TabPFNv3` yields the **strongest overall F1 (`0.7040`), Balanced Accuracy (`77.67%`), and Accuracy (`81.68%`)** in the study, simultaneously improving minority recall (`65.21% -> 65.84%`) and specificity (`89.02% -> 89.51%`).
3. **Methodological Coherence & Variance Correction (`XGBoost`):** Eliminating feature-selection and target-encoding leakage aligns Nested CV (`0.6911` vs. `0.6872`) with the vaulted test set (`0.6917` vs. `0.6880`) to within `0.0008`, while Nadeau–Bengio correction prevents false-positive claims on `XGBoost` recall ($p_{\text{naive}} = 0.0238$ vs. $p_{\text{NB}} = 0.0770$).
4. **Generative Fidelity & Privacy:** Official `TabRep-Flow` achieves **$0.00\%$ out-of-bounds violations**, **$0.0\%$ exact memorization**, pairwise correlation $\text{MAE} = 0.0404$, and mean 1-Wasserstein distance $= 0.3249$.
5. **SHAP Tree Interpretability:** Exact `TreeExplainer` attributions highlight maternal/paternal `ISCO-08` occupation (`OCOD1`, `OCOD2`), weekly instructional periods (`ST059Q02JA` — a direct institutional policy lever), mathematics self-efficacy (`MATHEFF`), household digital/book resources (`ST253Q01JA`, `ST255Q01JA`), and growth mindset (`ST263Q02JA`).

---

## 🔒 Thesis Submission & Privacy Compliance

- **Repository Privacy:** Must remain **Private** in accordance with graduate committee policy.
- **Evaluator Access:** Shared with evaluator GitHub user **`mtgca`** (*Settings → Collaborators → Add people → `mtgca`*).
- **Zero-Data / Zero-Checkpoint Git Policy:** `.gitignore` strictly excludes all `.sas7bdat`, `.zip`, `.parquet`, `.json` metadata caches, `.env` secrets, and `.pkl`/`.pt` weights.
