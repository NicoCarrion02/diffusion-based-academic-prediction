# Prediction of Student Academic Performance Using Diffusion-Based Models

**Master's Thesis in Data Science**  
**Author:** Nicolas Carrion  
**Institution:** Universidad San Francisco de Quito (USFQ) — Master's in Data Science  
**Target Submission & Evaluation:** user `mtgca` (Private Repository)

---

## Problem Statement & Research Context

Academic performance prediction on large-scale international educational assessments (such as the OECD Programme for International Student Assessment, **PISA 2022**) faces two structural challenges:
1. **High Dimensionality & Mixed Survey Modalities:** The raw PISA student questionnaire contains $1,278$ heterogeneous variables ($1,070$ analytical predictors after removing a 208-column leakage purge comprising secondary cognitive plausible values `PV2MATH`–`PV10MATH` and reading/science scores, student sampling weights `W_FSTUWT` and 80 replicate weights `W_FSTR1`–`80`, and administrative tracking identifiers such as `CNTSCHID` and `CNTSTUID`). Predictors span continuous Item Response Theory (IRT) socio-economic and attitudinal scales, discrete ordinal Likert items, binary indicators, and unordered nominal codes (`ISCO-08` parental occupations, national programme tracks).
2. **Class Imbalance at the Proficiency Threshold:** Following the OECD PISA mathematics framework, students scoring below Level 2 ($< 420.07$ points on Plausible Value 1 in Mathematics, `PV1MATH`) lack baseline mathematical proficiency. In the Argentina 2022 cohort ($N = 12,111$ students), **$66.90\%$** ($8,102$) fall below Level 2 (**Class 0: Low-Performing / At Academic Risk**), whereas only **$33.10\%$** ($4,009$) reach Level 2 or above (**Class 1: Proficient / High-Performing**)—an approximate $2:1$ class imbalance.

This repository implements an **idempotent, leakage-free tabular generative and machine learning pipeline** faithful to **`TabRep-Flow`** ([Si et al., arXiv:2504.04798](https://arxiv.org/abs/2504.04798)) to synthesize minority-class (`Class 1`) student profiles in native survey space, balance training cohorts $1:1$, and evaluate **Random Forest**, **XGBoost**, and the tabular foundation model **`TabPFNv3`** under a **Nested Cross-Validation** protocol with **Nadeau–Bengio variance-corrected inference**, **Exact McNemar & stratified bootstrap tests**, and **SHAP (`TreeExplainer`)** attributions.

---

## Goals & Non-Goals

### Goals
- **Empirical Generative Adaptation:** Implement and evaluate `TabRep-Flow` ([Si et al., arXiv:2504.04798](https://arxiv.org/abs/2504.04798)) on complex mixed-type educational survey microdata (OECD PISA 2022 Argentina cohort).
- **Rigorous 1:1 Balancing Evaluation:** Quantify minority-class proficiency recall and precision-recall trade-offs across Random Forest, XGBoost, and the tabular foundation model `TabPFNv3`.
- **Leakage-Free Statistical Inference:** Evaluate models under strict Nested Cross-Validation with Nadeau–Bengio variance corrections, paired Exact McNemar tests, stratified bootstrap confidence intervals, and SHAP interpretability.

### Non-Goals
- **Novel Generative Architecture Development:** The objective is a faithful empirical adaptation and boundary analysis of Si et al., not the invention of a new diffusion model family.
- **Cross-National Generalization:** Models and synthetic cohorts are evaluated specifically on the Argentina cohort ($N = 12,111$); cross-country transfer learning is left to future research.
- **Low-Latency Edge Deployment:** The pipeline is designed for reproducible offline research, statistical inference, and committee evaluation, not real-time production serving.
- **Threshold Optimization as an Oversampling Substitute:** Post-hoc decision threshold shifting is analyzed as an operating point comparison, not as a replacement for generative manifold modeling.

---

## Proposed Architecture & Directory Structure

### Pipeline Data Flow Architecture
```text
OECD Raw SAS ──> converter.py ──> Interim Parquet ──> loader.py ──> Stratified 80/20 Splits
                                                                          │
       ┌──────────────────────────────────────────────────────────────────┘
       ▼
train.parquet (Nested CV Splits)
       │
       ├──> FeatureSelector (k=100) ──> NativeImputer
       │                                   │
       │                                   ├──> TabRep-Flow (Minority Synthesis) ──> Balanced Split
       │                                   │                                              │
       │                                   └──> Real-Cohort FeatureEncoderScaler ─────────┤
       │                                                                                  ▼
       └──> Nested Tuning & Model Selection (RF, XGBoost, TabPFNv3) <─────────────────────┘
                                                        │
                                                        ▼ Champion Refit
                                              Held-Out test.parquet
                                                        │
                                                        ▼
                                      Statistical Inference & SHAP Analysis
```

### Directory Tree
```text
├── configs/
│   ├── config.yaml               # Data paths, country filter (ARG), Level 2 threshold (420.07), seeds
│   ├── features.yaml             # Intra-split feature selection (k=100), real-cohort target encoding
│   └── models.yaml               # TabRep-Flow hyperparameters, classifier search spaces, Nested CV & selection
├── data/                         # Ignored by git (downloaded/generated idempotently)
│   ├── raw/                      # Raw OECD STU_QQQ_SAS.zip, CY08MSP_STU_QQQ.SAS7BDAT & FORMAT.SAS
│   ├── interim/                  # Compressed Parquet (pisa2022.parquet) & SAS metadata dictionary JSON
│   └── processed/                # Stratified 80/20 splits, TabRep balanced split, & Nested CV cache
├── Dockerfile                    # CUDA 12.1 runtime container for university GPU servers
├── docker-compose.yml            # GPU container execution with volume mounts & token passthrough
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
│   │   ├── metadata_explorer.py  # SAS PROC FORMAT parser & 1,278-variable taxonomy
│   │   └── loader.py             # Filter ARG, PV1MATH binarization, 208-column leakage purge, 80/20 split
│   ├── features/
│   │   ├── cleaner.py            # Format-declared missing-code sanitizer & schema categorizer
│   │   ├── encoders.py           # Categorical encoders (Real-cohort Empirical Bayes TargetEncoder)
│   │   ├── selector.py           # Supervised FeatureSelector (Hybrid MI + ExtraTrees + Domain Anchors)
│   │   └── preprocessor.py       # Survey-native imputer & real-cohort FeatureEncoderScaler
│   ├── models/
│   │   ├── diffusion/
│   │   │   ├── tabrep_adapter.py # TabRep-Flow velocity backbone & converters (NQT + 2D PSK + 4-layer MLP)
│   │   │   ├── balancer.py       # Minority-class (Class 1) synthesis & 1:1 dataset balancer
│   │   │   └── diagnostics.py    # Disentangled numerical KS/Wasserstein & nominal TVD distribution audit
│   │   └── classifiers/
│   │       ├── random_forest.py  # Random Forest builder & trainer
│   │       ├── baseline_comparisons.py # CPU-hist XGBoost builder & trainer
│   │       └── tabpfn_classifier.py    # TabPFNv3 wrapper + single-pass raw-logit calibration helpers
│   ├── evaluation/
│   │   ├── metrics.py            # ROC-AUC, PR-AUC, Balanced Acc, Specificity, F1, Recall, Brier
│   │   ├── plots.py              # 6-curve ROC, PR, Confusion Matrix & Gini plotting utilities
│   │   ├── tuning.py             # Nested CV Optuna TPE tuner & manual/automatic champion selector
│   │   ├── statistical_tests.py  # Nadeau-Bengio CV t-test, Exact McNemar & Stratified Bootstrap CIs
│   │   └── explainability.py     # SHAP TreeExplainer beeswarm, global bar & contrastive waterfalls
│   └── pipeline.py               # Master pipeline orchestrator
├── tests/
│   └── test_workflow.py          # Cold-clone unit & integration test suite (14 tests via pytest)
└── run.py                        # Unified CLI entrypoint supporting auto & manual selection modes
```

---

## Methodological Design & Fidelity

### 1. TabRep-Flow Architecture vs. Explicit PISA-2022 Adaptations
- **TabRep-Flow Architectural Components (Implemented per [Si et al., arXiv:2504.04798](https://arxiv.org/abs/2504.04798)):**
  - Numerical representation via `sklearn.preprocessing.QuantileTransformer(output_distribution="normal")` and exact inverse quantile transformation.
  - Categorical representation via **2D `CatConverter` Phase-Shift Keying (PSK)** $(\sin(2\pi k / K_j), \cos(2\pi k / K_j))$ and `atan2` angular sector decoding.
  - Velocity backbone **`TabRepFlowMLP`**: `SinusoidalTimeEmbedding` ($d_t = 1024$) + 4-layer feed-forward ReLU MLP `[1024, 2048, 2048, 1024]` (`dropout = 0.0`, no `LayerNorm`), trained under linear Conditional Flow Matching with `Adam` (`lr = 1e-3`, `weight_decay = 5e-4`), `ReduceLROnPlateau`, and 50-step second-order Heun ODE integration.
- **Explicit Dataset-Specific PISA Adaptations:**
  1. `snap_discrete_levels=True` (configurable toggle): Post-inversion nearest-level snapping for the $44$ discrete ordinal/binary questionnaire scales so generated responses remain valid integer Likert/indicator levels.
  2. `"MISSING"` nominal category token preserving structured item non-response on routing-skipped fields (`OCOD1–3`, `PROGN`).
  3. Minority-only (`Class 1`, $N_1 = 3,207$) training to synthesize $N_0 - N_1 = 3,274$ profiles ($1:1$ balance: $6,481$ vs. $6,481$).

### 2. Leakage-Free Nested Cross-Validation & Statistical Testing
- **Intra-Split Preprocessing & Real-Cohort Target Encoding:** Every inner and outer CV split independently fits `FeatureSelector` ($k = 100$) $\to$ `NativeImputer` $\to$ `TabRep-Flow` $\to$ **real-cohort `FeatureEncoderScaler`** (`fit(X_tr_real, y_tr_real)`), preserving empirical class priors ($P(Y=1) = 0.331$) and preventing synthetic category-frequency distortion.
- **Nested Model Selection (`test.parquet` Held-Out & Isolated):** Hyperparameters $\hat{\theta}_k$ are tuned via Optuna Multivariate TPE on inner splits inside $\mathcal{D}_{\text{outer\_tr}}^{(k)}$, while candidate ranking (`--mode auto` or `--mode manual`) uses untouched outer validation folds $\mathcal{D}_{\text{outer\_val}}^{(k)}$.
- **Variance-Corrected Statistical Inference Protocol:** Correlated $K$-fold CV comparisons use the **Nadeau & Bengio (2003) variance-corrected resampled $t$-test** ($\text{SE}_{\text{NB}} = \sqrt{(1/J + n_{\text{val}}/n_{\text{tr}})s_d^2}$), while independent held-out test predictions ($N_{\text{test}} = 2,423$) are evaluated via **Exact McNemar binomial tests** (paired strictly with the binary correctness metrics they test: Minority Recall on the Class 1 subset and Overall Accuracy on the full test set) and **paired stratified bootstrap $95\%$ CIs** ($2,000$ resamples).

---

## Quickstart & Reproduction

All pipeline stages are **idempotent**: existing intermediate artifacts are reused automatically unless `--force` (or an explicit CLI selection override) is passed.

### Pipeline Stages & CLI Execution Mapping
| CLI Command Option | Pipeline Stage & Target Module | Description & Output Artifacts |
| :--- | :--- | :--- |
| `python run.py --step download` | Stage 1 (`src/data/downloader.py`) | Idempotent download of OECD PISA 2022 `STU_QQQ_SAS.zip` with SHA-256 verification. |
| `python run.py --step convert` | Stage 2 (`src/data/converter.py`) | Converts SAS7BDAT to compressed Parquet (`data/interim/pisa2022.parquet`). |
| `python run.py --step metadata` | Stage 3 (`src/data/metadata_explorer.py`) | Parses `FORMAT.SAS` to build the 1,278-variable survey metadata dictionary JSON. |
| `python run.py --step prepare` | Stage 4 (`src/data/loader.py`) | Filters ARG, binarizes `PV1MATH` at Level 2 (420.07), purges 208 leakage columns, and creates stratified 80/20 train/test splits. |
| `python run.py --step select` | Stage 5 (`src/features/selector.py`) | Fits intra-split feature selector ($k=100$) combining domain anchors with Mutual Information & ExtraTrees scores. |
| `python run.py --step diffusion` | Stage 6 (`src/models/diffusion/balancer.py`) | Fits `TabRep-Flow` on minority class ($N_1 = 3,207$) to generate synthetic profiles and create the 1:1 balanced training set. |
| `python run.py --step benchmark` | Stages 7–8 (`src/pipeline.py`) | Executes 5-fold Nested CV model tuning, refits champions on 100% training data, evaluates on held-out test cohort, and produces statistical test tables and SHAP plots. |
| `python run.py --step all` | Stages 1–8 (`src/pipeline.py`) | Runs complete end-to-end pipeline idempotently from raw OECD download to final test benchmark and figures. |

### System Requirements & Compute Footprint
- **Disk Storage:** $\approx 4.2\text{ GB}$ for the raw OECD SAS archive (`STU_QQQ_SAS.zip`); $\approx 500\text{ MB}$ for interim compressed Parquet files, metadata JSON dictionaries, and processed fold caches.
- **Hardware Requirements:** NVIDIA GPU with $\ge 6\text{ GB}$ VRAM recommended (tested on CUDA 12.1 / RTX 2060 Max-Q); CPU fallback supported. $\ge 16\text{ GB}$ system RAM recommended for TabPFNv3 in-context matrix evaluation.
- **Execution Times (GPU):** Diffusion balancing: $\approx 49\text{s}$ full training ($\approx 41\text{s}$ per CV outer fold). Full end-to-end execution (`python run.py --step all`): $\approx 15\text{--}20\text{ minutes}$.

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

### Automatic vs. Manual Model Selection Modes
```bash
# Automatic mode (default): Nested CV tuning & ranking across candidates
python run.py --step benchmark --mode auto --metric f1_high_performing --force

# Manual mode: tune & evaluate a specific model and regime independently
python run.py --step benchmark --mode manual --model xgboost --regime both --metric f1_high_performing
python run.py --step benchmark --mode manual --model random_forest --regime balanced --n-trials 20
python run.py --step benchmark --mode manual --model tabpfn --regime both

# Baseline evaluation without inner hyperparameter search
python run.py --step benchmark --no-tune --force
```

### Optional: `TabPFNv3` Foundation Model Activation (`.env`)
The pipeline and notebook execute by default using **Random Forest** and **XGBoost**. To also evaluate **`TabPFNv3`** (`TabPFNV3p5`):
1. Register a free academic account at the [PriorLabs Portal](https://ux.priorlabs.ai) and copy your token from [Account Settings](https://ux.priorlabs.ai/account) (or authorize gated weights via `HF_TOKEN`).
2. Copy `.env.example` to `.env` and set `TABPFN_TOKEN=...` (or `HF_TOKEN=...`).
3. If neither token is set, the pipeline logs a warning and evaluates Random Forest and XGBoost.

---

## Experimental Results ($k = 100$)

### 1. Independent Held-Out Test Set Benchmark ($N_{\text{test}} = 2,423$)
Evaluated once on the held-out Argentina cohort ($1,621$ Class 0 [`66.90%`], $802$ Class 1 [`33.10%`]) using real-cohort target encoding and Nested CV-selected hyperparameters:

| Model & Training Regime | ROC-AUC | PR-AUC | Accuracy | Balanced Acc | Specificity | High-Perf F1 | High-Perf Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Unbalanced)** | `0.8433` | `0.7580` | `79.49%` | `74.24%` | **89.76%** | `0.6546` | `58.73%` | `0.1465` |
| **Random Forest (TabRep Balanced)** | `0.8342` | `0.7397` | `77.51%` | `74.91%` | `82.60%` | `0.6642` | **67.21%** | `0.1546` |
| **XGBoost (Unbalanced)** *(Tree CV Optimal)* | `0.8678` | `0.7932` | `80.73%` | `76.84%` | `88.34%` | `0.6917` | `65.34%` | `0.1348` |
| **XGBoost (TabRep Balanced)** | `0.8652` | `0.7832` | `79.86%` | `76.63%` | `86.18%` | `0.6880` | `67.08%` | `0.1383` |
| **TabPFNv3 (Unbalanced)** | **0.8733** | **0.7936** | `81.14%` | `77.12%` | `89.02%` | `0.6959` | `65.21%` | **0.1324** |
| **TabPFNv3 (TabRep Balanced)** | `0.8712` | `0.7919` | **81.68%** | **77.67%** | `89.51%` | **0.7040** | `65.84%` | `0.1330` |

### 2. 5-Fold Nested Cross-Validation Inference (Nadeau–Bengio Corrected, `train.parquet` Only)
Evaluated strictly on held-out outer validation folds ($n_{\text{val}} / n_{\text{tr}} = 1,937 / 7,751$, accounting for $75\%$ training-set overlap):

| Model | Metric | Mean Unbalanced | Mean Balanced | Mean Diff | 95% NB CI | Folds Improved | $t_{\text{NB}}$ | $p_{\text{NB}}$ | Naive $p$ *(Ref)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Recall (High-Perf) | `57.84%` | `66.79%` | `+8.95%` | `[+4.35%, +13.55%]` | `5/5` | `5.41` | **0.0057** | `0.0013` |
| **Random Forest** | F1 (High-Perf) | `0.6570` | `0.6754` | `+0.0185` | `[-0.0170, +0.0539]` | `4/5` | `1.44` | `0.2220` | `0.0961` |
| **Random Forest** | Balanced Accuracy | `74.41%` | `75.73%` | `+1.31%` | `[-1.19%, +3.81%]` | `4/5` | `1.46` | `0.2190` | `0.0943` |
| **XGBoost** | Recall (High-Perf) | `64.26%` | `66.07%` | `+1.81%` | `[-0.31%, +3.93%]` | `5/5` | `2.37` | `0.0770` | `0.0238` |
| **XGBoost** | F1 (High-Perf) | `0.6911` | `0.6872` | `-0.0038` | `[-0.0185, +0.0109]` | `2/5` | `-0.72` | `0.5112` | `0.3408` |
| **XGBoost** | Balanced Accuracy | `76.76%` | `76.56%` | `-0.21%` | `[-1.27%, +0.86%]` | `2/5` | `-0.54` | `0.6179` | `0.4635` |

### 3. Held-Out Test Inference (Paired Stratified Bootstrap & Exact McNemar, $N_{\text{test}} = 2,423$)
Exact McNemar's test is a paired binomial test on binary correctness vectors and is therefore paired strictly with the binary correctness metric it evaluates (**Minority Recall** on the $N_1 = 802$ Class 1 subset and **Overall Accuracy** on the full $N = 2,423$ test cohort), while composite and threshold-free metrics (`F1`, `Balanced Accuracy`, `PR-AUC`, `ROC-AUC`) are evaluated via paired stratified bootstrap confidence intervals and two-sided bootstrap $p$-values:

| Model | Metric | Unbalanced | TabRep Balanced | Difference | 95% Bootstrap CI | Bootstrap $p$ | Exact McNemar $p$ (Discordant $b$ vs. $c$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Recall (High-Perf) | `58.73%` | `67.21%` | **`+8.48%`** | `[+6.23%, +10.85%]` | `< 0.0001` | **$1.40 \times 10^{-12}$** ($b=83, c=15$, Class 1) |
| **Random Forest** | F1 (High-Perf) | `0.6546` | `0.6642` | `+0.0096` | `[-0.0085, +0.0292]` | `0.3150` | — *(Composite metric)* |
| **Random Forest** | Overall Accuracy | `79.49%` | `77.51%` | `-1.98%` | `[-3.14%, -0.82%]` | `0.0010` | **`0.0022`** ($b=94, c=142$, Full test) |
| **XGBoost** | Recall (High-Perf) | `65.34%` | `67.08%` | `+1.75%` | `[-0.37%, +3.74%]` | `0.1100` | `0.1143` ($b=41, c=27$, Class 1) |
| **XGBoost** | Overall Accuracy | `80.73%` | `79.86%` | `-0.87%` | `[-1.90%, +0.12%]` | `0.1070` | `0.1147` ($b=70, c=91$, Full test) |
| **TabPFNv3** | F1 (High-Perf) | `0.6959` | `0.7040` | `+0.0081` | `[-0.0065, +0.0226]` | `0.2390` | — *(Composite metric)* |
| **TabPFNv3** | Balanced Accuracy | `77.12%` | `77.67%` | `+0.56%` | `[-0.47%, +1.62%]` | `0.2550` | — *(Composite metric)* |
| **TabPFNv3** | Overall Accuracy | `81.14%` | `81.68%` | `+0.54%` | `[-0.29%, +1.40%]` | `0.2250` | `0.2546` ($b=62, c=49$, Full test) |

### 4. Key Empirical Findings & Methodological Discussion
1. **Minority Recall vs. Specificity Trade-Off in Bagging (`Random Forest`):** `TabRep-Flow` $1:1$ balancing increases minority proficiency recall by **`+8.48 pp` on the held-out test set** (Class-1 Exact McNemar $p = 1.40 \times 10^{-12}$, reclassifying $83 - 15 = 68$ net proficient students from false negatives to true positives) and **`+8.95 pp` across 5-fold Nested CV** ($p_{\text{NB}} = 0.0057$). Because training on a $1:1$ balanced prior ($P_{\text{train}}(Y=1) = 0.500$) while evaluating at a fixed decision threshold ($\tau = 0.50$) shifts predicted posterior probabilities upward, majority-class specificity decreases from `89.76%` to `82.60%` (`-7.16 pp`), which reduces **Overall Accuracy** on the $2:1$ imbalanced test cohort from `79.49%` to `77.51%` (Full-Test Exact McNemar $p = 0.0022$, $b=94$ vs. $c=142$). On class-balanced criteria, point estimates show small positive shifts (`Balanced Accuracy`: `74.24% -> 74.91%`; `High-Perf F1`: `0.6546 -> 0.6642`, $p_{\text{boot}} = 0.3150$).
2. **In-Context Conditioning with `TabPFNv3`:** Supplying the `1:1` `TabRep-Flow` balanced training cohort to `TabPFNv3` achieves the highest point estimates in the benchmark (`F1 = 0.7040`, `Balanced Accuracy = 77.67%`, `Overall Accuracy = 81.68%`), yielding small increases in recall (`65.21% -> 65.84%`) and specificity (`89.02% -> 89.51%`). However, paired stratified bootstrap and Exact McNemar inference confirm that these gains over `TabPFNv3 (Unbalanced)` (`F1 = 0.6959`, $\Delta F_1 = +0.0081$, $95\%$ CI `[-0.0065, +0.0226]`, $p_{\text{boot}} = 0.2390$, Full-Test McNemar $p = 0.2546$) are **not statistically significant at $\alpha = 0.05$**. On this benchmark, pre-trained in-context conditioning with TabPFNv3 already handles moderate ($2:1$) class imbalance effectively without requiring synthetic oversampling.
3. **Gradient Boosting Calibration & Variance-Corrected CV Inference (`XGBoost`):** Across both 5-fold Nested CV (`F1 = 0.6911` Unbalanced vs. `0.6872` Balanced) and the held-out test set (`F1 = 0.6917` vs. `0.6880`), histogram-based `XGBoost` exhibits a `+1.75 pp` to `+1.81 pp` gain in minority recall accompanied by a `-0.0038` change in F1 and a `-0.99 pp` drop in PR-AUC (`0.7932 -> 0.7832`). Furthermore, Nadeau–Bengio variance correction demonstrates that the apparent 5-fold CV recall improvement ($p_{\text{naive}} = 0.0238$) does not reach significance once the $75\%$ inter-fold training overlap is accounted for ($p_{\text{NB}} = 0.0770$, consistent with held-out Class-1 McNemar $p = 0.1143$).
4. **Disentangled Generative Fidelity & High-Cardinality 2D PSK Bottleneck:**
   - **Numerical Manifold Fidelity ($85$ Continuous & Ordinal/Binary Features):** `TabRep-Flow` achieves a mean two-sample Kolmogorov–Smirnov statistic of **`0.1307`** (`median = 0.1198`), mean 1-Wasserstein distance of **`0.3275`**, pairwise Pearson correlation $\text{MAE} = \mathbf{0.0404}$, and **`0.0%` exact memorization** in standardized feature space (mean standardized nearest-neighbor $L_2$ distance $= 10.4759$). The **`0.00%` out-of-bounds violation rate** on continuous features is structurally guaranteed by `QuantileTransformer.inverse_transform()`, which clips latent Gaussian draws outside $[\Phi^{-1}(0), \Phi^{-1}(1)]$ onto the observed training minimum and maximum, while `snap_discrete_levels=True` projects the $44$ discrete Likert/binary variables onto valid survey codes.
   - **Nominal Category Fidelity ($15$ Unordered Categorical Features) & Angular Crowding:** Evaluated separately via Total Variation Distance ($\text{TVD} = \frac{1}{2}\sum_{c} |P_{\text{real}}(c) - P_{\text{synth}}(c)|$), the $15$ nominal features yield a mean TVD of **`0.3103`** (`median = 0.1939`). While low-cardinality nominal items ($K \le 7$, such as `ST327Q08JA` and `ST327Q07JA`) achieve low TVD (`0.112–0.146`), high-cardinality administrative and `ISCO-08` occupational codes exhibit substantial distributional divergence: `OCOD3` ($\text{TVD} = 0.6544$), `OCOD1` ($0.5865$, $K > 400$), `OCOD2` ($0.5462$, $K > 400$), `LANGN` ($0.5332$), and `PROGN` ($0.5273$). Under `TabRep`'s 2D unit-circle Phase-Shift Keying $(\sin(2\pi k / K), \cos(2\pi k / K))$, adjacent categories are separated by $\Delta\theta = 2\pi / K$ radians; when $K > 200\text{--}400$, $\Delta\theta < 0.016\text{ rad}$ ($< 0.9^\circ$), so small ODE integration errors during 50-step Heun sampling scatter trajectories into neighboring `atan2` angular sectors. Because `OCOD1` and `OCOD2` are the `#1` and `#2` global SHAP predictors in `XGBoost`, this high-cardinality PSK dispersion is a primary candidate mechanism for why synthetic balancing slightly degrades `XGBoost` PR-AUC (`-0.0099`), whereas `Random Forest` (`max_features="log2"` / `"sqrt"`) averages across randomized feature subsets and captures the `+8.48 pp` minority recall gain.
5. **SHAP Tree Interpretability:** Exact `TreeExplainer` attributions highlight maternal/paternal `ISCO-08` occupation (`OCOD1`, `OCOD2`), weekly instructional periods (`ST059Q02JA`), mathematics self-efficacy (`MATHEFF`), household digital/book resources (`ST253Q01JA`, `ST255Q01JA`), and growth mindset (`ST263Q02JA`).

---

## System Limitations, Risks & Mitigations

| System Limitation / Risk | Underlying Mechanism & Impact | Architectural Mitigation & Guidance |
| :--- | :--- | :--- |
| **2D PSK Angular Crowding on High-Cardinality Nominals** | Packing $K > 400$ categories on the 2D unit circle leaves $\Delta\theta < 0.9^\circ$. Small ODE trajectory dispersion during Heun sampling scatters categories into neighboring `atan2` sectors ($\text{TVD} \approx 0.55\text{--}0.65$), introducing noise on top SHAP predictors (`OCOD1`, `OCOD2`). | **Documented Boundary Invariant:** Reported as an empirical finding regarding Si et al.'s representation. Downstream Random Forest mitigates noise via feature subsampling (`max_features="sqrt"`). Hierarchical 2-digit ISCO grouping ($K \approx 43$) and learned entity embeddings are scoped for Future Work. |
| **Prior Shift vs. Threshold Sensitivity** | Training on a 1:1 synthetic balanced prior ($P_{\text{train}}(Y=1) = 0.500$) shifts predicted posterior probabilities upward. Evaluating at fixed $\tau = 0.50$ increases minority recall (+8.48 pp in Random Forest) but lowers majority specificity (-7.16 pp) and overall accuracy (-1.98 pp). | **Multi-Metric Evaluation:** Accuracy is never evaluated in isolation; balanced accuracy, high-performing F1, and Exact McNemar paired tests are reported alongside ROC/PR curves to expose trade-offs. |
| **Compute Footprint & Scaling Limits** | Training a 4-layer flow velocity MLP for 500–600 epochs across 5 CV folds with Optuna hyperparameter optimization is computationally intensive (~327,400 neural passes per fold). | **Idempotent Caching:** Precomputed fold splits, feature selection matrices, and balanced synthetic cohorts are cached in `data/processed/cv_cache/`, eliminating redundant diffusion training across inner Optuna trials. |
| **Data Privacy & Educational Sensitive Microdata** | PISA student records contain sensitive socio-economic and attitudinal data. | **Zero-Data Git Policy:** Raw `.sas7bdat`, Parquet splits, and model weights are git-ignored; synthetic cohorts are generated strictly within local user environments. |

---

## Thesis Submission & Privacy Compliance

- **Repository Privacy:** Must remain **Private** in accordance with graduate committee policy.
- **Evaluator Access:** Shared with evaluator GitHub user **`mtgca`** (*Settings → Collaborators → Add people → `mtgca`*).
- **Zero-Data / Zero-Checkpoint Git Policy:** `.gitignore` excludes all `.sas7bdat`, `.zip`, `.parquet`, `.json` metadata caches, `.env` secrets, and `.pkl`/`.pt` weights.
