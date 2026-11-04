"""
Comprehensive test suite validating end-to-end functionality of all workflow components,
including cold-clone execution without local SAS/Parquet caches, Official TabRep-Flow
fidelity, leakage-free intra-fold feature selection & real-cohort target encoding,
Nested CV manual/automatic model selection, and Nadeau-Bengio / McNemar statistical inference.

Run with:
    pytest tests/ -v
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import torch

from run import build_arg_parser
from src.data.metadata_explorer import PISAMetadataExplorer
from src.evaluation.explainability import compute_shap_explanations
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.statistical_tests import (
    compare_models_cv,
    compute_holdout_statistical_tests,
    nadeau_bengio_ttest,
)
from src.evaluation.tuning import (
    NestedFoldCacheManager,
    NestedModelTunerSelector,
    resolve_metric_name,
)
from src.features.cleaner import categorize_pisa_features, clean_pisa_survey_codes
from src.features.encoders import (
    CatConverterEncoder,
    FrequencyEncoder,
    build_categorical_encoder,
)
from src.features.preprocessor import (
    FeatureEncoderScaler,
    NativeImputer,
    TabularPreprocessor,
)
from src.features.selector import FeatureSelector
from src.models.classifiers.baseline_comparisons import train_xgboost
from src.models.classifiers.random_forest import train_random_forest
from src.models.classifiers.tabpfn_classifier import (
    has_tabpfn_auth_token,
    tabpfn_logits_to_probs,
)
from src.models.diffusion.tabrep_adapter import (
    TabRepContinuousEncoder,
    TabRepDiffusionModel,
    TabRepFlowMLP,
    convert_psk,
    psk2cat,
)


def _build_fallback_metadata_dict() -> Dict[str, Dict[str, Any]]:
    """Minimal authoritative OECD schema dictionary for cold-clone unit testing without raw SAS files."""
    return {
        "ESCS": {
            "name": "ESCS",
            "label": "Index of economic, social and cultural status",
            "variable_type": "continuous_scale",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "ESCSF",
            "categories": {},
            "missing_codes": {"99": "Missing"},
        },
        "HOMEPOS": {
            "name": "HOMEPOS",
            "label": "Home possessions",
            "variable_type": "continuous_scale",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "HOMEPOSF",
            "categories": {},
            "missing_codes": {"99": "Missing"},
        },
        "HISEI": {
            "name": "HISEI",
            "label": "Highest parental occupational status",
            "variable_type": "continuous_scale",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "HISEIF",
            "categories": {},
            "missing_codes": {"999": "Missing"},
        },
        "MATHEFF": {
            "name": "MATHEFF",
            "label": "Mathematics self-efficacy",
            "variable_type": "continuous_scale",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "MATHEFFF",
            "categories": {},
            "missing_codes": {"99": "Missing"},
        },
        "ANXMAT": {
            "name": "ANXMAT",
            "label": "Mathematics anxiety",
            "variable_type": "continuous_scale",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "ANXMATF",
            "categories": {},
            "missing_codes": {"99": "Missing"},
        },
        "ST004D01T": {
            "name": "ST004D01T",
            "label": "Student Standardized Gender",
            "variable_type": "binary",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "ST004F",
            "categories": {"1": "Female", "2": "Male"},
            "missing_codes": {"9": "Missing"},
        },
        "REPEAT": {
            "name": "REPEAT",
            "label": "Grade repetition",
            "variable_type": "binary",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": "REPEATF",
            "categories": {"0": "Did not repeat", "1": "Repeated"},
            "missing_codes": {"9": "Missing"},
        },
        "PROGN": {
            "name": "PROGN",
            "label": "Unique national study programme code",
            "variable_type": "nominal",
            "storage_type": "string",
            "storage_width": 8,
            "format_name": "$PROGNF",
            "categories": {"1": "Prog 1", "2": "Prog 2", "3": "Prog 3"},
            "missing_codes": {},
        },
        "LANGN": {
            "name": "LANGN",
            "label": "Language spoken at home",
            "variable_type": "nominal",
            "storage_type": "string",
            "storage_width": 8,
            "format_name": "$LANGNF",
            "categories": {"111": "Spanish", "313": "Other"},
            "missing_codes": {},
        },
        "STRATUM": {
            "name": "STRATUM",
            "label": "Stratum ID",
            "variable_type": "identifier_admin",
            "storage_type": "string",
            "storage_width": 8,
            "format_name": "$STRATUMF",
            "categories": {},
            "missing_codes": {},
        },
        "PV1MATH": {
            "name": "PV1MATH",
            "label": "Plausible Value 1 in Mathematics",
            "variable_type": "plausible_value",
            "storage_type": "double",
            "storage_width": 8,
            "format_name": None,
            "categories": {},
            "missing_codes": {},
        },
    }


@pytest.fixture(scope="session")
def metadata_explorer():
    explorer = PISAMetadataExplorer()
    if os.path.exists(explorer.cache_path):
        explorer.build_dictionary()
    else:
        explorer.dictionary = _build_fallback_metadata_dict()
    return explorer


@pytest.fixture
def synthetic_pisa_data():
    """Generates synthetic student batch mimicking PISA features with realistic signal."""
    rng = np.random.RandomState(42)
    n = 150
    escs = rng.normal(0, 1, n)
    matheff = rng.normal(0, 1, n)
    anxmat = rng.normal(0, 1, n)
    logits = 0.9 * escs + 1.1 * matheff - 0.6 * anxmat - 0.7
    probs = 1.0 / (1.0 + np.exp(-logits))
    target = (rng.uniform(0, 1, n) < probs).astype(int)

    df = pd.DataFrame({
        "ESCS": escs,
        "HOMEPOS": rng.normal(0.2, 0.9, n),
        "HISEI": rng.uniform(15, 85, n),
        "MATHEFF": matheff,
        "ANXMAT": anxmat,
        "ST004D01T": rng.choice([1.0, 2.0], size=n),
        "REPEAT": rng.choice([0.0, 1.0], size=n, p=[0.8, 0.2]),
        "PROGN": rng.choice(["1", "2", "3", "MISSING"], size=n),
        "LANGN": rng.choice(["111", "313", "MISSING"], size=n),
        "target": target,
    })
    return df


def test_metadata_explorer(metadata_explorer):
    assert len(metadata_explorer.dictionary) >= 11
    cats = metadata_explorer.get_schema_categorization(
        ["ESCS", "ST004D01T", "PROGN", "STRATUM", "PV1MATH", "UNKNOWN_NONEXIST"]
    )
    assert "ESCS" in cats["continuous_cols"]
    assert "PROGN" in cats["nominal_cols"]
    assert "STRATUM" in cats["other_cols"]
    assert "PV1MATH" in cats["other_cols"]
    assert "UNKNOWN_NONEXIST" in cats["other_cols"]


def test_cold_clone_without_cached_metadata(synthetic_pisa_data, tmp_path):
    """Simulates a fresh git clone where neither raw SAS files nor JSON metadata caches exist."""
    cold_explorer = PISAMetadataExplorer(
        sas_data_path=str(tmp_path / "missing.sas7bdat"),
        sas_format_path=str(tmp_path / "missing.sas"),
        cache_path=str(tmp_path / "missing.json"),
    )
    cold_explorer.dictionary = _build_fallback_metadata_dict()

    cleaned = clean_pisa_survey_codes(synthetic_pisa_data, cold_explorer)
    reg = categorize_pisa_features(cleaned.drop(columns=["target"]), cold_explorer)
    assert set(reg.nominal_cols) == {"PROGN", "LANGN"}
    assert set(reg.ordinal_discrete_cols) == {"ST004D01T", "REPEAT"}


def test_cleaner_and_categorization(synthetic_pisa_data, metadata_explorer):
    cleaned = clean_pisa_survey_codes(synthetic_pisa_data, metadata_explorer)
    assert cleaned.shape == synthetic_pisa_data.shape
    reg = categorize_pisa_features(cleaned.drop(columns=["target"]), metadata_explorer)
    assert len(reg.nominal_cols) >= 1
    assert len(reg.ordinal_discrete_cols) >= 1
    assert len(reg.continuous_cols) >= 1

    invalid_df = synthetic_pisa_data.copy()
    invalid_df["STRATUM"] = "S1"
    with pytest.raises(ValueError, match="unmapped or non-analytical"):
        clean_pisa_survey_codes(invalid_df, metadata_explorer)
    with pytest.raises(ValueError, match="unmapped or non-analytical"):
        categorize_pisa_features(invalid_df.drop(columns=["target"]), metadata_explorer)


def test_feature_selector(synthetic_pisa_data, metadata_explorer):
    X = synthetic_pisa_data.drop(columns=["target"]).copy()
    y = synthetic_pisa_data["target"]

    # 1. Supervised hybrid importance mode
    selector = FeatureSelector(target_k=5, max_missing_ratio=0.50, variance_threshold=0.001, method="hybrid_importance")
    selector.fit(X, y, metadata_explorer)
    assert len(selector.selected_features_) == 5
    X_trans = selector.transform(X)
    assert X_trans.shape == (len(X), 5)

    # 2. Label-free domain_anchors mode (verifying zero dependence on target labels y)
    sel_a = FeatureSelector(target_k=5, method="domain_anchors", random_state=42)
    sel_b = FeatureSelector(target_k=5, method="domain_anchors", random_state=42)
    sel_a.fit(X, y, metadata_explorer)
    sel_b.fit(X, 1 - y, metadata_explorer)
    assert sel_a.selected_features_ == sel_b.selected_features_

    X_invalid = X.copy()
    X_invalid["UNMAPPED_COL"] = 1.0
    with pytest.raises(ValueError, match="unmapped or non-analytical"):
        selector.fit(X_invalid, y, metadata_explorer)


def test_preprocessor(synthetic_pisa_data, metadata_explorer):
    X = synthetic_pisa_data.drop(columns=["target"]).copy()
    y = synthetic_pisa_data["target"]
    reg = categorize_pisa_features(X, metadata_explorer)

    prep_dto = TabularPreprocessor(
        categorical_strategy="target_encoder",
        categorization=reg,
        scaler_type="robust",
        imputation_strategy="median",
    )
    X_proc_dto = prep_dto.fit_transform(X, y)
    assert X_proc_dto.isnull().sum().sum() == 0
    assert X_proc_dto.shape[0] == len(X)
    assert list(X_proc_dto.columns) == list(prep_dto.transform(X).columns)

    cat_cols = ["PROGN", "LANGN"]
    num_cols = ["ESCS", "HOMEPOS", "HISEI", "MATHEFF", "ANXMAT", "ST004D01T", "REPEAT"]
    prep_exp = TabularPreprocessor(
        categorical_strategy="target_encoder",
        categorical_cols=cat_cols,
        numerical_cols=num_cols,
        scaler_type="robust",
        imputation_strategy="median",
    )
    X_proc_exp = prep_exp.fit_transform(X, y)
    assert X_proc_exp.shape[0] == len(X)

    prep_missing_cols = TabularPreprocessor(
        categorical_strategy="target_encoder",
        categorical_cols=cat_cols,
        numerical_cols=["ESCS"],
    )
    with pytest.raises(ValueError, match="Unassigned columns found"):
        prep_missing_cols.fit(X, y)

    prep_empty = TabularPreprocessor()
    with pytest.raises(ValueError, match="requires explicit column lists"):
        prep_empty.fit(X, y)


def test_native_imputer_and_real_cohort_target_encoding(synthetic_pisa_data, metadata_explorer):
    X = synthetic_pisa_data.drop(columns=["target"]).copy()
    y = synthetic_pisa_data["target"].copy()
    reg = categorize_pisa_features(X, metadata_explorer)

    X_missing = X.copy()
    X_missing.loc[0:5, "ESCS"] = np.nan
    X_missing.loc[3:8, "PROGN"] = np.nan

    imputer = NativeImputer(categorization=reg)
    X_imp = imputer.fit_transform(X_missing)
    assert X_imp["ESCS"].isnull().sum() == 0
    assert X_imp["PROGN"].isnull().sum() == 0
    assert (X_imp.loc[3:8, "PROGN"] == "MISSING").all()

    # Verify real-cohort TargetEncoder avoids synthetic y=1 prior shift on unseen categories
    encoder_real = FeatureEncoderScaler(
        categorical_strategy="target_encoder",
        categorization=reg,
        scaler_type="robust",
    )
    encoder_real.fit(X_imp, y)

    X_unseen = X_imp.iloc[:5].copy()
    X_unseen["PROGN"] = "UNSEEN_RARE_CODE"
    encoded_real = encoder_real.transform(X_unseen)
    assert abs(encoded_real["PROGN"].iloc[0] - float(y.mean())) < 1e-4


def test_official_tabrep_flow_fidelity_and_adaptations(synthetic_pisa_data, metadata_explorer):
    features = synthetic_pisa_data.drop(columns=["target"]).copy()
    reg = categorize_pisa_features(features, metadata_explorer)

    # 1. Verify exact CatConverter PSK round-trip identity ([sin_phases, cos_phases] layout)
    cats = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.long)
    psk = convert_psk(cats, [3, 4])
    assert psk.shape == (3, 4)
    assert torch.allclose(psk[:, 0] ** 2 + psk[:, 2] ** 2, torch.ones(3), atol=1e-5)
    assert torch.allclose(psk[:, 1] ** 2 + psk[:, 3] ** 2, torch.ones(3), atol=1e-5)
    recovered_cats = psk2cat(psk, [3, 4])
    assert torch.equal(cats, recovered_cats)

    # 2. Verify Official TabRepFlowMLP architecture (PositionalEmbedding + 4 Linear/ReLU blocks, no LayerNorm)
    net = TabRepFlowMLP(input_dim=11, dim_t=64, hidden_dims=[64, 128, 128, 64], dropout=0.0, activation="relu")
    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in net.modules())
    assert not has_layernorm
    out = net(torch.randn(8, 11), torch.rand(8))
    assert out.shape == (8, 11)

    # 3. Verify unseen nominal token fallback to "MISSING" and discrete snapping toggle
    enc_snap = TabRepContinuousEncoder(random_state=42, snap_discrete_levels=True)
    enc_snap.fit(features, categorization=reg)
    f_unseen = features.copy()
    f_unseen.loc[0, "PROGN"] = "BRAND_NEW_CATEGORY"
    num_norm, cat_idx = enc_snap.transform(f_unseen)
    assert num_norm.shape[0] == len(features)

    # Perturb discrete latent coordinates and verify snap_discrete_levels=True vs False
    noisy_norm = num_norm + 0.15
    rec_snapped = enc_snap.inverse_transform(noisy_norm, cat_idx)
    assert set(rec_snapped["REPEAT"].unique()).issubset({0.0, 1.0})

    enc_unsnapped = TabRepContinuousEncoder(random_state=42, snap_discrete_levels=False)
    enc_unsnapped.fit(features, categorization=reg)
    rec_unsnapped = enc_unsnapped.inverse_transform(noisy_norm, cat_idx)
    assert enc_unsnapped.is_fitted and rec_unsnapped.shape == features.shape

    # 4. End-to-end mini TabRep-Flow synthesis
    model = TabRepDiffusionModel(
        num_timesteps=5,
        hidden_dims=[32, 64, 64, 32],
        dim_t=32,
        device="cpu",
        random_state=42,
    )
    model.fit(features, categorization=reg, epochs=6, batch_size=32)
    synth = model.sample(num_samples=20, seed=42)
    assert synth.shape == (20, features.shape[1])
    assert synth.isnull().sum().sum() == 0

    # 5. Must raise error when columns are unassigned
    encoder_bad = TabRepContinuousEncoder(random_state=42)
    with pytest.raises(ValueError, match="Unassigned columns found"):
        encoder_bad.fit(features, cat_cols=reg.nominal_cols, num_cols=["ESCS"])


def test_classifiers_and_metrics(synthetic_pisa_data):
    X = synthetic_pisa_data[["ESCS", "HOMEPOS", "HISEI", "MATHEFF"]]
    y = synthetic_pisa_data["target"]

    rf, y_pred, y_prob, imp = train_random_forest(X, y, X, n_estimators=10, max_depth=3)
    assert len(y_pred) == len(y)
    m = evaluate_predictions(y.values, y_pred, y_prob, model_name="TestRF")
    assert "roc_auc" in m and "specificity" in m

    xgb, xgb_pred, xgb_prob = train_xgboost(X, y, X, n_estimators=10, max_depth=3)
    if xgb is not None:
        assert len(xgb_pred) == len(y)


def test_shap_explanations(synthetic_pisa_data):
    X = synthetic_pisa_data[["ESCS", "HOMEPOS", "HISEI", "MATHEFF"]]
    y = synthetic_pisa_data["target"]
    rf, _, _, _ = train_random_forest(X, y, X, n_estimators=10, max_depth=3)
    explainer, shap_values = compute_shap_explanations(rf, X)
    assert shap_values.shape == X.shape


def test_statistical_tests_nadeau_bengio_and_holdout(synthetic_pisa_data):
    # 1. Verify Nadeau-Bengio variance correction widens SE relative to naive paired t-test
    unbal = np.array([0.61, 0.62, 0.60, 0.63, 0.61])
    bal = np.array([0.65, 0.66, 0.64, 0.66, 0.65])
    nb = nadeau_bengio_ttest(bal, unbal, n_train=7750, n_val=1938)
    naive_se = np.std(bal - unbal, ddof=1) / np.sqrt(5)
    assert nb["se_nadeau_bengio"] > naive_se
    assert nb["p_value_nadeau_bengio"] >= nb["p_value_ttest"]

    # 2. Verify held-out Exact McNemar & Stratified Bootstrap CIs
    rng = np.random.RandomState(7)
    y_true = np.array([0] * 60 + [1] * 30)
    probs_u = np.clip(0.30 + 0.35 * y_true + rng.normal(0, 0.1, 90), 0.01, 0.99)
    probs_b = np.clip(0.38 + 0.42 * y_true + rng.normal(0, 0.1, 90), 0.01, 0.99)
    ht_df = compute_holdout_statistical_tests(
        y_true=y_true,
        probs_unbal=probs_u,
        preds_unbal=(probs_u >= 0.5).astype(int),
        probs_bal=probs_b,
        preds_bal=(probs_b >= 0.5).astype(int),
        model_name="XGBoost",
        n_bootstrap=200,
        random_state=42,
        output_path=None,
    )
    assert len(ht_df) == 5
    assert {"bootstrap_ci95_low", "bootstrap_ci95_high", "mcnemar_exact_pvalue"}.issubset(ht_df.columns)

    # 3. Verify intra-fold FeatureSelector inside compare_models_cv
    X = synthetic_pisa_data[["ESCS", "HOMEPOS", "HISEI", "MATHEFF", "ANXMAT"]]
    y = synthetic_pisa_data["target"]
    stat_df = compare_models_cv(
        X,
        y,
        n_splits=3,
        model_type="random_forest",
        clf_params={"n_estimators": 8, "max_depth": 3},
        balancer_strategy="random_oversample",
        apply_feature_selection=True,
        target_k=3,
        output_path=None,
    )
    assert len(stat_df) == 4
    assert "p_value_nadeau_bengio" in stat_df.columns
    assert "p_value_ttest" in stat_df.columns


def test_nested_cv_manual_and_auto_modes(synthetic_pisa_data, metadata_explorer, tmp_path):
    cleaned = clean_pisa_survey_codes(synthetic_pisa_data, metadata_explorer)
    X_train = cleaned.drop(columns=["target"]).iloc[:120].reset_index(drop=True)
    y_train = cleaned["target"].iloc[:120].reset_index(drop=True)

    cache_mgr = NestedFoldCacheManager(
        target_k=5,
        max_missing_ratio=0.50,
        variance_threshold=0.001,
        selection_method="hybrid_importance",
        apply_feature_selection=True,
        balancer_strategy="random_oversample",
        metadata_explorer=metadata_explorer,
        cache_dir=str(tmp_path / "cv_cache"),
    )

    # 1. Manual Mode: explicitly tune only Random Forest with 2 Optuna TPE trials
    manual_tuner = NestedModelTunerSelector(
        mode="manual",
        selected_models=["random_forest"],
        regime="both",
        primary_metric="f1",
        strategy="tpe",
        outer_cv_folds=2,
        inner_cv_folds=1,
        n_trials_override=2,
        random_state=42,
    )
    manual_res = manual_tuner.run_nested_selection(
        X_clean=X_train,
        y=y_train,
        cache_manager=cache_mgr,
        output_tables_dir=str(tmp_path / "manual_tables"),
    )
    assert set(manual_res["selection_df"]["model_key"].unique()) == {"random_forest"}
    assert len(manual_res["selection_df"]) == 2
    assert (tmp_path / "manual_tables" / "best_hyperparameters.json").exists()

    # 2. Automatic Mode: compare Random Forest & XGBoost across both regimes and rank on outer folds
    auto_tuner = NestedModelTunerSelector(
        mode="auto",
        regime="both",
        primary_metric="f1_high_performing",
        strategy="tpe",
        outer_cv_folds=2,
        inner_cv_folds=1,
        n_trials_override=2,
        random_state=42,
    )
    auto_res = auto_tuner.run_nested_selection(
        X_clean=X_train,
        y=y_train,
        cache_manager=cache_mgr,
        output_tables_dir=str(tmp_path / "auto_tables"),
    )
    sel_df = auto_res["selection_df"]
    assert len(sel_df) == 4  # {random_forest, xgboost} x {unbalanced, balanced}
    assert int(sel_df["is_selected_champion"].sum()) == 1
    assert sel_df.iloc[0]["mean_f1_high_performing"] >= sel_df.iloc[-1]["mean_f1_high_performing"]
    assert not auto_res["statistical_df"].empty

    with open(tmp_path / "auto_tables" / "best_hyperparameters.json", "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert "champion" in saved_json and "candidates" in saved_json


def test_tabpfn_logits_helper_and_cli_validation(monkeypatch):
    # Verify placeholder tokens are rejected
    monkeypatch.setenv("TABPFN_TOKEN", "your_tabpfn_api_key_here")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not has_tabpfn_auth_token()

    monkeypatch.setenv("HF_TOKEN", "hf_valid_test_token_123")
    assert has_tabpfn_auth_token()

    # Verify raw-logit calibration helper (shape: [n_estimators=4, n_samples=6, n_classes=2])
    raw_logits = np.array(
        [[[0.8, -0.2], [-0.4, 0.9], [0.1, 0.1], [-0.9, 1.2], [0.5, -0.5], [-0.1, 0.3]]] * 4,
        dtype=np.float32,
    )
    probs_t09 = tabpfn_logits_to_probs(None, raw_logits, n_estimators=2, softmax_temperature=0.9)
    probs_t11 = tabpfn_logits_to_probs(None, raw_logits, n_estimators=4, softmax_temperature=1.1, average_before_softmax=True)
    assert probs_t09.shape == (6,) and probs_t11.shape == (6,)
    assert np.all((probs_t09 > 0.0) & (probs_t09 < 1.0))

    # Verify CLI argument validation
    parser = build_arg_parser()
    args = parser.parse_args(["--step", "benchmark", "--mode", "manual", "--model", "xgboost", "--metric", "pr_auc"])
    assert args.mode == "manual" and args.model == ["xgboost"]
    assert resolve_metric_name(args.metric) == "pr_auc"


def test_categorical_encoders_strict(synthetic_pisa_data):
    df_cat = synthetic_pisa_data[["PROGN", "LANGN"]].copy()

    with pytest.raises(ValueError, match="requires an explicit list of columns"):
        FrequencyEncoder().fit(df_cat)

    with pytest.raises(ValueError, match="requires an explicit list of columns"):
        CatConverterEncoder().fit(df_cat)

    freq_enc = FrequencyEncoder(cols=["PROGN", "LANGN"]).fit(df_cat)
    df_freq = freq_enc.transform(df_cat)
    assert df_freq.shape == df_cat.shape

    cat_conv = CatConverterEncoder(cols=["PROGN", "LANGN"]).fit(df_cat)
    df_conv = cat_conv.transform(df_cat)
    assert "PROGN_cos" in df_conv.columns and "PROGN_sin" in df_conv.columns

    with pytest.raises(ValueError, match="Unknown categorical encoding strategy"):
        build_categorical_encoder("unsupported_strategy")
