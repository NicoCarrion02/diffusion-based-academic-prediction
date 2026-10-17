"""
Comprehensive test suite validating end-to-end functionality of all workflow components.
Run with:
    pytest tests/ -v
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.metadata_explorer import PISAMetadataExplorer
from src.features.cleaner import clean_pisa_survey_codes, categorize_pisa_features
from src.features.selector import FeatureSelector
from src.features.preprocessor import TabularPreprocessor
from src.features.encoders import FrequencyEncoder, CatConverterEncoder, build_categorical_encoder
from src.models.diffusion.tabrep_adapter import TabRepContinuousEncoder, TabRepDiffusionModel
from src.models.classifiers.random_forest import train_random_forest
from src.models.classifiers.baseline_comparisons import train_xgboost
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.statistical_tests import compare_models_cv
from src.evaluation.explainability import compute_shap_explanations


@pytest.fixture(scope="session")
def metadata_explorer():
    explorer = PISAMetadataExplorer()
    if os.path.exists(explorer.cache_path):
        explorer.build_dictionary()
    return explorer


@pytest.fixture
def synthetic_pisa_data():
    """Generates synthetic student batch mimicking PISA features."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "ESCS": np.random.normal(0, 1, n),
        "HOMEPOS": np.random.normal(0.2, 0.9, n),
        "HISEI": np.random.uniform(15, 85, n),
        "MATHEFF": np.random.normal(0, 1, n),
        "ANXMAT": np.random.normal(0, 1, n),
        "ST004D01T": np.random.choice([1.0, 2.0], size=n),
        "REPEAT": np.random.choice([0.0, 1.0], size=n, p=[0.8, 0.2]),
        "PROGN": np.random.choice(["1", "2", "3", "MISSING"], size=n),
        "LANGN": np.random.choice(["111", "313", "MISSING"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.67, 0.33]),
    })
    return df


def test_metadata_explorer(metadata_explorer):
    if metadata_explorer.dictionary:
        assert len(metadata_explorer.dictionary) >= 1000
        cats = metadata_explorer.get_schema_categorization(["ESCS", "ST004D01T", "PROGN", "STRATUM", "PV1MATH", "UNKNOWN_NONEXIST"])
        assert "ESCS" in cats["continuous_cols"]
        assert "PROGN" in cats["nominal_cols"]
        assert "STRATUM" in cats["other_cols"]
        assert "PV1MATH" in cats["other_cols"]
        assert "UNKNOWN_NONEXIST" in cats["other_cols"]


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
    selector = FeatureSelector(target_k=5, max_missing_ratio=0.50, variance_threshold=0.001)
    selector.fit(X, y, metadata_explorer)
    assert len(selector.selected_features_) == 5
    X_trans = selector.transform(X)
    assert X_trans.shape == (len(X), 5)

    X_invalid = X.copy()
    X_invalid["UNMAPPED_COL"] = 1.0
    with pytest.raises(ValueError, match="unmapped or non-analytical"):
        selector.fit(X_invalid, y, metadata_explorer)


def test_preprocessor(synthetic_pisa_data, metadata_explorer):
    X = synthetic_pisa_data.drop(columns=["target"]).copy()
    y = synthetic_pisa_data["target"]
    reg = categorize_pisa_features(X, metadata_explorer)

    # 1. Using FeatureCategorization DTO
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

    # 2. Using explicit column lists
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

    # 3. Must raise error when columns are unassigned
    prep_missing_cols = TabularPreprocessor(
        categorical_strategy="target_encoder",
        categorical_cols=cat_cols,
        numerical_cols=["ESCS"],
    )
    with pytest.raises(ValueError, match="Unassigned columns found"):
        prep_missing_cols.fit(X, y)

    # 4. Must raise error when neither categorization nor column lists provided
    prep_empty = TabularPreprocessor()
    with pytest.raises(ValueError, match="requires explicit column lists"):
        prep_empty.fit(X, y)


def test_tabrep_encoder_and_model(synthetic_pisa_data, metadata_explorer):
    features = synthetic_pisa_data.drop(columns=["target"])
    reg = categorize_pisa_features(features, metadata_explorer)

    # 1. Using categorization DTO
    encoder = TabRepContinuousEncoder(random_state=42)
    encoder.fit(features, categorization=reg)
    assert encoder.is_fitted
    num_norm, cat_idx = encoder.transform(features)
    assert num_norm.shape[0] == len(features)
    recovered = encoder.inverse_transform(num_norm, cat_idx)
    assert recovered.shape == features.shape

    # 2. Using explicit column lists
    encoder_exp = TabRepContinuousEncoder(random_state=42)
    encoder_exp.fit(
        features,
        cat_cols=reg.nominal_cols,
        num_cols=reg.numerical_cols,
        discrete_cols=reg.ordinal_discrete_cols,
    )
    assert encoder_exp.is_fitted

    # 3. Must raise error when columns are unassigned
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


def test_statistical_tests_cv(synthetic_pisa_data):
    X = synthetic_pisa_data[["ESCS", "HOMEPOS", "HISEI", "MATHEFF"]]
    y = synthetic_pisa_data["target"]
    stat_df = compare_models_cv(X, y, n_splits=3, model_type="random_forest", clf_params={"n_estimators": 10, "max_depth": 3}, output_path=None)
    assert len(stat_df) == 4
    assert "t_statistic" in stat_df.columns
    assert "p_value_ttest" in stat_df.columns


def test_categorical_encoders_strict(synthetic_pisa_data):
    df_cat = synthetic_pisa_data[["PROGN", "LANGN"]].copy()

    # Must raise when cols is not provided
    with pytest.raises(ValueError, match="requires an explicit list of columns"):
        FrequencyEncoder().fit(df_cat)

    with pytest.raises(ValueError, match="requires an explicit list of columns"):
        CatConverterEncoder().fit(df_cat)

    # Valid execution when cols is explicitly provided
    freq_enc = FrequencyEncoder(cols=["PROGN", "LANGN"]).fit(df_cat)
    df_freq = freq_enc.transform(df_cat)
    assert df_freq.shape == df_cat.shape

    cat_conv = CatConverterEncoder(cols=["PROGN", "LANGN"]).fit(df_cat)
    df_conv = cat_conv.transform(df_cat)
    assert "PROGN_cos" in df_conv.columns and "PROGN_sin" in df_conv.columns

    # Unknown strategy must raise ValueError
    with pytest.raises(ValueError, match="Unknown categorical encoding strategy"):
        build_categorical_encoder("unsupported_strategy")
