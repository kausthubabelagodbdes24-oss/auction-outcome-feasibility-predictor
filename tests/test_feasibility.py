"""Lightweight tests for artifact compatibility, input validation, novelty, and SVM prediction."""

import copy
import json
import sys
import zipfile
from pathlib import Path

import pytest
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feasibility import (
    CLASS_LABELS,
    DEFAULT_BUNDLE_PATH,
    EXPECTED_FEATURES,
    NOVEL_MESSAGE,
    REQUIRED_MEMBERS,
    SEEN_MESSAGE,
    ArtifactError,
    InvalidConfigurationError,
    configuration_key,
    load_bundle,
    load_training_keys,
    predict_outcome,
    seen_during_training,
    validate_configuration,
    validate_metadata,
)


@pytest.fixture(scope="module")
def artifact():
    return load_bundle(DEFAULT_BUNDLE_PATH)


@pytest.fixture(scope="module")
def training_keys(artifact):
    return load_training_keys(artifact.metadata)


def _config_from_key(key: str) -> dict[str, int]:
    return dict(zip(EXPECTED_FEATURES, (int(v) for v in key.split("|"))))


def test_real_artifact_compatibility(artifact, training_keys):
    with zipfile.ZipFile(DEFAULT_BUNDLE_PATH) as zf:
        for member in REQUIRED_MEMBERS:
            assert member in zf.namelist()
    assert artifact.model_sha256 == artifact.metadata["model_artifact"]["sha256"]
    assert sklearn.__version__ == artifact.metadata["sklearn_version"] == "1.6.1"
    assert artifact.metadata["feature_names"] == EXPECTED_FEATURES
    assert list(artifact.model.feature_names_in_) == EXPECTED_FEATURES
    assert len(training_keys) == artifact.metadata["split"]["train_records"] == 1634


def test_artifact_error_cases(tmp_path: Path, artifact, monkeypatch):
    with pytest.raises(ArtifactError, match="not found"):
        load_bundle(tmp_path / "missing.zip")

    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not a valid ZIP"):
        load_bundle(bad_zip)

    incomplete = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete, "w") as zf:
        zf.writestr("model_metadata.json", "{}")
    with pytest.raises(ArtifactError, match="missing required files"):
        load_bundle(incomplete)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(DEFAULT_BUNDLE_PATH) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in REQUIRED_MEMBERS:
            data = src.read(name)
            if name == "auction_model.joblib":
                data = data + b"0"
            dst.writestr(name, data)
    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        load_bundle(tampered)

    bad_feature_meta = copy.deepcopy(artifact.metadata)
    bad_feature_meta["feature_names"] = list(reversed(EXPECTED_FEATURES))
    with pytest.raises(ArtifactError, match="Incompatible feature contract"):
        validate_metadata(bad_feature_meta)

    monkeypatch.setattr(sklearn, "__version__", "9.9.9")
    with pytest.raises(ArtifactError, match="scikit-learn 1.6.1"):
        load_bundle(DEFAULT_BUNDLE_PATH)


def test_valid_prediction_returns_single_binary_estimate(artifact):
    config = _config_from_key("0|3|1|0|60|4|0")
    frame = validate_configuration(config, artifact.metadata)
    assert list(frame.columns) == EXPECTED_FEATURES
    assert frame.shape == (1, 7)

    pred = predict_outcome(artifact.model, frame)
    assert pred in (0, 1)
    assert CLASS_LABELS[pred] in ("Feasible model estimate", "Infeasible model estimate")
    assert predict_outcome(artifact.model, frame) == pred


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d.update({"property.price": 999}), "out of domain"),
        (lambda d: d.update({"process.b1.capacity": -1}), "out of domain"),
        (lambda d: d.update({"property.product": 0}), "out of domain"),
        (lambda d: d.update({"property.winner": 5}), "out of domain"),
        (lambda d: d.update({"property.price": True}), "must be an integer"),
        (lambda d: d.update({"property.price": "70"}), "must be an integer"),
        (lambda d: d.pop("property.winner"), "Missing"),
        (lambda d: d.update({"verification.time": 1.0}), "Unexpected"),
    ],
)
def test_invalid_inputs_rejected_before_prediction(artifact, mutator, match):
    config = _config_from_key("0|3|1|0|60|4|0")
    mutator(config)
    with pytest.raises(InvalidConfigurationError, match=match):
        validate_configuration(config, artifact.metadata)


def test_seen_training_configuration(artifact, training_keys):
    seen_key = "0|3|1|0|60|4|0"
    frame = validate_configuration(_config_from_key(seen_key), artifact.metadata)
    assert configuration_key(frame) == seen_key
    assert seen_during_training(frame, training_keys) is True
    assert (SEEN_MESSAGE if seen_during_training(frame, training_keys) else NOVEL_MESSAGE) == SEEN_MESSAGE


def test_novel_in_domain_configuration(artifact, training_keys):
    # Held-out test configuration (in-domain values, absent from the 1,634 training records).
    novel_key = "1|3|2|0|70|2|0"
    frame = validate_configuration(_config_from_key(novel_key), artifact.metadata)
    assert seen_during_training(frame, training_keys) is False
    assert (SEEN_MESSAGE if seen_during_training(frame, training_keys) else NOVEL_MESSAGE) == NOVEL_MESSAGE


def test_novelty_is_independent_of_predicted_feasibility(artifact, training_keys):
    # Both novel (held-out) configurations below are in-domain and unseen in training,
    # yet receive opposite binary SVM estimates; checking novelty never alters predict_outcome.
    novel_infeasible = validate_configuration(_config_from_key("1|3|2|0|70|2|0"), artifact.metadata)
    novel_feasible = validate_configuration(_config_from_key("0|3|2|1|77|2|0"), artifact.metadata)

    pred_before_0 = predict_outcome(artifact.model, novel_infeasible)
    pred_before_1 = predict_outcome(artifact.model, novel_feasible)

    assert seen_during_training(novel_infeasible, training_keys) is False
    assert seen_during_training(novel_feasible, training_keys) is False
    assert pred_before_0 == 0 and predict_outcome(artifact.model, novel_infeasible) == 0
    assert pred_before_1 == 1 and predict_outcome(artifact.model, novel_feasible) == 1


def test_streamlit_app_ui_contract():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not at.exception
    # Permanent scope statement visible before prediction.
    assert any("not formal verification" in info.value for info in at.info)
    # Exactly seven bounded configuration inputs in feature order.
    assert len(at.selectbox) == 7
    for sb, feature in zip(at.selectbox, EXPECTED_FEATURES):
        assert feature in sb.label

    # Submit a seen training configuration (0|3|1|0|60|4|0 -> Infeasible).
    for sb, val in zip(at.selectbox, [0, 3, 1, 0, 60, 4, 0]):
        sb.select(val)
    at.button[0].click().run()
    assert not at.exception
    assert any("Infeasible model estimate" in w.value for w in at.warning)
    assert any(SEEN_MESSAGE in m.value for m in at.markdown)

    # Submit a novel configuration (0|3|2|1|77|2|0 -> Feasible).
    for sb, val in zip(at.selectbox, [0, 3, 2, 1, 77, 2, 0]):
        sb.select(val)
    at.button[0].click().run()
    assert not at.exception
    assert any("Feasible model estimate" in s.value for s in at.success)
    assert any(NOVEL_MESSAGE in m.value for m in at.markdown)

