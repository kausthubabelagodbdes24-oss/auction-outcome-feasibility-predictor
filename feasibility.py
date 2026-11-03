"""Artifact loading, input validation, training-set novelty, and SVM prediction.

Kept outside app.py so every rule can be tested directly without a browser.
Terminology follows CONTEXT.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

EXPECTED_FEATURES = [
    "process.b1.capacity",
    "process.b2.capacity",
    "process.b3.capacity",
    "process.b4.capacity",
    "property.price",
    "property.product",
    "property.winner",
]
EXPECTED_NUMERIC = EXPECTED_FEATURES[:5]
EXPECTED_CATEGORICAL = EXPECTED_FEATURES[5:]

MODEL_MEMBER = "auction_model.joblib"
METADATA_MEMBER = "model_metadata.json"
REQUIRED_MEMBERS = [MODEL_MEMBER, METADATA_MEMBER, "metrics.json", "confusion_matrix.png"]

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_PATH = ROOT / "auction_model_bundle.zip"
TRAINING_REFERENCE_PATH = ROOT / "data" / "training_configurations.json"

CLASS_LABELS = {0: "Infeasible model estimate", 1: "Feasible model estimate"}
SEEN_MESSAGE = "In-domain configuration seen during training"
NOVEL_MESSAGE = "Novel configuration: this exact combination was not present in the training records"


class ArtifactError(Exception):
    """The model bundle or training reference is missing, corrupt, or incompatible."""


class InvalidConfigurationError(ValueError):
    """An Auction Configuration does not satisfy the seven-feature input contract."""


@dataclass(frozen=True)
class Artifact:
    model: Any
    metadata: dict
    model_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bundle(path: str | Path) -> tuple[dict, bytes]:
    """Return (metadata, model bytes) from the bundle without deserializing the model."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(
            f"Model bundle not found at '{path}'. Run the Colab notebook, download "
            "auction_model_bundle.zip from its final cell, and place it in the repository root "
            "(or set AUCTION_MODEL_BUNDLE to its path)."
        )
    try:
        with zipfile.ZipFile(path) as bundle:
            missing = [m for m in REQUIRED_MEMBERS if m not in bundle.namelist()]
            if missing:
                raise ArtifactError(f"Model bundle '{path}' is missing required files: {missing}.")
            metadata_bytes = bundle.read(METADATA_MEMBER)
            model_bytes = bundle.read(MODEL_MEMBER)
    except zipfile.BadZipFile as exc:
        raise ArtifactError(f"'{path}' is not a valid ZIP file. Re-download it from Colab.") from exc
    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{METADATA_MEMBER} is not valid JSON: {exc}") from exc
    validate_metadata(metadata)
    return metadata, model_bytes


def validate_metadata(metadata: dict) -> None:
    """Check the fields the application relies on and the exact seven-feature contract."""
    required = [
        "feature_names", "numeric_features", "categorical_features", "input_domains",
        "class_meanings", "split", "sklearn_version", "dataset", "model_artifact",
    ]
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ArtifactError(f"{METADATA_MEMBER} is missing required fields: {missing}.")
    if "sha256" not in metadata["model_artifact"] or "sha256" not in metadata["dataset"]:
        raise ArtifactError(f"{METADATA_MEMBER} must record both the model and dataset SHA-256.")
    if metadata["feature_names"] != EXPECTED_FEATURES:
        raise ArtifactError(
            f"Incompatible feature contract: expected {EXPECTED_FEATURES}, "
            f"found {metadata['feature_names']}."
        )
    if (metadata["numeric_features"] != EXPECTED_NUMERIC
            or metadata["categorical_features"] != EXPECTED_CATEGORICAL):
        raise ArtifactError("Incompatible feature contract: numeric/categorical split differs.")
    for feature in EXPECTED_FEATURES:
        values = metadata["input_domains"].get(feature, {}).get("observed_values")
        if not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise ArtifactError(f"input_domains['{feature}'] must list integer observed_values.")
    if set(metadata["class_meanings"]) != {"0", "1"}:
        raise ArtifactError("class_meanings must describe exactly the classes 0 and 1.")


def load_bundle(path: str | Path = DEFAULT_BUNDLE_PATH) -> Artifact:
    """Verify integrity and compatibility, then load the fitted preprocessing + SVM pipeline."""
    metadata, model_bytes = read_bundle(path)

    actual_sha = _sha256(model_bytes)
    expected_sha = metadata["model_artifact"]["sha256"]
    if actual_sha != expected_sha:
        raise ArtifactError(
            f"Model SHA-256 mismatch: metadata records {expected_sha}, file is {actual_sha}. "
            "The bundle may be corrupt or mixed from different runs; re-download it."
        )
    required_version = metadata["sklearn_version"]
    if sklearn.__version__ != required_version:
        raise ArtifactError(
            f"This model was saved with scikit-learn {required_version}, but "
            f"{sklearn.__version__} is installed. Run: pip install scikit-learn=={required_version}"
        )

    model = joblib.load(io.BytesIO(model_bytes))
    if list(getattr(model, "feature_names_in_", [])) != EXPECTED_FEATURES:
        raise ArtifactError("The loaded pipeline was not fitted on the expected seven features.")
    return Artifact(model=model, metadata=metadata, model_sha256=actual_sha)


def validate_configuration(values: Mapping[str, Any], metadata: dict) -> pd.DataFrame:
    """Return a one-row DataFrame in training feature order, or raise InvalidConfigurationError."""
    features = metadata["feature_names"]
    missing = [f for f in features if f not in values]
    unexpected = [f for f in values if f not in features]
    if missing or unexpected:
        raise InvalidConfigurationError(
            f"Expected exactly the seven features {features}. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    row = []
    for feature in features:
        value = values[feature]
        allowed = metadata["input_domains"][feature]["observed_values"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise InvalidConfigurationError(f"{feature} must be an integer, got {value!r}.")
        if int(value) not in allowed:
            raise InvalidConfigurationError(
                f"{feature}={value} is out of domain. Allowed values: {allowed}."
            )
        row.append(int(value))
    return pd.DataFrame([row], columns=features)


def predict_outcome(model: Any, frame: pd.DataFrame) -> int:
    """Return exactly one binary SVM class: 1 = feasible, 0 = infeasible."""
    prediction = np.asarray(model.predict(frame))
    if prediction.shape != (1,):
        raise ValueError(f"Expected exactly one prediction, got shape {prediction.shape}.")
    label = int(prediction[0])
    if label not in CLASS_LABELS:
        raise ValueError(f"Prediction is not binary: {prediction[0]!r}.")
    return label


def configuration_key(frame: pd.DataFrame) -> str:
    """Deterministic canonical key for one row, e.g. '0|3|1|0|60|4|0' (feature order)."""
    if len(frame) != 1:
        raise ValueError("configuration_key expects a one-row DataFrame.")
    return "|".join(str(int(v)) for v in frame.iloc[0][EXPECTED_FEATURES])


def load_training_keys(metadata: dict, path: str | Path = TRAINING_REFERENCE_PATH) -> frozenset[str]:
    """Load the committed training-configuration keys and check they match this artifact."""
    path = Path(path)
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Cannot read training reference '{path}': {exc}. "
            "Regenerate it with: python build_training_reference.py"
        ) from exc
    split = metadata["split"]
    checks = {
        "dataset_sha256": metadata["dataset"]["sha256"],
        "feature_names": metadata["feature_names"],
        "test_size": split["test_size"],
        "random_state": split["random_state"],
        "train_records": split["train_records"],
    }
    for field, expected in checks.items():
        if reference.get(field) != expected:
            raise ArtifactError(
                f"Training reference does not match the model bundle ({field}: "
                f"{reference.get(field)!r} != {expected!r}). Regenerate it."
            )
    keys = frozenset(reference["configuration_keys"])
    if len(keys) != split["train_records"]:
        raise ArtifactError("Training reference key count does not match train_records.")
    return keys


def seen_during_training(frame: pd.DataFrame, training_keys: frozenset[str]) -> bool:
    """True if this exact seven-value combination is one of the training records."""
    return configuration_key(frame) in training_keys
