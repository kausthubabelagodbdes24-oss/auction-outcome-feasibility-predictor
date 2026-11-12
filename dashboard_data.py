"""Dashboard data loaders, SVM hyperplane margin, price sensitivity, and demo presets.

Provides structured data helpers for the upgraded Streamlit dashboard without
modifying the core prediction contract in ``feasibility.py``. All terminology
follows ``CONTEXT.md``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feasibility import (
    CLASS_LABELS,
    DEFAULT_BUNDLE_PATH,
    EXPECTED_FEATURES,
    METADATA_MEMBER,
    REQUIRED_MEMBERS,
    TRAINING_REFERENCE_PATH,
    Artifact,
    ArtifactError,
    predict_outcome,
    seen_during_training,
    validate_configuration,
    validate_metadata,
)

METRICS_MEMBER = "metrics.json"
CONFUSION_MATRIX_MEMBER = "confusion_matrix.png"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_bundle_extras(bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> dict[str, Any]:
    """Read metrics, confusion matrix PNG bytes, and metadata from the bundle ZIP.

    Uses the same ZIP path and required-member validation rules as
    ``feasibility.read_bundle``, raising ``ArtifactError`` if the archive or any
    expected member is missing or malformed.
    """
    path = Path(bundle_path)
    if not path.is_file():
        raise ArtifactError(
            f"Model bundle not found at '{path}'. Run the Colab notebook, download "
            "auction_model_bundle.zip from its final cell, and place it in the repository root "
            "(or set AUCTION_MODEL_BUNDLE to its path)."
        )
    try:
        with zipfile.ZipFile(path) as bundle:
            missing = [member for member in REQUIRED_MEMBERS if member not in bundle.namelist()]
            if missing:
                raise ArtifactError(f"Model bundle '{path}' is missing required files: {missing}.")
            metadata_bytes = bundle.read(METADATA_MEMBER)
            metrics_bytes = bundle.read(METRICS_MEMBER)
            confusion_matrix_png = bundle.read(CONFUSION_MATRIX_MEMBER)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArtifactError(f"'{path}' is not a valid ZIP file. Re-download it from Colab.") from exc

    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{METADATA_MEMBER} is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError(f"{METADATA_MEMBER} must decode to a JSON object.")
    validate_metadata(metadata)

    try:
        metrics = json.loads(metrics_bytes)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{METRICS_MEMBER} is not valid JSON: {exc}") from exc
    if not isinstance(metrics, dict) or "metrics" not in metrics or "confusion_matrix" not in metrics:
        raise ArtifactError(f"{METRICS_MEMBER} must contain 'metrics' and 'confusion_matrix' objects.")

    if not confusion_matrix_png.startswith(_PNG_SIGNATURE):
        raise ArtifactError(f"{CONFUSION_MATRIX_MEMBER} is missing or not a valid PNG image.")

    return {
        "metrics": metrics,
        "confusion_matrix_png": confusion_matrix_png,
        "metadata": metadata,
    }


def _parse_configuration_keys(raw_keys: Any, expected_width: int) -> list[list[int]]:
    """Parse canonical pipe-delimited training configuration keys into integer rows."""
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ArtifactError("Training reference 'configuration_keys' must be a non-empty list.")
    rows: list[list[int]] = []
    for item in raw_keys:
        if not isinstance(item, str):
            raise ArtifactError(f"Training configuration key must be a string, got {item!r}.")
        parts = item.split("|")
        if len(parts) != expected_width:
            raise ArtifactError(
                f"Expected {expected_width} values in key {item!r}, found {len(parts)}."
            )
        try:
            rows.append([int(part) for part in parts])
        except ValueError as exc:
            raise ArtifactError(f"Non-integer value in training key {item!r}: {exc}") from exc
    return rows


def load_training_distribution(path: str | Path = TRAINING_REFERENCE_PATH) -> dict[str, Any]:
    """Parse ``data/training_configurations.json`` into per-feature frequency tables and stats.

    Returns a dictionary containing:
    - ``metadata``: split and provenance metadata (``train_records``, ``random_state``,
      ``test_size``, ``split_method``, ``dataset_sha256``, ``feature_names``, ``source``,
      ``license``).
    - ``feature_frequencies``: mapping from feature name to a DataFrame with columns
      ``["feature", "value", "count", "share"]`` sorted by ``value``.
    - ``frequencies_df``: concatenated DataFrame of all per-feature frequencies.
    - ``price_stats``: summary dictionary for ``property.price`` with keys
      ``min``, ``max``, ``median``, and ``mean``.
    - ``total_records``: integer count of training configuration keys parsed.
    """
    ref_path = Path(path)
    try:
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Cannot read training reference '{ref_path}': {exc}. "
            "Regenerate it with: python build_training_reference.py"
        ) from exc

    if not isinstance(reference, dict):
        raise ArtifactError(f"Training reference '{ref_path}' must decode to a JSON object.")

    feature_names = reference.get("feature_names")
    if feature_names != EXPECTED_FEATURES:
        raise ArtifactError(
            f"Training reference feature_names mismatch: expected {EXPECTED_FEATURES}, "
            f"found {feature_names!r}."
        )

    rows = _parse_configuration_keys(reference.get("configuration_keys"), len(feature_names))
    total_records = len(rows)
    declared_records = reference.get("train_records")
    if declared_records is not None and declared_records != total_records:
        raise ArtifactError(
            f"Training reference train_records ({declared_records}) does not match "
            f"parsed key count ({total_records})."
        )

    records_df = pd.DataFrame(rows, columns=feature_names)
    feature_frequencies: dict[str, pd.DataFrame] = {}
    for feature in feature_names:
        counts = records_df[feature].value_counts().sort_index()
        feature_frequencies[feature] = pd.DataFrame(
            {
                "feature": feature,
                "value": counts.index.astype(int),
                "count": counts.to_numpy(dtype=int),
                "share": (counts / total_records).to_numpy(dtype=float),
            }
        )

    frequencies_df = pd.concat(feature_frequencies.values(), ignore_index=True)
    price_series = records_df["property.price"]
    price_stats: dict[str, int | float] = {
        "min": int(price_series.min()),
        "max": int(price_series.max()),
        "median": float(price_series.median()),
        "mean": float(price_series.mean()),
    }

    metadata = {
        "description": reference.get("description"),
        "source": reference.get("source"),
        "source_url": reference.get("source_url"),
        "license": reference.get("license"),
        "dataset_sha256": reference.get("dataset_sha256"),
        "feature_names": list(feature_names),
        "split_method": reference.get("split_method"),
        "test_size": reference.get("test_size"),
        "random_state": reference.get("random_state"),
        "train_records": total_records,
    }

    return {
        "metadata": metadata,
        "feature_frequencies": feature_frequencies,
        "frequencies_df": frequencies_df,
        "price_stats": price_stats,
        "total_records": total_records,
    }


def compute_decision_margin(artifact: Artifact, frame: pd.DataFrame) -> float:
    """Return the signed distance to the RBF SVM decision hyperplane for one row.

    Note: The bundle's ``SVC`` is fitted with ``probability=False``. This value is
    the raw geometric margin from ``decision_function`` (positive for class 1
    ``Feasible model estimate``, negative for class 0 ``Infeasible model estimate``)
    and MUST NOT be interpreted or presented as a calibrated probability.
    """
    if len(frame) != 1:
        raise ValueError("compute_decision_margin expects a one-row DataFrame.")
    # SVC uses probability=False; decision_function returns distance to the RBF hyperplane, not a probability.
    scores = np.asarray(artifact.model.decision_function(frame))
    if scores.shape != (1,):
        raise ValueError(f"Expected one decision margin score, got shape {scores.shape}.")
    return float(scores[0])


def compute_price_sensitivity(
    artifact: Artifact,
    base_values: dict[str, int],
    training_keys: frozenset[str],
) -> pd.DataFrame:
    """Sweep ``property.price`` across its observed domain with the other six features fixed.

    Evaluates Outcome Feasibility Prediction, RBF SVM decision hyperplane margin,
    and training-set membership for every observed ``property.price`` value in
    ``artifact.metadata["input_domains"]["property.price"]["observed_values"]``.
    """
    validate_configuration(base_values, artifact.metadata)
    selected_price = int(base_values["property.price"])
    observed_prices = sorted(artifact.metadata["input_domains"]["property.price"]["observed_values"])

    records: list[dict[str, Any]] = []
    for price in observed_prices:
        candidate = {**base_values, "property.price": int(price)}
        frame = validate_configuration(candidate, artifact.metadata)
        prediction = predict_outcome(artifact.model, frame)
        margin = compute_decision_margin(artifact, frame)
        was_seen = seen_during_training(frame, training_keys)
        records.append(
            {
                "price": int(price),
                "prediction": prediction,
                "outcome_label": CLASS_LABELS[prediction],
                "decision_margin": margin,
                "seen_during_training": was_seen,
                "is_selected": int(price) == selected_price,
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "price",
            "prediction",
            "outcome_label",
            "decision_margin",
            "seen_during_training",
            "is_selected",
        ],
    )
