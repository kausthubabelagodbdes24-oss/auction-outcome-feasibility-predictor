"""Regenerate data/training_configurations.json from the official UCI dataset.

Recreates the notebook's exact training split and stores only the seven-value
input combination of each training record (no labels), so the app can tell
whether a configuration was seen during training without internet access.

Usage:  python build_training_reference.py
Needs:  auction_model_bundle.zip in the repository root (or AUCTION_MODEL_BUNDLE),
        and either tmp/auction-verification.zip or internet access.
"""

import hashlib
import io
import json
import os
import urllib.request
import zipfile

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from feasibility import (
    DEFAULT_BUNDLE_PATH, EXPECTED_FEATURES, ROOT, TRAINING_REFERENCE_PATH, load_bundle,
)

LOCAL_DATASET_ZIP = ROOT / "tmp" / "auction-verification.zip"


def read_dataset_csv(dataset: dict) -> bytes:
    if LOCAL_DATASET_ZIP.is_file():
        print(f"Using local dataset archive: {LOCAL_DATASET_ZIP}")
        archive_bytes = LOCAL_DATASET_ZIP.read_bytes()
    else:
        print(f"Downloading dataset archive: {dataset['url']}")
        with urllib.request.urlopen(dataset["url"], timeout=60) as response:
            archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return archive.read(dataset["member"])


def main() -> None:
    artifact = load_bundle(os.environ.get("AUCTION_MODEL_BUNDLE", DEFAULT_BUNDLE_PATH))
    metadata = artifact.metadata
    dataset, split = metadata["dataset"], metadata["split"]

    csv_bytes = read_dataset_csv(dataset)
    sha = hashlib.sha256(csv_bytes).hexdigest()
    assert sha == dataset["sha256"], f"Dataset SHA-256 {sha} != metadata {dataset['sha256']}"

    # Same columns, target, and split call as the training notebook.
    df = pd.read_csv(io.BytesIO(csv_bytes))
    X = df[EXPECTED_FEATURES].copy()
    y = df[metadata["target"]].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split["test_size"], stratify=y, random_state=split["random_state"]
    )
    assert len(X_train) == split["train_records"] == 1634, len(X_train)
    assert len(X_test) == split["test_records"], len(X_test)

    # Cross-check: the loaded model reproduces the recorded held-out confusion matrix,
    # so this is the same partition the model was evaluated on.
    with zipfile.ZipFile(os.environ.get("AUCTION_MODEL_BUNDLE", DEFAULT_BUNDLE_PATH)) as bundle:
        recorded = json.loads(bundle.read("metrics.json"))["confusion_matrix"]["matrix"]
    reproduced = confusion_matrix(y_test, artifact.model.predict(X_test), labels=[0, 1]).tolist()
    assert reproduced == recorded, f"Confusion matrix {reproduced} != recorded {recorded}"

    keys = sorted("|".join(str(int(v)) for v in row) for row in X_train.itertuples(index=False))
    assert len(set(keys)) == len(keys) == 1634, "Training configurations are not unique"

    reference = {
        "description": (
            "Exact seven-value input combinations of the 1,634 training records "
            "(labels excluded). Key = values joined by '|' in feature_names order."
        ),
        "source": dataset["citation"],
        "source_url": dataset["url"],
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
                   "Derived: input columns selected for training records; values unaltered.",
        "dataset_sha256": sha,
        "feature_names": EXPECTED_FEATURES,
        "split_method": split["method"],
        "test_size": split["test_size"],
        "random_state": split["random_state"],
        "train_records": len(keys),
        "configuration_keys": keys,
    }
    TRAINING_REFERENCE_PATH.parent.mkdir(exist_ok=True)
    TRAINING_REFERENCE_PATH.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(keys)} training configuration keys to {TRAINING_REFERENCE_PATH}")
    print(f"Held-out confusion matrix reproduced: {reproduced}")


if __name__ == "__main__":
    main()
