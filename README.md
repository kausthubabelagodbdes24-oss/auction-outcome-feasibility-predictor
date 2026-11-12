# Auction Outcome Feasibility Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kausthubabelagodbdes24-oss/auction-outcome-feasibility-predictor/blob/main/notebooks/auction_outcome_feasibility_training.ipynb)

A small academic project for a CIE 3 presentation. It demonstrates two syllabus concepts:

1. **Supervised learning**: labelled records, inputs and target kept separate, a fixed train/test split, and evaluation on held-out data.
2. **Support Vector Machine**: one class-balanced RBF SVM that makes one **Outcome Feasibility Prediction** for an **Auction Configuration**.

> **Permanent scope limitation.** This is an ML estimate of **Outcome Feasibility** within the modeled UCI auction. It is an academic surrogate for formal model checking — **not** formal verification, and it does **not** validate a real-world auction, fraud, authenticity, fairness, or legal status. Terms are defined in [CONTEXT.md](CONTEXT.md).

## 1. Obtain `auction_model_bundle.zip` from Google Colab

1. Open [`notebooks/auction_outcome_feasibility_training.ipynb`](notebooks/auction_outcome_feasibility_training.ipynb) in Google Colab (keep the default **CPU** runtime) and choose **Runtime → Run all**.
2. The final code cell downloads `auction_model_bundle.zip`.
3. Place `auction_model_bundle.zip` in the **repository root** (default path expected by the app and tests, or set `AUCTION_MODEL_BUNDLE` to a custom path).

`auction_model_bundle.zip` and the raw UCI dataset archive (`tmp/auction-verification.zip`) are intentionally excluded from Git via [`.gitignore`](.gitignore).

### What the notebook trains and exports

| Step | Detail |
|---|---|
| Data | Official UCI archive, 2,043 records, SHA-256 verified |
| Inputs (7) | `process.b1–b4.capacity`, `property.price` (scaled); `property.product`, `property.winner` (one-hot encoded) |
| Target | `verification.result` (`1` = feasible, `0` = infeasible) |
| Excluded | `verification.time` (model-checker runtime, never an input) |
| Split | Stratified 80/20 (`1,634` train / `409` held-out test), `random_state=42` |
| Model | `Pipeline(ColumnTransformer → SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=False))` |
| Evaluation | Held-out MCC (`0.5725`), accuracy (`0.8557`), positive-class precision/recall/F1, confusion matrix |
| Bundle files | `auction_model.joblib`, `model_metadata.json`, `metrics.json`, `confusion_matrix.png` |

## 2. Local environment setup and startup

The exported pipeline was serialized with **`scikit-learn==1.6.1`**. Artifact loading verifies the `auction_model.joblib` SHA-256 and requires `scikit-learn==1.6.1` before deserializing.

### Windows launcher

The official **Windows release ZIP** already includes the verified model bundle and runtime-only dependency list. Extract the ZIP completely, then double-click [`RUN_APP.bat`](RUN_APP.bat). Python 3 and an internet connection are required only for the first-time environment setup.

For a source checkout, place `auction_model_bundle.zip` in the repository root first, then double-click `RUN_APP.bat` or run it from Command Prompt:

```bat
RUN_APP.bat
```

The launcher creates or reuses `.venv`, installs the pinned runtime dependencies from `requirements-app.txt` without retaining pip's download cache, and starts the Streamlit application. A version marker skips repeated dependency installation on later launches.

### Manual setup

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
```

Start the Streamlit application with the single command:

```bash
streamlit run app.py
```

The app runs offline using `auction_model_bundle.zip` and the committed training-key reference [`data/training_configurations.json`](data/training_configurations.json) (1,634 canonical seven-value training keys recreated from the `random_state=42` stratified split via `python build_training_reference.py`).

### Seen during training vs. Novel Configuration

- **In-domain configuration seen during training**: all seven values belong to the dataset's observed domains and the exact seven-value combination occurred among the 1,634 training records.
- **Novel Configuration**: all seven values are in-domain, but the exact seven-value combination did **not** occur in the 1,634 training records (for example, a held-out test record or another unseen combination).
- Novelty and the predicted class are **independent**: a Novel Configuration is valid in-domain input and can receive either a **Feasible model estimate** or an **Infeasible model estimate**.

## 3. Run lightweight checks

With `auction_model_bundle.zip` in the repository root:

```bash
pytest
```

## 4. CIE 3 presentation demo sequence

1. **Labelled records**: in the Colab notebook, show the 2,043 UCI records with known `verification.result` labels.
2. **Inputs vs. target**: point out the seven configuration inputs (`process.b1–b4.capacity`, `property.price`, `property.product`, `property.winner`) and the binary target `verification.result` (`verification.time` excluded).
3. **Train/test split**: point out the stratified 80/20 split (`random_state=42`: 1,634 training, 409 held-out test records).
4. **Single pipeline**: show `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) inside one `Pipeline` so preprocessing is fitted on training data only.
5. **Class-balanced RBF SVM**: point out `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")` handling class imbalance.
6. **Held-out evaluation**: highlight the held-out MCC (`0.5725`) and confusion matrix on the 409 test records.
7. **Start Streamlit**: on Windows, double-click `RUN_APP.bat`; otherwise run `streamlit run app.py` after completing the manual setup.
8. **Enter one Auction Configuration**: choose values in the seven bounded select boxes and click **Predict outcome**.
9. **Show the SVM estimate**: point out the single binary result (`Feasible` or `Infeasible`). After prediction, use the revealed tabs to present price sensitivity, held-out evaluation versus the always-infeasible baseline, and the selected value's training distribution.

## Data source and licence

The dataset and derived training-configuration reference are used under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. Input columns were selected for modeling and no records were altered.

> Ordoni, E., Bach, J., Fleck, A., & Bach, J. (2022). *Auction Verification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52K6N

Introductory paper: E. Ordoni, J. Bach, and A.-K. Fleck, "Analyzing and Predicting Verification of Data-Aware Process Models – A Case Study With Spectrum Auctions," *IEEE Access*, vol. 10, 2022. https://doi.org/10.1109/ACCESS.2022.3154445
