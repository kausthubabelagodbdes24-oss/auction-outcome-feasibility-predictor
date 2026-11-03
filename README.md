# Auction Outcome Feasibility Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kausthubabelagodbdes24-oss/auction-outcome-feasibility-predictor/blob/main/notebooks/auction_outcome_feasibility_training.ipynb)

A small academic project for a CIE 3 presentation. It demonstrates two syllabus concepts:

1. **Supervised learning**: labelled records, inputs and target kept separate, a fixed train/test split, and evaluation on held-out data.
2. **Support Vector Machine**: one class-balanced RBF SVM that predicts whether an unseen **Model Outcome** is feasible.

> **Scope.** An *Outcome Feasibility Prediction* estimates whether a price/winner outcome is possible **within the modeled auction** of the UCI Auction Verification dataset. It is an ML stand-in for formal model checking. It is **not** fraud detection, authenticity checking, a fairness assessment, or a judgement of legal validity. Terms are defined in [CONTEXT.md](CONTEXT.md).

## Train the model in Google Colab

1. Open [`notebooks/auction_outcome_feasibility_training.ipynb`](notebooks/auction_outcome_feasibility_training.ipynb) in Colab, using the badge above or **File → Upload notebook**.
2. Keep the default **CPU** runtime.
3. Choose **Runtime → Run all**.
4. The last code cell downloads `auction_model_bundle.zip`.

Colab already includes every dependency. [`requirements.txt`](requirements.txt) lists the libraries for use outside Colab.

### What the notebook does

| Step | Detail |
|---|---|
| Data | Official UCI archive, 2,043 records, SHA-256 recorded |
| Inputs (7) | `process.b1–b4.capacity`, `property.price` (scaled); `property.product`, `property.winner` (one-hot encoded) |
| Target | `verification.result` (1 = feasible, 0 = infeasible) |
| Excluded | `verification.time` (model-checker runtime, never an input) |
| Split | Stratified 80/20, `random_state=42` |
| Model | `Pipeline(ColumnTransformer → SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced"))`, no tuning |
| Evaluation | Held-out MCC (primary), accuracy, positive-class precision/recall/F1, confusion matrix |

Scores describe held-out performance **on the same dataset distribution** only.

### Exported bundle (`auction_model_bundle.zip`)

| File | Contents |
|---|---|
| `auction_model.joblib` | Fitted preprocessing + SVM pipeline |
| `model_metadata.json` | Feature order, feature types, class meanings, observed input domains, split settings, SVM config, scikit-learn version, dataset and model SHA-256, metrics |
| `metrics.json` | Held-out metrics and confusion matrix |
| `confusion_matrix.png` | Confusion-matrix plot |

## Data source and licence

The dataset is used under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. Columns were selected for modeling and no records were altered.

> Ordoni, E., Bach, J., Fleck, A., & Bach, J. (2022). *Auction Verification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52K6N

Introductory paper: E. Ordoni, J. Bach, and A.-K. Fleck, "Analyzing and Predicting Verification of Data-Aware Process Models – A Case Study With Spectrum Auctions," *IEEE Access*, vol. 10, 2022. https://doi.org/10.1109/ACCESS.2022.3154445
