"""Streamlit demonstration of one SVM Outcome Feasibility Prediction for CIE 3."""

import os

import streamlit as st

from feasibility import (
    CLASS_LABELS,
    DEFAULT_BUNDLE_PATH,
    NOVEL_MESSAGE,
    SEEN_MESSAGE,
    ArtifactError,
    InvalidConfigurationError,
    load_bundle,
    load_training_keys,
    predict_outcome,
    seen_during_training,
    validate_configuration,
)

SCOPE_STATEMENT = (
    "This is an ML estimate of Outcome Feasibility within the modeled UCI auction. "
    "It is not formal verification and does not validate a real-world auction, "
    "fraud, authenticity, fairness, or legal status."
)

FIELD_LABELS = {
    "process.b1.capacity": ("Bidder 1 capacity (`process.b1.capacity`)", "Max items Bidder 1 can win (scaled numeric input)."),
    "process.b2.capacity": ("Bidder 2 capacity (`process.b2.capacity`)", "Max items Bidder 2 can win (scaled numeric input)."),
    "process.b3.capacity": ("Bidder 3 capacity (`process.b3.capacity`)", "Max items Bidder 3 can win (scaled numeric input)."),
    "process.b4.capacity": ("Bidder 4 capacity (`process.b4.capacity`)", "Max items Bidder 4 can win (scaled numeric input)."),
    "property.price": ("Evaluated price (`property.price`)", "Price checked within the modeled auction (scaled numeric input)."),
    "property.product": ("Product (`property.product`)", "Product identifier 1–6 (one-hot encoded categorical input)."),
    "property.winner": ("Potential winner (`property.winner`)", "0 = price-only check; 1–4 = Bidder 1–4 (one-hot encoded categorical input)."),
}


@st.cache_resource
def get_resources(bundle_path: str):
    artifact = load_bundle(bundle_path)
    training_keys = load_training_keys(artifact.metadata)
    return artifact, training_keys


st.set_page_config(page_title="Auction Outcome Feasibility Predictor", layout="centered")
st.title("Auction Outcome Feasibility Predictor")
st.caption("CIE 3 demonstration — supervised learning and one class-balanced RBF Support Vector Machine.")
st.info(f"**Scope limitation:** {SCOPE_STATEMENT}")

bundle_path = os.environ.get("AUCTION_MODEL_BUNDLE", str(DEFAULT_BUNDLE_PATH))
try:
    artifact, training_keys = get_resources(bundle_path)
except ArtifactError as err:
    st.error(str(err))
    st.stop()

metadata = artifact.metadata
split = metadata["split"]
metrics = metadata["metrics"]
st.markdown(
    f"**Loaded pipeline:** `ColumnTransformer` (5 scaled numeric + 2 one-hot categorical features) → "
    f"`SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')`  \n"
    f"**Supervised split:** {split['train_records']} training / {split['test_records']} held-out test records "
    f"(stratified 80/20, `random_state={split['random_state']}`) · "
    f"**Held-out MCC:** `{metrics['mcc']:.4f}` · **Target:** `{metadata['target']}` (`verification.time` excluded)"
)

st.subheader("Auction Configuration (7 bounded inputs)")
with st.form("auction_configuration_form"):
    selected = {}
    for feature in metadata["feature_names"]:
        label, help_text = FIELD_LABELS[feature]
        options = metadata["input_domains"][feature]["observed_values"]
        selected[feature] = st.selectbox(label, options=options, help=help_text)
    submitted = st.form_submit_button("Predict Outcome Feasibility")

if submitted:
    try:
        frame = validate_configuration(selected, metadata)
        prediction = predict_outcome(artifact.model, frame)
    except (InvalidConfigurationError, ValueError) as err:
        st.error(str(err))
    else:
        was_seen = seen_during_training(frame, training_keys)
        st.subheader("Outcome Feasibility Prediction")
        if prediction == 1:
            st.success(f"**{CLASS_LABELS[1]}** — `{metadata['class_meanings']['1']}`")
        else:
            st.warning(f"**{CLASS_LABELS[0]}** — `{metadata['class_meanings']['0']}`")

        st.markdown(
            f"**Training-data membership (independent of the predicted class):**  \n"
            f"- {SEEN_MESSAGE if was_seen else NOVEL_MESSAGE}"
        )
        st.caption(
            "Novelty only indicates whether this exact 7-value In-Domain Configuration appeared in the "
            "1,634 training records; a Novel Configuration can still be predicted feasible or infeasible."
        )
        st.dataframe(frame, hide_index=True)
