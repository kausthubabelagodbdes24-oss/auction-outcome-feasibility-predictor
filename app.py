"""Streamlit demonstration of one SVM Outcome Feasibility Prediction for CIE 3."""

import os

import streamlit as st

from feasibility import (
    DEFAULT_BUNDLE_PATH,
    ArtifactError,
    InvalidConfigurationError,
    load_bundle,
    predict_outcome,
    validate_configuration,
)

FIELD_LABELS = {
    "process.b1.capacity": "Bidder 1 capacity",
    "process.b2.capacity": "Bidder 2 capacity",
    "process.b3.capacity": "Bidder 3 capacity",
    "process.b4.capacity": "Bidder 4 capacity",
    "property.price": "Evaluated price",
    "property.product": "Product",
    "property.winner": "Potential winner",
}


@st.cache_resource
def get_artifact(bundle_path: str):
    return load_bundle(bundle_path)


st.set_page_config(page_title="Auction Outcome Feasibility Predictor", layout="centered")
st.title("Auction Outcome Feasibility Predictor")

bundle_path = os.environ.get("AUCTION_MODEL_BUNDLE", str(DEFAULT_BUNDLE_PATH))
try:
    artifact = get_artifact(bundle_path)
except ArtifactError as err:
    st.error(str(err))
    st.stop()

metadata = artifact.metadata
with st.form("auction_configuration_form"):
    selected = {}
    for feature in metadata["feature_names"]:
        options = metadata["input_domains"][feature]["observed_values"]
        selected[feature] = st.selectbox(FIELD_LABELS[feature], options=options)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        frame = validate_configuration(selected, metadata)
        prediction = predict_outcome(artifact.model, frame)
    except (InvalidConfigurationError, ValueError) as err:
        st.error(str(err))
    else:
        st.subheader("Outcome Feasibility Prediction")
        if prediction == 1:
            st.success("Feasible")
        else:
            st.warning("Infeasible")
