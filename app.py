"""Streamlit demonstration of SVM Outcome Feasibility Prediction for CIE 3."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from charts import (
    build_confusion_matrix_chart,
    build_domain_distribution_chart,
    build_metrics_comparison_chart,
    build_price_sensitivity_chart,
)
from dashboard_data import (
    DEMO_PRESETS,
    compute_decision_margin,
    compute_price_sensitivity,
    load_bundle_extras,
    load_training_distribution,
)
from feasibility import (
    CLASS_LABELS,
    DEFAULT_BUNDLE_PATH,
    EXPECTED_FEATURES,
    NOVEL_MESSAGE,
    SEEN_MESSAGE,
    TRAINING_REFERENCE_PATH,
    ArtifactError,
    InvalidConfigurationError,
    configuration_key,
    load_bundle,
    load_training_keys,
    predict_outcome,
    seen_during_training,
    validate_configuration,
)
from theme_astryx import (
    inject_astryx_theme,
    register_astryx_altair_theme,
    render_hero_header,
    render_kpi_strip,
    render_prediction_card,
    render_section_header,
    render_terminology_grid,
    resolve_tokens,
)

SCOPE_STATEMENT = (
    "This is an ML estimate of Outcome Feasibility within the modeled UCI auction. "
    "It is not formal verification and does not validate a real-world auction, "
    "fraud, authenticity, fairness, or legal status."
)

FIELD_LABELS = {
    "process.b1.capacity": (
        "Bidder 1 capacity (`process.b1.capacity`)",
        "Max items Bidder 1 can win (scaled numeric input).",
    ),
    "process.b2.capacity": (
        "Bidder 2 capacity (`process.b2.capacity`)",
        "Max items Bidder 2 can win (scaled numeric input).",
    ),
    "process.b3.capacity": (
        "Bidder 3 capacity (`process.b3.capacity`)",
        "Max items Bidder 3 can win (scaled numeric input).",
    ),
    "process.b4.capacity": (
        "Bidder 4 capacity (`process.b4.capacity`)",
        "Max items Bidder 4 can win (scaled numeric input).",
    ),
    "property.price": (
        "Evaluated price (`property.price`)",
        "Price checked within the modeled auction (scaled numeric input).",
    ),
    "property.product": (
        "Product (`property.product`)",
        "Product identifier 1–6 (one-hot encoded categorical input).",
    ),
    "property.winner": (
        "Potential winner (`property.winner`)",
        "0 = price-only check; 1–4 = Bidder 1–4 (one-hot encoded categorical input).",
    ),
}

TERMINOLOGY_TERMS: list[tuple[str, str, str]] = [
    (
        "Auction Configuration",
        "The seven bidder-capacity, price, product, and potential-winner values supplied for one modeled verification run.",
        "Real auction, auction transaction",
    ),
    (
        "In-Domain Configuration",
        "An Auction Configuration whose individual values all fall within the published dataset domains.",
        "Valid auction, verified configuration",
    ),
    (
        "Novel Configuration",
        "An In-Domain Configuration whose exact seven-value combination does not occur in the model's training data.",
        "Invalid configuration, impossible auction",
    ),
    (
        "Model Outcome",
        "A price and, when specified, winner assignment evaluated within the modeled simultaneous multi-round auction.",
        "Real-world auction result, verified transaction",
    ),
    (
        "Outcome Feasibility",
        "Whether a Model Outcome is possible within the modeled auction according to the dataset's binary verification result.",
        "Auction validity, authenticity, fairness, fraud status, legal validity",
    ),
    (
        "Outcome Feasibility Prediction",
        "An SVM estimate of Outcome Feasibility used as an academic surrogate for formal model checking.",
        "Auction verification, fraud detection, real-world validation",
    ),
    (
        "Verification Result",
        "The source dataset field `verification.result`, used as the binary label for Outcome Feasibility.",
        "Verified auction, legal verdict",
    ),
]

_PRESETS_BY_LABEL: dict[str, dict[str, Any]] = {
    preset["label"]: preset for preset in DEMO_PRESETS
}


@st.cache_resource
def get_resources(bundle_path: str):
    """Load the verified model bundle and matching training configuration keys."""
    artifact = load_bundle(bundle_path)
    training_keys = load_training_keys(artifact.metadata)
    return artifact, training_keys


@st.cache_resource
def get_bundle_extras(bundle_path: str) -> dict[str, Any]:
    """Load held-out evaluation metrics and bundled confusion matrix PNG bytes."""
    return load_bundle_extras(bundle_path)


@st.cache_data
def get_training_distribution(reference_path: str) -> dict[str, Any]:
    """Load per-feature observed frequency distributions from the training reference."""
    return load_training_distribution(reference_path)


def _feature_state_key(feature: str) -> str:
    return f"cfg_{feature}"


def _init_configuration_state(metadata: dict[str, Any]) -> None:
    """Ensure all seven bounded configuration keys exist in session state."""
    for feature in metadata["feature_names"]:
        key = _feature_state_key(feature)
        if key not in st.session_state:
            st.session_state[key] = int(metadata["input_domains"][feature]["observed_values"][0])


def _apply_demo_preset() -> None:
    """Populate the seven configuration selectbox states when a CIE 3 preset is chosen."""
    chosen = st.session_state.get("cie3_demo_preset")
    preset = _PRESETS_BY_LABEL.get(chosen)
    if not preset:
        return
    for feature, value in preset["values"].items():
        st.session_state[_feature_state_key(feature)] = int(value)


def _collect_active_values(feature_names: list[str]) -> dict[str, int]:
    """Read the active seven-feature Auction Configuration from session state."""
    return {feature: int(st.session_state[_feature_state_key(feature)]) for feature in feature_names}


def _render_sidebar_appearance() -> tuple[str, str]:
    """Render the sidebar appearance control and return (css_mode, altair_mode)."""
    with st.sidebar:
        st.markdown("### Astryx Workspace")
        appearance = st.segmented_control(
            "Appearance",
            options=["Auto", "Light", "Dark"],
            default="Auto",
            required=True,
            key="astryx_appearance_mode",
            help="Switch Astryx Neutral theme surface contrast.",
        )
    css_mode = (appearance or "Auto").lower()
    altair_mode = "dark" if css_mode == "dark" else "light"
    return css_mode, altair_mode


def _render_sidebar_controls(metadata: dict[str, Any], model_sha256: str) -> None:
    """Render CIE 3 demo presets and compact artifact provenance in the sidebar."""
    split = metadata["split"]
    svm = metadata["svm_config"]
    with st.sidebar:
        st.divider()
        st.markdown("#### CIE 3 Demo Presets")
        chosen = st.pills(
            "Select a preset configuration",
            options=[preset["label"] for preset in DEMO_PRESETS],
            selection_mode="single",
            key="cie3_demo_preset",
            on_change=_apply_demo_preset,
            help="Pre-fills the seven Auction Configuration inputs below.",
        )
        if chosen and chosen in _PRESETS_BY_LABEL:
            st.caption(_PRESETS_BY_LABEL[chosen]["description"])
        else:
            st.caption("Choose a preset to load an example Seen or Novel Configuration.")

        st.divider()
        st.markdown("#### Artifact & Split Provenance")
        st.markdown(
            f"- **Pipeline:** `RBF SVM` (`C={svm['C']}`, `{svm['class_weight']}`)\n"
            f"- **Runtime:** `scikit-learn {metadata['sklearn_version']}`\n"
            f"- **Split:** `{split['train_records']:,}` train / `{split['test_records']:,}` test "
            f"(80/20, `random_state={split['random_state']}`)\n"
            f"- **Excluded:** `{', '.join(metadata['excluded_columns'])}`\n"
            f"- **Model SHA-256:** `{model_sha256[:12]}...`"
        )


def _render_configuration_studio(
    artifact: Any,
    metadata: dict[str, Any],
    training_keys: frozenset[str],
    active_key: str,
) -> None:
    """Render Band 1: the seven-input Configuration Studio form and Outcome Summary."""
    studio_col, outcome_col = st.columns([1.2, 1.0], gap="large")

    with studio_col:
        render_section_header(
            "Configuration Studio",
            "Specify 7 bounded integer inputs within the UCI Auction Verification domains.",
        )
        with st.form("auction_configuration_form"):
            st.markdown("##### Bidder Capacities (`b1`–`b4`)")
            selected: dict[str, int] = {}

            b_row1_c1, b_row1_c2 = st.columns(2)
            for col, feature in zip((b_row1_c1, b_row1_c2), metadata["feature_names"][:2]):
                label, help_text = FIELD_LABELS[feature]
                options = metadata["input_domains"][feature]["observed_values"]
                with col:
                    selected[feature] = st.selectbox(
                        label,
                        options=options,
                        key=_feature_state_key(feature),
                        help=help_text,
                    )

            b_row2_c1, b_row2_c2 = st.columns(2)
            for col, feature in zip((b_row2_c1, b_row2_c2), metadata["feature_names"][2:4]):
                label, help_text = FIELD_LABELS[feature]
                options = metadata["input_domains"][feature]["observed_values"]
                with col:
                    selected[feature] = st.selectbox(
                        label,
                        options=options,
                        key=_feature_state_key(feature),
                        help=help_text,
                    )

            st.markdown("##### Auction Properties (`price`, `product`, `winner`)")
            p_col1, p_col2, p_col3 = st.columns(3)
            for col, feature in zip((p_col1, p_col2, p_col3), metadata["feature_names"][4:7]):
                label, help_text = FIELD_LABELS[feature]
                options = metadata["input_domains"][feature]["observed_values"]
                with col:
                    selected[feature] = st.selectbox(
                        label,
                        options=options,
                        key=_feature_state_key(feature),
                        help=help_text,
                    )

            submitted = st.form_submit_button(
                "Predict Outcome Feasibility",
                type="primary",
                width="stretch",
            )

    if submitted:
        try:
            frame = validate_configuration(selected, metadata)
            prediction = predict_outcome(artifact.model, frame)
            margin = compute_decision_margin(artifact, frame)
        except (InvalidConfigurationError, ValueError) as err:
            st.session_state["last_prediction_error"] = str(err)
            st.session_state.pop("last_prediction_result", None)
        else:
            st.session_state.pop("last_prediction_error", None)
            was_seen = seen_during_training(frame, training_keys)
            st.session_state["last_prediction_result"] = {
                "frame": frame,
                "prediction": prediction,
                "margin": margin,
                "was_seen": was_seen,
                "config_key": configuration_key(frame),
            }

    with outcome_col:
        render_section_header(
            "Outcome Summary",
            "Binary SVM surrogate estimate and independent training-set membership check.",
        )
        pred_error = st.session_state.get("last_prediction_error")
        result = st.session_state.get("last_prediction_result")
        if pred_error:
            st.error(pred_error)
        elif result is not None:
            prediction = result["prediction"]
            was_seen = result["was_seen"]
            margin = result["margin"]
            config_key = result["config_key"]
            frame = result["frame"]

            if prediction == 1:
                st.success(f"**{CLASS_LABELS[1]}** — `{metadata['class_meanings']['1']}`")
            else:
                st.warning(f"**{CLASS_LABELS[0]}** — `{metadata['class_meanings']['0']}`")

            st.markdown(
                f"**Training-data membership (independent of the predicted class):**  \n"
                f"- {SEEN_MESSAGE if was_seen else NOVEL_MESSAGE}"
            )
            render_prediction_card(
                outcome_label=CLASS_LABELS[prediction],
                is_feasible=(prediction == 1),
                is_seen=was_seen,
                decision_margin=margin,
                config_key=config_key,
            )
            if active_key != config_key:
                st.caption(
                    f"Form inputs (`{active_key}`) changed after this estimate (`{config_key}`). "
                    "Click **Predict Outcome Feasibility** to evaluate the updated inputs."
                )
            st.caption(
                "Novelty only indicates whether this exact 7-value In-Domain Configuration appeared in the "
                "1,634 training records; a Novel Configuration can still be predicted feasible or infeasible."
            )
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.caption(
                "Select a CIE 3 demo preset in the sidebar or adjust the seven inputs on the left, "
                "then click **Predict Outcome Feasibility** to inspect the SVM Outcome Feasibility Prediction."
            )
            st.markdown(
                f"**Active Auction Configuration key:** `{active_key}`  \n"
                "The live RBF hyperplane margin in the KPI strip and Price Sensitivity tab updates "
                "as inputs change."
            )


def _render_price_sensitivity_tab(
    artifact: Any,
    active_values: dict[str, int],
    training_keys: frozenset[str],
    theme_tokens: dict[str, str],
) -> None:
    """Render Tab 1: Price Sensitivity sweep across observed property.price values."""
    render_section_header(
        "Price Sensitivity & Hyperplane Margin",
        "Sweeps `property.price` (59–90) while holding the other six inputs fixed.",
    )
    sweep_df = compute_price_sensitivity(artifact, active_values, training_keys)
    selected_price = int(active_values["property.price"])
    chart = build_price_sensitivity_chart(sweep_df, selected_price=selected_price, theme_tokens=theme_tokens)
    st.altair_chart(chart, width="stretch", theme=None)

    feasible_prices = sweep_df.loc[sweep_df["prediction"] == 1, "price"].tolist()
    seen_count = int(sweep_df["seen_during_training"].sum())
    selected_row = sweep_df.loc[sweep_df["is_selected"]].iloc[0]
    if feasible_prices:
        regime_note = (
            f"**{len(feasible_prices)} of {len(sweep_df)}** observed prices are estimated feasible "
            f"(range `{min(feasible_prices)}`–`{max(feasible_prices)}`)"
        )
    else:
        regime_note = f"**0 of {len(sweep_df)}** observed prices cross into the feasible region (`margin >= 0`)"

    st.markdown(
        f"{regime_note}. At the selected price `{selected_price}`, the SVM hyperplane margin is "
        f"`{selected_row['decision_margin']:+.4f}` (`{selected_row['outcome_label']}`). "
        f"Across this slice, **{seen_count}** prices appear in the training records and "
        f"**{len(sweep_df) - seen_count}** are Novel Configurations."
    )

    display_df = sweep_df.rename(
        columns={
            "price": "Price (property.price)",
            "outcome_label": "Model Estimate",
            "decision_margin": "SVM Decision Margin",
            "seen_during_training": "Seen in Training",
            "is_selected": "Selected",
        }
    )[
        [
            "Price (property.price)",
            "Model Estimate",
            "SVM Decision Margin",
            "Seen in Training",
            "Selected",
        ]
    ]
    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        height=260,
        column_config={
            "SVM Decision Margin": st.column_config.NumberColumn(format="%+.4f"),
        },
    )


def _render_evaluation_tab(
    extras: dict[str, Any],
    theme_tokens: dict[str, str],
) -> None:
    """Render Tab 2: Held-out Confusion Matrix and Baseline Comparison."""
    render_section_header(
        "Model Evaluation & Always-Infeasible Baseline",
        "Held-out stratified 20% test partition (n = 409 records; 52 feasible, 357 infeasible).",
    )
    metrics_payload = extras["metrics"]
    col_cm, col_bars = st.columns(2, gap="large")
    with col_cm:
        cm_chart = build_confusion_matrix_chart(metrics_payload, theme_tokens=theme_tokens)
        st.altair_chart(cm_chart, width="stretch", theme=None)
    with col_bars:
        bars_chart = build_metrics_comparison_chart(metrics_payload, theme_tokens=theme_tokens)
        st.altair_chart(bars_chart, width="stretch", theme=None)

    m = metrics_payload["metrics"]
    baseline = metrics_payload["always_infeasible_accuracy"]
    st.caption(
        f"Because only {metrics_payload['test_feasible_records']}/{metrics_payload['test_records']} "
        f"({1.0 - baseline:.2%}) held-out records are feasible, predicting infeasible for every input "
        f"achieves {baseline:.2%} accuracy yet 0.0000 MCC. The class-balanced RBF SVM achieves "
        f"{m['mcc']:.4f} MCC and {m['recall_feasible']:.2%} feasible recall."
    )

    with st.expander("Bundled Evaluation Reference (`metrics.json` & `confusion_matrix.png`)"):
        ref_col1, ref_col2 = st.columns([1.0, 1.1], gap="large")
        with ref_col1:
            metrics_rows = pd.DataFrame(
                [
                    {"Metric": "Held-out MCC (Primary)", "Score": f"{m['mcc']:.4f}"},
                    {"Metric": "Held-out Accuracy", "Score": f"{m['accuracy']:.4f}"},
                    {"Metric": "Always-Infeasible Baseline Accuracy", "Score": f"{baseline:.4f}"},
                    {"Metric": "Feasible Recall (Class 1)", "Score": f"{m['recall_feasible']:.4f}"},
                    {"Metric": "Feasible Precision (Class 1)", "Score": f"{m['precision_feasible']:.4f}"},
                    {"Metric": "Feasible F1 (Class 1)", "Score": f"{m['f1_feasible']:.4f}"},
                ]
            )
            st.dataframe(metrics_rows, hide_index=True, width="stretch")
        with ref_col2:
            st.image(
                extras["confusion_matrix_png"],
                caption="Bundled confusion_matrix.png from the Colab training artifact",
                width=440,
            )


def _render_domain_explorer_tab(
    distribution: dict[str, Any],
    active_values: dict[str, int],
    theme_tokens: dict[str, str],
) -> None:
    """Render Tab 3: Training Domain Explorer across the 1,634 training configurations."""
    render_section_header(
        "Training Domain Explorer",
        f"Observed feature frequencies across n = {distribution['total_records']:,} training records.",
    )
    selected_feature = st.pills(
        "Inspect feature domain",
        options=EXPECTED_FEATURES,
        default="property.price",
        required=True,
        selection_mode="single",
        key="domain_explorer_feature",
    )
    feature_name = selected_feature or "property.price"
    feature_df = distribution["feature_frequencies"][feature_name]
    active_val = int(active_values[feature_name])

    dist_chart = build_domain_distribution_chart(
        feature_df,
        feature_name=feature_name,
        selected_value=active_val,
        theme_tokens=theme_tokens,
    )
    st.altair_chart(dist_chart, width="stretch", theme=None)

    match_row = feature_df.loc[feature_df["value"] == active_val]
    price_stats = distribution["price_stats"]
    if not match_row.empty:
        row = match_row.iloc[0]
        st.markdown(
            f"Active value **`{feature_name} = {active_val}`** appears in **{int(row['count']):,}** "
            f"of **{distribution['total_records']:,}** training records (**{float(row['share']):.1%}**)."
        )
    st.caption(
        f"Training `property.price` summary (n = {distribution['total_records']:,}): "
        f"min `{price_stats['min']}` · median `{price_stats['median']:.1f}` · "
        f"mean `{price_stats['mean']:.2f}` · max `{price_stats['max']}`."
    )


def _render_guide_tab(metadata: dict[str, Any]) -> None:
    """Render Tab 4: CIE 3 Terminology Grid and Pipeline Architecture."""
    render_section_header(
        "CIE 3 Domain Terminology (`CONTEXT.md`)",
        "Standardized vocabulary for presenting this UCI Auction Verification surrogate model.",
    )
    render_terminology_grid(TERMINOLOGY_TERMS)

    render_section_header(
        "Pipeline Architecture & Data Leakage Guard",
        "Preprocessing, kernel specification, and split provenance recorded in model_metadata.json.",
    )
    split = metadata["split"]
    svm = metadata["svm_config"]
    dataset = metadata["dataset"]
    st.markdown(
        f"1. **Preprocessing (`ColumnTransformer`):** `StandardScaler` on 5 numeric inputs "
        f"(`{', '.join(metadata['numeric_features'])}`) and `OneHotEncoder(handle_unknown='ignore')` "
        f"on 2 categorical inputs (`{', '.join(metadata['categorical_features'])}`).\n"
        f"2. **Classifier (`SVC`):** `kernel='{svm['kernel']}'`, `C={svm['C']}`, "
        f"`gamma='{svm['gamma']}'`, `class_weight='{svm['class_weight']}'`, "
        f"`probability={svm['probability']}` (outputs class label and geometric hyperplane margin, never probability).\n"
        f"3. **Leakage exclusion:** `{', '.join(metadata['excluded_columns'])}` is excluded because "
        "model-checking duration is only available post-hoc after running formal verification.\n"
        f"4. **Dataset & Split:** {dataset['name']} (`{dataset['records']:,}` records, {dataset['license']}), "
        f"{split['method']} (`test_size={split['test_size']}`, `random_state={split['random_state']}`)."
    )


def main() -> None:
    st.set_page_config(
        page_title="Auction Outcome Feasibility Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css_mode, altair_mode = _render_sidebar_appearance()
    inject_astryx_theme(mode=css_mode)
    register_astryx_altair_theme(mode=altair_mode)
    theme_tokens = resolve_tokens(altair_mode)

    render_hero_header(
        "Auction Outcome Feasibility Predictor",
        "CIE 3 demonstration — supervised learning with one class-balanced RBF Support Vector Machine.",
        badges=[
            "UCI Auction Verification (n = 2,043)",
            "RBF SVM · C=1.0 · balanced",
            "Academic Surrogate Only",
        ],
    )
    st.info(f"**Scope limitation:** {SCOPE_STATEMENT}")

    bundle_path = os.environ.get("AUCTION_MODEL_BUNDLE", str(DEFAULT_BUNDLE_PATH))
    try:
        artifact, training_keys = get_resources(bundle_path)
        extras = get_bundle_extras(bundle_path)
        distribution = get_training_distribution(str(TRAINING_REFERENCE_PATH))
    except ArtifactError as err:
        st.error(str(err))
        st.stop()

    metadata = artifact.metadata
    _init_configuration_state(metadata)
    _render_sidebar_controls(metadata, artifact.model_sha256)

    active_values = _collect_active_values(metadata["feature_names"])
    active_frame = validate_configuration(active_values, metadata)
    active_margin = compute_decision_margin(artifact, active_frame)
    active_key = configuration_key(active_frame)

    metrics_payload = extras["metrics"]
    m = metrics_payload["metrics"]
    cm_info = metrics_payload["confusion_matrix"]
    render_kpi_strip(
        [
            {
                "label": "Held-out MCC (Primary)",
                "value": f"{m['mcc']:.4f}",
                "delta": "Primary class-imbalance metric",
                "tone": "info",
            },
            {
                "label": "Held-out Accuracy",
                "value": f"{m['accuracy']:.2%}",
                "delta": f"vs {metrics_payload['always_infeasible_accuracy']:.2%} Always-Infeasible Baseline",
                "tone": "neutral",
            },
            {
                "label": "Feasible Recall (Class 1)",
                "value": f"{m['recall_feasible']:.2%}",
                "delta": f"{cm_info['true_positive']}/{metrics_payload['test_feasible_records']} feasible test records",
                "tone": "success",
            },
            {
                "label": "Active Decision Margin",
                "value": f"{active_margin:+.4f}",
                "delta": "Feasible side (>= 0)" if active_margin >= 0 else "Infeasible side (< 0)",
                "tone": "success" if active_margin >= 0 else "warning",
            },
        ]
    )

    _render_configuration_studio(artifact, metadata, training_keys, active_key)

    st.divider()
    tab_price, tab_eval, tab_domain, tab_guide = st.tabs(
        [
            "Price Sensitivity & Hyperplane",
            "Model Evaluation & Baseline",
            "Training Domain Explorer",
            "CIE 3 Guide & Terminology",
        ]
    )
    with tab_price:
        _render_price_sensitivity_tab(artifact, active_values, training_keys, theme_tokens)
    with tab_eval:
        _render_evaluation_tab(extras, theme_tokens)
    with tab_domain:
        _render_domain_explorer_tab(distribution, active_values, theme_tokens)
    with tab_guide:
        _render_guide_tab(metadata)


if __name__ == "__main__":
    main()
