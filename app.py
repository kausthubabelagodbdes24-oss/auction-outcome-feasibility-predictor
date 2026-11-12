"""Focused Streamlit interface for one auction outcome feasibility prediction."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from charts import (
    build_domain_distribution_chart,
    build_metrics_comparison_chart,
    build_price_sensitivity_chart,
)
from dashboard_data import (
    compute_decision_margin,
    compute_price_sensitivity,
    load_bundle_extras,
    load_training_distribution,
)
from feasibility import (
    DEFAULT_BUNDLE_PATH,
    ArtifactError,
    InvalidConfigurationError,
    load_bundle,
    load_training_keys,
    predict_outcome,
    seen_during_training,
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
def get_resources(bundle_path: str):
    """Load the verified model and its matching training-configuration keys."""
    artifact = load_bundle(bundle_path)
    return artifact, load_training_keys(artifact.metadata)


@st.cache_data
def get_analysis_sources(bundle_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load held-out metrics and training distributions used by the dashboard."""
    return load_bundle_extras(bundle_path), load_training_distribution()


def _format_product(value: Any) -> str:
    return f"Product {int(value)}"


def _format_winner(value: Any) -> str:
    winner = int(value)
    return "Price only" if winner == 0 else f"Bidder {winner}"


def _select_feature(
    feature: str,
    metadata: dict[str, Any],
    *,
    format_func: Any | None = None,
) -> int:
    """Render one bounded selector from the model's recorded input domain."""
    options = metadata["input_domains"][feature]["observed_values"]
    kwargs: dict[str, Any] = {
        "label": FIELD_LABELS[feature],
        "options": options,
        "key": f"input_{feature}",
    }
    if format_func is not None:
        kwargs["format_func"] = format_func
    return int(st.selectbox(**kwargs))


def _render_form(metadata: dict[str, Any]) -> tuple[dict[str, int], bool]:
    """Render the seven model inputs as one compact, grouped form."""
    selected: dict[str, int] = {}
    feature_names = metadata["feature_names"]

    with st.form("auction_configuration_form", border=False):
        st.markdown("#### Bidder capacity")
        first_row = st.columns(2, gap="medium")
        second_row = st.columns(2, gap="medium")
        for column, feature in zip((*first_row, *second_row), feature_names[:4]):
            with column:
                selected[feature] = _select_feature(feature, metadata)

        st.markdown("#### Auction details")
        detail_columns = st.columns(3, gap="medium")
        detail_formatters = (None, _format_product, _format_winner)
        for column, feature, formatter in zip(
            detail_columns,
            feature_names[4:],
            detail_formatters,
        ):
            with column:
                selected[feature] = _select_feature(
                    feature,
                    metadata,
                    format_func=formatter,
                )

        submitted = st.form_submit_button(
            "Predict outcome",
            type="primary",
            use_container_width=True,
        )

    return selected, submitted


def _inject_page_styles() -> None:
    """Apply a restrained presentation layer around native Streamlit controls."""
    st.markdown(
        """
        <style>
        .stMainBlockContainer {
            max-width: 96rem;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        h1 {
            letter-spacing: -0.035em;
            line-height: 1.08;
            margin-bottom: 0.45rem;
        }
        h4 {
            color: var(--text-color);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-top: 1.4rem;
            opacity: 0.78;
            text-transform: uppercase;
        }
        h2 {
            margin-bottom: 0.75rem;
            margin-top: 2.75rem;
        }
        h3 {
            margin-bottom: 0.5rem;
            margin-top: 1.25rem;
        }
        [data-testid="stForm"] {
            margin-top: 1.5rem;
        }
        [data-testid="stFormSubmitButton"] {
            margin-top: 1rem;
        }
        [data-testid="stAlert"] {
            border-radius: 0.75rem;
            font-size: 1.05rem;
            margin-top: 1rem;
            padding-block: 0.85rem;
        }
        [data-testid="stCaptionContainer"] {
            line-height: 1.55;
            margin-bottom: 1rem;
        }
        [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            margin-top: 0.75rem;
        }
        button[data-baseweb="tab"] {
            height: auto;
            padding: 0.75rem 1.25rem;
        }
        [data-testid="stMetric"] {
            padding-block: 0.4rem;
        }
        .app-kicker {
            color: var(--primary-color);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
        }
        .app-intro {
            color: var(--text-color);
            font-size: 1rem;
            margin: 0;
            opacity: 0.68;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_analysis_dashboard(
    artifact: Any,
    training_keys: frozenset[str],
    values: dict[str, int],
    bundle_path: str,
) -> None:
    """Render presentation aids for the most recently predicted configuration."""
    frame = validate_configuration(values, artifact.metadata)
    prediction = predict_outcome(artifact.model, frame)
    margin = compute_decision_margin(artifact, frame)
    was_seen = seen_during_training(frame, training_keys)

    try:
        extras, distribution = get_analysis_sources(bundle_path)
    except ArtifactError as err:
        st.error(f"Prediction succeeded, but analysis data could not be loaded: {err}")
        return

    metrics_payload = extras["metrics"]
    metrics = metrics_payload["metrics"]

    st.markdown("#### Prediction")
    if prediction == 1:
        st.success("Feasible")
    else:
        st.warning("Infeasible")

    summary_columns = st.columns(4, gap="medium")
    summary_columns[0].metric("Outcome", "Feasible" if prediction == 1 else "Infeasible")
    summary_columns[1].metric("Decision margin", f"{margin:+.4f}")
    summary_columns[2].metric("Training membership", "Seen" if was_seen else "Novel")
    summary_columns[3].metric("Held-out MCC", f"{metrics['mcc']:.4f}")

    st.divider()
    st.subheader("Model analysis")
    st.caption("Presentation aids for the configuration evaluated above.")

    sensitivity_tab, evaluation_tab, distribution_tab = st.tabs(
        ["Price sensitivity", "Evaluation comparison", "Training distribution"]
    )

    with sensitivity_tab:
        with st.container(border=True):
            st.markdown("### Price sensitivity across the observed domain")
            st.caption(
                "The selected price moves across every observed value while the other six "
                "inputs remain fixed. Green and blue indicate feasible and infeasible estimates; "
                "circles and diamonds indicate seen and novel configurations."
            )
            sweep = compute_price_sensitivity(artifact, values, training_keys)
            selected_price = int(values["property.price"])
            chart = build_price_sensitivity_chart(sweep, selected_price=selected_price)
            st.altair_chart(chart, width="stretch", theme=None, key="price_sensitivity_chart")

            selected_row = sweep.loc[sweep["is_selected"]].iloc[0]
            feasible_count = int((sweep["prediction"] == 1).sum())
            st.caption(
                f"{feasible_count} of {len(sweep)} observed prices are predicted feasible. "
                f"At the selected price ({selected_price}), the raw SVM decision margin is "
                f"{selected_row['decision_margin']:+.4f}."
            )
            with st.expander("View price sweep data"):
                st.dataframe(
                    sweep.rename(
                        columns={
                            "price": "Price",
                            "outcome_label": "Prediction",
                            "decision_margin": "Decision margin",
                            "seen_during_training": "Seen in training",
                        }
                    )[["Price", "Prediction", "Decision margin", "Seen in training"]],
                    hide_index=True,
                    width="stretch",
                )

    with evaluation_tab:
        with st.container(border=True):
            st.markdown("### Held-out metric comparison")
            st.caption(
                "The class-balanced SVM is compared with an always-infeasible baseline. "
                "MCC is the primary metric because the held-out classes are imbalanced."
            )
            st.altair_chart(
                build_metrics_comparison_chart(metrics_payload),
                width="stretch",
                theme=None,
                key="metric_comparison_chart",
            )

        with st.container(border=True):
            st.markdown("### Error breakdown")
            st.caption(
                f"Error breakdown across the stratified test partition "
                f"(n = {metrics_payload['test_records']:,} records)."
            )
            image_left, image_center, image_right = st.columns([1, 1.35, 1])
            with image_center:
                st.image(extras["confusion_matrix_png"], width="stretch")
            confusion = metrics_payload["confusion_matrix"]
            matrix_columns = st.columns(4, gap="medium")
            matrix_columns[0].metric("True negative", f"{confusion['true_negative']:,}")
            matrix_columns[1].metric("False positive", f"{confusion['false_positive']:,}")
            matrix_columns[2].metric("False negative", f"{confusion['false_negative']:,}")
            matrix_columns[3].metric("True positive", f"{confusion['true_positive']:,}")

        baseline_accuracy = metrics_payload["always_infeasible_accuracy"]
        st.caption(
            f"The class-balanced SVM reaches {metrics['accuracy']:.2%} accuracy and "
            f"{metrics['mcc']:.4f} MCC. The always-infeasible baseline reaches "
            f"{baseline_accuracy:.2%} accuracy but 0 MCC, showing why accuracy alone is "
            "misleading for this imbalanced dataset."
        )

    with distribution_tab:
        with st.container(border=True):
            control_column, value_column = st.columns([2, 1], gap="large")
            with control_column:
                selected_feature = st.selectbox(
                    "Feature to inspect",
                    options=artifact.metadata["feature_names"],
                    index=4,
                    format_func=lambda feature: FIELD_LABELS[feature],
                    key="analysis_distribution_feature",
                )
            active_value = int(values[selected_feature])
            with value_column:
                st.metric("Selected configuration value", active_value)

            st.markdown(f"### Training distribution — {FIELD_LABELS[selected_feature]}")
            st.caption(
                f"Observed frequencies across {distribution['total_records']:,} training "
                "configurations. The selected configuration value is highlighted in orange."
            )
            feature_distribution = distribution["feature_frequencies"][selected_feature]
            st.altair_chart(
                build_domain_distribution_chart(
                    feature_distribution,
                    selected_feature,
                    selected_value=active_value,
                ),
                width="stretch",
                theme=None,
                key="training_distribution_chart",
            )
            active_row = feature_distribution.loc[
                feature_distribution["value"] == active_value
            ].iloc[0]
            st.caption(
                f"{FIELD_LABELS[selected_feature]} = {active_value} appears in "
                f"{int(active_row['count']):,} of {distribution['total_records']:,} training "
                f"records ({float(active_row['share']):.1%})."
            )


def main() -> None:
    st.set_page_config(
        page_title="Auction Outcome Feasibility Predictor",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_page_styles()

    st.markdown('<p class="app-kicker">SVM classifier</p>', unsafe_allow_html=True)
    st.title("Auction Outcome Feasibility Predictor")
    st.markdown(
        '<p class="app-intro">Enter an auction configuration to predict its outcome.</p>',
        unsafe_allow_html=True,
    )

    bundle_path = os.environ.get("AUCTION_MODEL_BUNDLE", str(DEFAULT_BUNDLE_PATH))
    try:
        artifact, training_keys = get_resources(bundle_path)
    except ArtifactError as err:
        st.error(str(err))
        return

    selected, submitted = _render_form(artifact.metadata)
    if submitted:
        try:
            frame = validate_configuration(selected, artifact.metadata)
            prediction = predict_outcome(artifact.model, frame)
        except (InvalidConfigurationError, ValueError) as err:
            st.session_state.pop("last_prediction", None)
            st.error(str(err))
        else:
            st.session_state["last_prediction"] = {
                "values": selected,
                "prediction": prediction,
            }

    result = st.session_state.get("last_prediction")
    if result is not None:
        _render_analysis_dashboard(
            artifact,
            training_keys,
            result["values"],
            bundle_path,
        )


if __name__ == "__main__":
    main()
