"""Focused tests for the post-prediction model-analysis dashboard."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from charts import (
    build_domain_distribution_chart,
    build_metrics_comparison_chart,
    build_price_sensitivity_chart,
)
from dashboard_data import (
    compute_price_sensitivity,
    load_bundle_extras,
    load_training_distribution,
)
from feasibility import DEFAULT_BUNDLE_PATH, load_bundle, load_training_keys


def test_analysis_data_and_charts_match_the_model_artifact():
    artifact = load_bundle(DEFAULT_BUNDLE_PATH)
    training_keys = load_training_keys(artifact.metadata)
    extras = load_bundle_extras(DEFAULT_BUNDLE_PATH)
    distribution = load_training_distribution()
    values = {
        "process.b1.capacity": 0,
        "process.b2.capacity": 3,
        "process.b3.capacity": 2,
        "process.b4.capacity": 1,
        "property.price": 77,
        "property.product": 2,
        "property.winner": 0,
    }

    sweep = compute_price_sensitivity(artifact, values, training_keys)
    assert len(sweep) == 32
    assert int(sweep["is_selected"].sum()) == 1
    assert distribution["total_records"] == 1634
    assert extras["metrics"]["metrics"]["mcc"] == 0.5725

    chart_specs = [
        build_price_sensitivity_chart(sweep, selected_price=77).to_dict(),
        build_metrics_comparison_chart(extras["metrics"]).to_dict(),
        build_domain_distribution_chart(
            distribution["feature_frequencies"]["property.price"],
            "property.price",
            selected_value=77,
        ).to_dict(),
    ]
    assert all("layer" in spec for spec in chart_specs)
    assert extras["confusion_matrix_png"].startswith(b"\x89PNG\r\n\x1a\n")
