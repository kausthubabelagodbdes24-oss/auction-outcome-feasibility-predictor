"""Comprehensive tests for dashboard_data, theme_astryx, charts, and upgraded app.py flows."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from charts import (
    ASTRYX_BLUE,
    ASTRYX_BLUE_RAMP,
    ASTRYX_GREEN,
    ASTRYX_ORANGE,
    build_confusion_matrix_chart,
    build_domain_distribution_chart,
    build_metrics_comparison_chart,
    build_price_sensitivity_chart,
)
from dashboard_data import (
    CONFUSION_MATRIX_MEMBER,
    DEMO_PRESETS,
    METRICS_MEMBER,
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
    REQUIRED_MEMBERS,
    SEEN_MESSAGE,
    TRAINING_REFERENCE_PATH,
    ArtifactError,
    configuration_key,
    load_bundle,
    load_training_keys,
    predict_outcome,
    seen_during_training,
    validate_configuration,
)
from theme_astryx import (
    ASTRYX_DARK,
    ASTRYX_DATA_CATEGORICAL,
    ASTRYX_LIGHT,
    build_astryx_css,
    get_sequential_ramp,
    inject_astryx_theme,
    register_astryx_altair_theme,
    render_hero_header,
    render_kpi_strip,
    render_prediction_card,
    render_section_header,
    render_terminology_grid,
    resolve_tokens,
)


def _bundle_path() -> Path:
    return Path(os.environ.get("AUCTION_MODEL_BUNDLE", str(DEFAULT_BUNDLE_PATH)))


@pytest.fixture(scope="module")
def artifact():
    return load_bundle(_bundle_path())


@pytest.fixture(scope="module")
def training_keys(artifact):
    return load_training_keys(artifact.metadata, TRAINING_REFERENCE_PATH)


@pytest.fixture(scope="module")
def bundle_extras():
    return load_bundle_extras(_bundle_path())


@pytest.fixture(scope="module")
def training_distribution():
    return load_training_distribution(TRAINING_REFERENCE_PATH)


def test_demo_presets_validity_and_outcomes(artifact, training_keys):
    assert len(DEMO_PRESETS) == 4
    expected_outcomes = {
        "Seen Infeasible (0,3,1,0,60,4,0)": (0, True),
        "Seen Feasible (0,0,2,1,59,1,0)": (1, True),
        "Novel Feasible (0,3,2,1,77,2,0)": (1, False),
        "Novel Infeasible (1,3,2,0,70,2,0)": (0, False),
    }
    for preset in DEMO_PRESETS:
        label = preset["label"]
        assert set(preset["values"].keys()) == set(EXPECTED_FEATURES)
        frame = validate_configuration(preset["values"], artifact.metadata)
        pred = predict_outcome(artifact.model, frame)
        margin = compute_decision_margin(artifact, frame)
        seen = seen_during_training(frame, training_keys)
        exp_pred, exp_seen = expected_outcomes[label]
        assert pred == exp_pred
        assert seen is exp_seen
        assert (margin >= 0.0) == (pred == 1)


def test_load_bundle_extras_and_errors(tmp_path: Path):
    extras = load_bundle_extras(_bundle_path())
    assert extras["metrics"]["primary_metric"] == "mcc"
    assert extras["metrics"]["metrics"]["mcc"] == pytest.approx(0.5725)
    assert extras["metrics"]["always_infeasible_accuracy"] == pytest.approx(0.8729)
    assert extras["metrics"]["confusion_matrix"]["matrix"] == [[304, 53], [6, 46]]
    assert extras["confusion_matrix_png"].startswith(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ArtifactError, match="not found"):
        load_bundle_extras(tmp_path / "missing.zip")

    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not a valid ZIP"):
        load_bundle_extras(bad_zip)

    bad_png_bundle = tmp_path / "bad_png.zip"
    with zipfile.ZipFile(_bundle_path()) as src, zipfile.ZipFile(bad_png_bundle, "w") as dst:
        for member in REQUIRED_MEMBERS:
            data = b"not-a-png" if member == CONFUSION_MATRIX_MEMBER else src.read(member)
            dst.writestr(member, data)
    with pytest.raises(ArtifactError, match="valid PNG"):
        load_bundle_extras(bad_png_bundle)

    bad_metrics_bundle = tmp_path / "bad_metrics.zip"
    with zipfile.ZipFile(_bundle_path()) as src, zipfile.ZipFile(bad_metrics_bundle, "w") as dst:
        for member in REQUIRED_MEMBERS:
            data = b"{}" if member == METRICS_MEMBER else src.read(member)
            dst.writestr(member, data)
    with pytest.raises(ArtifactError, match="must contain 'metrics'"):
        load_bundle_extras(bad_metrics_bundle)


def test_load_training_distribution_and_errors(tmp_path: Path, training_distribution):
    assert training_distribution["total_records"] == 1634
    assert set(training_distribution["feature_frequencies"].keys()) == set(EXPECTED_FEATURES)
    for feature, df in training_distribution["feature_frequencies"].items():
        assert list(df.columns) == ["feature", "value", "count", "share"]
        assert df["count"].sum() == 1634
        assert df["share"].sum() == pytest.approx(1.0)
        assert (df["feature"] == feature).all()

    stats = training_distribution["price_stats"]
    assert stats["min"] == 59
    assert stats["max"] == 90
    assert stats["median"] == pytest.approx(70.0)
    assert 70.0 < stats["mean"] < 73.0

    bad_ref = tmp_path / "bad_ref.json"
    bad_ref.write_text(json.dumps({"feature_names": ["wrong"]}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="feature_names mismatch"):
        load_training_distribution(bad_ref)


def test_compute_decision_margin_and_price_sensitivity(artifact, training_keys):
    novel_feasible = DEMO_PRESETS[2]["values"]
    frame = validate_configuration(novel_feasible, artifact.metadata)
    margin = compute_decision_margin(artifact, frame)
    assert margin == pytest.approx(float(artifact.model.decision_function(frame)[0]))
    assert margin > 0.0

    with pytest.raises(ValueError, match="one-row DataFrame"):
        compute_decision_margin(artifact, pd.concat([frame, frame], ignore_index=True))

    sweep = compute_price_sensitivity(artifact, novel_feasible, training_keys)
    assert len(sweep) == 32
    assert list(sweep.columns) == [
        "price",
        "prediction",
        "outcome_label",
        "decision_margin",
        "seen_during_training",
        "is_selected",
    ]
    assert int(sweep["is_selected"].sum()) == 1
    assert int(sweep.loc[sweep["is_selected"], "price"].iloc[0]) == 77
    assert (sweep["prediction"] == 1).any() and (sweep["prediction"] == 0).any()


def test_theme_astryx_tokens_css_altair_and_html_escaping():
    assert resolve_tokens("light") == ASTRYX_LIGHT
    assert resolve_tokens("dark") == ASTRYX_DARK
    with pytest.raises(ValueError):
        resolve_tokens("sepia")

    ramp_low = get_sequential_ramp("blue", mode="light", low_to_high=True)
    assert ramp_low == ASTRYX_BLUE_RAMP
    gray_dark = get_sequential_ramp("gray", mode="dark")
    assert len(gray_dark) == 5

    css_auto = build_astryx_css("auto")
    css_light = build_astryx_css("light")
    css_dark = build_astryx_css("dark")
    assert "light-dark(" in css_auto
    assert ASTRYX_LIGHT["background_body"] in css_light and "light-dark(" not in css_light
    assert ASTRYX_DARK["background_surface"] in css_dark and "light-dark(" not in css_dark
    assert "Figtree" in css_auto
    with pytest.raises(ValueError):
        build_astryx_css("invalid")

    theme_name = register_astryx_altair_theme("light")
    assert theme_name == "astryx_neutral"
    assert alt.theme.active == "astryx_neutral"
    cfg = alt.theme.get()()
    assert cfg["background"] == "transparent"
    assert cfg["config"]["axisX"]["grid"] is False
    assert cfg["config"]["axisY"]["grid"] is True
    assert cfg["config"]["bar"]["cornerRadiusEnd"] == 4
    assert cfg["config"]["range"]["category"] == list(ASTRYX_DATA_CATEGORICAL.values())

    def _render_widget_harness():
        from theme_astryx import (
            inject_astryx_theme,
            render_hero_header,
            render_kpi_strip,
            render_prediction_card,
            render_section_header,
            render_terminology_grid,
        )

        inject_astryx_theme("light")
        render_hero_header("Title <script>alert(1)</script>", "Sub & text", ["Badge <b>1</b>"])
        render_kpi_strip([{"label": "MCC", "value": "0.5725", "delta": "Primary", "tone": "info"}])
        render_section_header("Sec", "Sub")
        render_prediction_card("Feasible model estimate", True, False, 0.279, "0|3|2|1|77|2|0")
        render_terminology_grid([("Term", "Def", "Avoid <tag>")])

    at = AppTest.from_function(_render_widget_harness).run()
    assert not at.exception
    joined = "\n".join(m.value for m in at.markdown)
    assert "<script>" not in joined
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in joined
    assert "&lt;b&gt;1&lt;/b&gt;" in joined


def test_charts_builders_light_and_dark(artifact, training_keys, bundle_extras, training_distribution):
    sweep = compute_price_sensitivity(artifact, DEMO_PRESETS[2]["values"], training_keys)
    light_tokens = resolve_tokens("light")
    dark_tokens = resolve_tokens("dark")

    for tokens in (light_tokens, dark_tokens):
        c1 = build_price_sensitivity_chart(sweep, selected_price=77, theme_tokens=tokens).to_dict()
        c2 = build_confusion_matrix_chart(bundle_extras["metrics"], theme_tokens=tokens).to_dict()
        c3 = build_metrics_comparison_chart(bundle_extras["metrics"], theme_tokens=tokens).to_dict()
        c4 = build_domain_distribution_chart(
            training_distribution["feature_frequencies"]["property.price"],
            "property.price",
            selected_value=77,
            theme_tokens=tokens,
        ).to_dict()
        for spec in (c1, c2, c3, c4):
            assert spec["background"] == "transparent"
            assert "layer" in spec

    c4_none = build_domain_distribution_chart(
        training_distribution["feature_frequencies"]["process.b1.capacity"],
        "process.b1.capacity",
        selected_value=None,
    ).to_dict()
    assert c4_none["background"] == "transparent"
    assert ASTRYX_BLUE in str(c4_none)
    assert ASTRYX_ORANGE in str(build_domain_distribution_chart(
        training_distribution["feature_frequencies"]["process.b1.capacity"],
        "process.b1.capacity",
        selected_value=0,
    ).to_dict())
    assert ASTRYX_GREEN in str(build_price_sensitivity_chart(sweep, selected_price=77).to_dict())

    with pytest.raises(ValueError):
        build_price_sensitivity_chart(pd.DataFrame(), selected_price=70)
    with pytest.raises(ValueError):
        build_confusion_matrix_chart({})
    with pytest.raises(ValueError):
        build_metrics_comparison_chart({})
    with pytest.raises(ValueError):
        build_domain_distribution_chart(pd.DataFrame(), "property.price")


def test_app_full_dashboard_flows_and_terminology():
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception

    # Scope banner and selectbox/button contracts
    assert any("not formal verification" in info.value for info in at.info)
    assert len(at.selectbox) == 7
    for sb, feature in zip(at.selectbox, EXPECTED_FEATURES):
        assert feature in sb.label
    assert at.button[0].label == "Predict Outcome Feasibility"

    # KPI strip & tabs rendered
    all_md = "\n".join(m.value for m in at.markdown)
    assert "0.5725" in all_md
    assert "85.57%" in all_md
    assert "88.46%" in all_md
    assert [t.label for t in at.tabs] == [
        "Price Sensitivity & Hyperplane",
        "Model Evaluation & Baseline",
        "Training Domain Explorer",
        "CIE 3 Guide & Terminology",
    ]

    # Sidebar Appearance control switches CSS tokens cleanly
    assert len(at.sidebar.segmented_control) == 1
    at.sidebar.segmented_control[0].select("Dark").run()
    assert not at.exception
    dark_md = "\n".join(m.value for m in at.markdown)
    assert ASTRYX_DARK["background_surface"] in dark_md

    at.sidebar.segmented_control[0].select("Light").run()
    assert not at.exception
    light_md = "\n".join(m.value for m in at.markdown)
    assert ASTRYX_LIGHT["background_body"] in light_md

    # Preset selection populates the 7 selectboxes and submits to the expected prediction
    novel_feasible_preset = DEMO_PRESETS[2]["label"]
    at.sidebar.pills[0].select(novel_feasible_preset).run()
    assert not at.exception
    assert [sb.value for sb in at.selectbox] == [0, 3, 2, 1, 77, 2, 0]

    at.button[0].click().run()
    assert not at.exception
    assert any(CLASS_LABELS[1] in s.value for s in at.success)
    assert any(NOVEL_MESSAGE in m.value for m in at.markdown)
    assert any("0|3|2|1|77|2|0" in m.value for m in at.markdown)

    # Switch to Seen Infeasible preset and re-submit
    seen_infeasible_preset = DEMO_PRESETS[0]["label"]
    at.sidebar.pills[0].select(seen_infeasible_preset).run()
    assert not at.exception
    assert [sb.value for sb in at.selectbox] == [0, 3, 1, 0, 60, 4, 0]

    at.button[0].click().run()
    assert not at.exception
    assert any(CLASS_LABELS[0] in w.value for w in at.warning)
    assert any(SEEN_MESSAGE in m.value for m in at.markdown)
    assert any("0|3|1|0|60|4|0" in m.value for m in at.markdown)

    # Training Domain Explorer feature switcher (pills, NOT selectbox)
    assert len(at.main.pills) == 1
    at.main.pills[0].select("process.b1.capacity").run()
    assert not at.exception
    assert len(at.selectbox) == 7
    assert any("process.b1.capacity = 0" in m.value for m in at.markdown)

    # Strict terminology audit across rendered text (outside scope negation & Avoid column)
    rendered_text_blocks = [
        *[m.value for m in at.markdown if "astryx-term-list" not in m.value],
        *[c.value for c in at.caption],
        *[s.value for s in at.success],
        *[w.value for w in at.warning],
    ]
    joined_lower = "\n".join(rendered_text_blocks).lower()
    for forbidden in (
        "valid auction",
        "verified auction",
        "real auction",
        "real-world auction result",
        "verified transaction",
        "fraud detection",
        "legal verdict",
        "calibrated probability",
    ):
        assert forbidden not in joined_lower


def test_app_missing_bundle_renders_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AUCTION_MODEL_BUNDLE", str(tmp_path / "does_not_exist.zip"))
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception
    assert any("Model bundle not found" in e.value for e in at.error)
