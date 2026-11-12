"""Astryx Neutral Altair chart builders for the Auction Outcome Feasibility dashboard.

All visual terminology follows CONTEXT.md: this module visualizes academic
SVM surrogate estimates (decision hyperplane margin, held-out test metrics,
and training-domain distributions) and never frames outputs as formal
verification, legal validity, fraud detection, or calibrated probability.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

# Exact Astryx Neutral data-visualization palette tokens
ASTRYX_BLUE = "#0171E3"
ASTRYX_ORANGE = "#EB6E00"
ASTRYX_PURPLE = "#6B1EFD"
ASTRYX_GREEN = "#0B991F"
ASTRYX_DATA_NEUTRAL = "#8494A3"

ASTRYX_FONT_STACK = (
    "Figtree, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)

_DEFAULT_LIGHT_TOKENS = {
    "text_primary": "#000000",
    "text_secondary": "#474747",
    "grid_color": "#00000014",
    "surface_color": "#FFFFFF",
    "data_neutral": ASTRYX_DATA_NEUTRAL,
    "font": ASTRYX_FONT_STACK,
}

_FEASIBLE_LABEL = "Feasible model estimate"
_INFEASIBLE_LABEL = "Infeasible model estimate"


def _resolve_tokens(theme_tokens: dict[str, Any] | None) -> dict[str, str]:
    """Resolve light/dark surface and typography colors from an optional token dict."""
    resolved = dict(_DEFAULT_LIGHT_TOKENS)
    if not theme_tokens:
        return resolved

    key_map = {
        "text_primary": ("text_primary", "--color-text-primary"),
        "text_secondary": ("text_secondary", "--color-text-secondary"),
        "grid_color": ("grid_color", "border", "--color-border"),
        "surface_color": ("surface_color", "background_card", "background", "--color-background-card"),
        "data_neutral": ("data_neutral", "--color-data-neutral"),
        "font": ("font", "font_family_body", "--font-family-body"),
    }
    for target_key, candidates in key_map.items():
        for candidate in candidates:
            value = theme_tokens.get(candidate)
            if isinstance(value, str) and value.strip():
                resolved[target_key] = value.strip()
                break
    return resolved


def _apply_astryx_chrome(
    chart: alt.TopLevelMixin,
    tokens: dict[str, str],
    *,
    height: int = 360,
    title: str | alt.TitleParams | None = None,
) -> alt.Chart | alt.LayerChart:
    """Apply shared Astryx Neutral chrome, typography, and transparent canvas."""
    props: dict[str, Any] = {
        "background": "transparent",
        "height": height,
        "width": "container",
    }
    if title is not None:
        props["title"] = title

    return (
        chart.properties(**props)
        .configure_view(strokeWidth=0)
        .configure_axis(
            domain=False,
            ticks=False,
            labelFont=tokens["font"],
            titleFont=tokens["font"],
            labelFontSize=13,
            titleFontSize=13,
            titleFontWeight=500,
            titlePadding=14,
            labelPadding=8,
            labelColor=tokens["text_secondary"],
            titleColor=tokens["text_secondary"],
            gridColor=tokens["grid_color"],
        )
        .configure_legend(
            labelFont=tokens["font"],
            titleFont=tokens["font"],
            labelFontSize=13,
            titleFontSize=13,
            titleFontWeight=600,
            labelColor=tokens["text_secondary"],
            titleColor=tokens["text_primary"],
            orient="top",
            direction="horizontal",
        )
        .configure_title(
            font=tokens["font"],
            fontSize=16,
            fontWeight=600,
            color=tokens["text_primary"],
            subtitleFont=tokens["font"],
            subtitleFontSize=13,
            subtitleColor=tokens["text_secondary"],
            anchor="start",
        )
    )


def build_price_sensitivity_chart(
    sweep_df: pd.DataFrame,
    selected_price: int,
    theme_tokens: dict[str, Any] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Build an Altair chart of SVM decision margin across observed property prices.

    Expects `sweep_df` with `price` and `decision_margin`, and optionally
    `outcome_label` (or `prediction`) and `seen_during_training`.
    """
    if sweep_df.empty:
        raise ValueError("sweep_df must contain at least one row.")
    for col in ("price", "decision_margin"):
        if col not in sweep_df.columns:
            raise ValueError(f"sweep_df is missing required column '{col}'.")

    tokens = _resolve_tokens(theme_tokens)
    df = sweep_df.copy()
    df["price"] = df["price"].astype(int)
    df["decision_margin"] = df["decision_margin"].astype(float)
    df = df.sort_values("price").reset_index(drop=True)

    if "outcome_label" not in df.columns:
        if "prediction" in df.columns:
            df["outcome_label"] = df["prediction"].map(
                {1: _FEASIBLE_LABEL, 0: _INFEASIBLE_LABEL}
            )
        else:
            df["outcome_label"] = df["decision_margin"].apply(
                lambda m: _FEASIBLE_LABEL if m >= 0 else _INFEASIBLE_LABEL
            )

    if "seen_during_training" in df.columns:
        df["novelty_label"] = df["seen_during_training"].map(
            {True: "In-domain (seen in training)", False: "Novel configuration"}
        ).fillna("Novel configuration")
    else:
        df["novelty_label"] = "Novel configuration"

    df["margin_pos"] = df["decision_margin"].clip(lower=0.0)
    df["margin_neg"] = df["decision_margin"].clip(upper=0.0)
    df["zero_baseline"] = 0.0

    selected_int = int(selected_price)
    df["is_selected"] = df["price"] == selected_int

    min_price = int(df["price"].min())
    max_price = int(df["price"].max())
    y_min = min(float(df["decision_margin"].min()), -0.25)
    y_max = max(float(df["decision_margin"].max()), 0.25)
    y_pad = max((y_max - y_min) * 0.12, 0.2)
    y_domain = [round(y_min - y_pad, 3), round(y_max + y_pad, 3)]

    x_encoding = alt.X(
        "price:Q",
        title="Property Price (property.price)",
        scale=alt.Scale(domain=[min_price - 0.5, max_price + 0.5], nice=False),
        axis=alt.Axis(grid=False, format="d", tickCount=16),
    )
    y_encoding = alt.Y(
        "decision_margin:Q",
        title="SVM Decision Margin (not probability)",
        scale=alt.Scale(domain=y_domain),
        axis=alt.Axis(grid=True),
    )

    pos_area = (
        alt.Chart(df)
        .mark_area(color=ASTRYX_GREEN, opacity=0.14)
        .encode(
            x=x_encoding,
            y=alt.Y("margin_pos:Q", scale=alt.Scale(domain=y_domain)),
            y2="zero_baseline:Q",
        )
    )
    neg_area = (
        alt.Chart(df)
        .mark_area(color=ASTRYX_BLUE, opacity=0.10)
        .encode(
            x=x_encoding,
            y=alt.Y("margin_neg:Q", scale=alt.Scale(domain=y_domain)),
            y2="zero_baseline:Q",
        )
    )

    boundary_label_x = min_price + max((max_price - min_price) * 0.10, 2.0)
    boundary_df = pd.DataFrame(
        [{"y": 0.0, "label": "Hyperplane Boundary (margin = 0)", "x": boundary_label_x}]
    )
    boundary_rule = (
        alt.Chart(boundary_df)
        .mark_rule(
            color=tokens["data_neutral"],
            strokeDash=[6, 4],
            strokeWidth=1.5,
        )
        .encode(y="y:Q")
    )
    boundary_label = (
        alt.Chart(boundary_df)
        .mark_text(
            align="left",
            baseline="bottom",
            dx=4,
            dy=-5,
            font=tokens["font"],
            fontSize=11,
            fontWeight=500,
            color=tokens["data_neutral"],
        )
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    margin_line = (
        alt.Chart(df)
        .mark_line(color=tokens["data_neutral"], strokeWidth=2)
        .encode(x=x_encoding, y=y_encoding)
    )

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=72, stroke=tokens["surface_color"], strokeWidth=1)
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=alt.Color(
                "outcome_label:N",
                scale=alt.Scale(
                    domain=[_FEASIBLE_LABEL, _INFEASIBLE_LABEL],
                    range=[ASTRYX_GREEN, ASTRYX_BLUE],
                ),
                legend=None,
            ),
            shape=alt.Shape(
                "novelty_label:N",
                scale=alt.Scale(
                    domain=["In-domain (seen in training)", "Novel configuration"],
                    range=["circle", "diamond"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("price:Q", title="Property Price", format="d"),
                alt.Tooltip("outcome_label:N", title="Model Estimate"),
                alt.Tooltip("decision_margin:Q", title="SVM Decision Margin", format="+.4f"),
                alt.Tooltip("novelty_label:N", title="Training Coverage"),
            ],
        )
    )

    layers: list[alt.TopLevelMixin] = [
        pos_area,
        neg_area,
        boundary_rule,
        boundary_label,
        margin_line,
        points,
    ]

    selected_rows = df[df["is_selected"]].copy()
    if not selected_rows.empty:
        sel_margin = float(selected_rows["decision_margin"].iloc[0])
        selected_rows["selected_caption"] = (
            f"Selected price {selected_int}: {sel_margin:+.3f}"
        )
        near_right = selected_int >= (min_price + (max_price - min_price) * 0.7)
        near_top = sel_margin >= (y_domain[1] - (y_domain[1] - y_domain[0]) * 0.18)

        selected_vline = (
            alt.Chart(selected_rows)
            .mark_rule(color=ASTRYX_ORANGE, strokeDash=[3, 3], strokeWidth=1.5, opacity=0.7)
            .encode(x="price:Q")
        )
        selected_marker = (
            alt.Chart(selected_rows)
            .mark_point(
                shape="circle",
                filled=True,
                size=190,
                color=ASTRYX_ORANGE,
                stroke=tokens["surface_color"],
                strokeWidth=2.5,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=[
                    alt.Tooltip("price:Q", title="Selected Price", format="d"),
                    alt.Tooltip("outcome_label:N", title="Model Estimate"),
                    alt.Tooltip("decision_margin:Q", title="SVM Decision Margin", format="+.4f"),
                    alt.Tooltip("novelty_label:N", title="Training Coverage"),
                ],
            )
        )
        selected_label = (
            alt.Chart(selected_rows)
            .mark_text(
                align="right" if near_right else "left",
                baseline="top" if near_top else "bottom",
                dx=-10 if near_right else 10,
                dy=12 if near_top else -12,
                font=tokens["font"],
                fontSize=12,
                fontWeight=600,
                color=ASTRYX_ORANGE,
            )
            .encode(x=x_encoding, y=y_encoding, text="selected_caption:N")
        )
        layers.extend([selected_vline, selected_marker, selected_label])

    layered = alt.layer(*layers)
    return _apply_astryx_chrome(
        layered,
        tokens,
        height=390,
    )


def build_metrics_comparison_chart(
    metrics_dict: dict[str, Any],
    theme_tokens: dict[str, Any] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Build a horizontal Astryx comparison bar chart of held-out metrics vs baseline."""
    m = metrics_dict.get("metrics", metrics_dict)
    required = ("accuracy", "mcc", "recall_feasible", "f1_feasible", "precision_feasible")
    missing = [k for k in required if k not in m]
    if missing:
        raise ValueError(f"metrics_dict is missing required metrics: {missing}")

    baseline = float(metrics_dict.get("always_infeasible_accuracy", 0.0))
    tokens = _resolve_tokens(theme_tokens)

    rows = [
        {
            "metric": "Feasible Recall",
            "value": float(m["recall_feasible"]),
            "category": "Feasible Class",
            "color": ASTRYX_GREEN,
            "order": 1,
        },
        {
            "metric": "Always-Infeasible Baseline",
            "value": baseline,
            "category": "Reference Baseline",
            "color": tokens["data_neutral"],
            "order": 2,
        },
        {
            "metric": "Held-out Accuracy",
            "value": float(m["accuracy"]),
            "category": "Overall Partition",
            "color": ASTRYX_BLUE,
            "order": 3,
        },
        {
            "metric": "Feasible F1",
            "value": float(m["f1_feasible"]),
            "category": "Feasible Class",
            "color": ASTRYX_BLUE,
            "order": 4,
        },
        {
            "metric": "Held-out MCC (Primary)",
            "value": float(m["mcc"]),
            "category": "Primary Metric",
            "color": ASTRYX_PURPLE,
            "order": 5,
        },
        {
            "metric": "Feasible Precision",
            "value": float(m["precision_feasible"]),
            "category": "Feasible Class",
            "color": ASTRYX_ORANGE,
            "order": 6,
        },
    ]
    df = pd.DataFrame(rows)
    df["value_label"] = df["value"].map(lambda v: f"{v:.4f}")
    sort_order = [r["metric"] for r in sorted(rows, key=lambda item: item["order"])]

    x_scale = alt.Scale(domain=[0.0, 1.06])
    bars = (
        alt.Chart(df)
        .mark_bar(
            height=22,
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
        )
        .encode(
            x=alt.X(
                "value:Q",
                title="Score (0.00 to 1.00)",
                scale=x_scale,
                axis=alt.Axis(grid=True, format=".2f"),
            ),
            y=alt.Y(
                "metric:N",
                title=None,
                sort=sort_order,
                axis=alt.Axis(grid=False),
            ),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("category:N", title="Group"),
                alt.Tooltip("value:Q", title="Score", format=".4f"),
            ],
        )
    )

    baseline_rule_df = pd.DataFrame([{"baseline": baseline}])
    baseline_rule = (
        alt.Chart(baseline_rule_df)
        .mark_rule(
            color=tokens["data_neutral"],
            strokeDash=[4, 4],
            strokeWidth=1.5,
            opacity=0.8,
        )
        .encode(x=alt.X("baseline:Q", scale=x_scale))
    )

    labels = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=6,
            font=tokens["font"],
            fontSize=12,
            fontWeight=600,
            color=tokens["text_primary"],
        )
        .encode(
            x=alt.X("value:Q", scale=x_scale),
            y=alt.Y("metric:N", sort=sort_order),
            text="value_label:N",
        )
    )

    layered = alt.layer(baseline_rule, bars, labels)
    return _apply_astryx_chrome(
        layered,
        tokens,
        height=340,
    )


def build_cv_stability_chart(
    repeated_cv: dict[str, Any],
    theme_tokens: dict[str, Any] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Build a mean ± standard-deviation chart for repeated CV metrics."""
    metrics = repeated_cv.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("repeated_cv must contain a metrics object.")

    labels = {
        "mcc": "MCC (Primary)",
        "accuracy": "Accuracy",
        "precision_feasible": "Feasible Precision",
        "recall_feasible": "Feasible Recall",
        "f1_feasible": "Feasible F1",
    }
    missing = [name for name in labels if name not in metrics]
    if missing:
        raise ValueError(f"repeated_cv metrics are missing: {missing}")

    rows = []
    for order, (name, label) in enumerate(labels.items(), start=1):
        mean = float(metrics[name]["mean"])
        std = float(metrics[name]["std"])
        rows.append(
            {
                "metric": label,
                "mean": mean,
                "lower": max(-1.0 if name == "mcc" else 0.0, mean - std),
                "upper": min(1.0, mean + std),
                "mean_label": f"{mean:.4f} ± {std:.4f}",
                "order": order,
                "color": ASTRYX_PURPLE if name == "mcc" else ASTRYX_BLUE,
            }
        )

    df = pd.DataFrame(rows)
    tokens = _resolve_tokens(theme_tokens)
    sort_order = [row["metric"] for row in rows]
    x_scale = alt.Scale(domain=[-0.05, 1.05])

    intervals = (
        alt.Chart(df)
        .mark_rule(strokeWidth=5, opacity=0.35)
        .encode(
            x=alt.X("lower:Q", title="Cross-validation score", scale=x_scale),
            x2="upper:Q",
            y=alt.Y("metric:N", title=None, sort=sort_order),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("lower:Q", title="Mean - SD", format=".4f"),
                alt.Tooltip("upper:Q", title="Mean + SD", format=".4f"),
            ],
        )
    )
    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=150, stroke=tokens["surface_color"], strokeWidth=2)
        .encode(
            x=alt.X("mean:Q", scale=x_scale),
            y=alt.Y("metric:N", sort=sort_order),
            color=alt.Color("color:N", scale=None, legend=None),
        )
    )
    values = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=12,
            font=tokens["font"],
            fontSize=13,
            fontWeight=600,
            color=tokens["text_primary"],
        )
        .encode(
            x=alt.X("upper:Q", scale=x_scale),
            y=alt.Y("metric:N", sort=sort_order),
            text="mean_label:N",
        )
    )
    return _apply_astryx_chrome(
        alt.layer(intervals, points, values),
        tokens,
        height=300,
    )


def build_domain_distribution_chart(
    feature_df: pd.DataFrame,
    feature_name: str,
    selected_value: int | None = None,
    theme_tokens: dict[str, Any] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Build a training-record distribution bar chart highlighting `selected_value`."""
    if feature_df.empty:
        raise ValueError("feature_df must contain at least one row.")
    for col in ("value", "count"):
        if col not in feature_df.columns:
            raise ValueError(f"feature_df is missing required column '{col}'.")

    tokens = _resolve_tokens(theme_tokens)
    if "feature" in feature_df.columns:
        df = feature_df[feature_df["feature"] == feature_name].copy()
        if df.empty:
            raise ValueError(f"No rows found in feature_df for '{feature_name}'.")
    else:
        df = feature_df.copy()

    df["value"] = df["value"].astype(int)
    df["count"] = df["count"].astype(int)
    df = df.sort_values("value").reset_index(drop=True)

    total_records = int(df["count"].sum())
    if "share" not in df.columns:
        df["share"] = df["count"] / max(total_records, 1)
    else:
        df["share"] = df["share"].astype(float)

    df["value_label"] = df["value"].astype(str)
    sorted_labels = df["value_label"].tolist()

    selected_int = int(selected_value) if selected_value is not None else None
    if selected_int is not None:
        df["selection_state"] = df["value"].apply(
            lambda v: "Selected value" if v == selected_int else "Observed training values"
        )
        color_encoding = alt.Color(
            "selection_state:N",
            title=None,
            scale=alt.Scale(
                domain=["Selected value", "Observed training values"],
                range=[ASTRYX_ORANGE, ASTRYX_BLUE],
            ),
        )
    else:
        df["selection_state"] = "Observed training values"
        color_encoding = alt.Color(
            "selection_state:N",
            title=None,
            scale=alt.Scale(
                domain=["Observed training values"],
                range=[ASTRYX_BLUE],
            ),
            legend=None,
        )

    many_values = len(df) > 12
    x_axis = alt.Axis(
        grid=False,
        labelAngle=0,
        labelOverlap="greedy" if many_values else False,
    )
    y_max = int(df["count"].max())
    y_scale = alt.Scale(domain=[0, max(int(y_max * 1.14), y_max + 5)])

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "value_label:N",
                title="Observed Value",
                sort=sorted_labels,
                axis=x_axis,
            ),
            y=alt.Y(
                "count:Q",
                title="Training Records",
                scale=y_scale,
                axis=alt.Axis(grid=True, format=",d"),
            ),
            color=color_encoding,
            tooltip=[
                alt.Tooltip("value_label:N", title=feature_name),
                alt.Tooltip("count:Q", title="Training Records", format=","),
                alt.Tooltip("share:Q", title="Share of Training Set", format=".1%"),
                alt.Tooltip("selection_state:N", title="Selection"),
            ],
        )
    )

    layers: list[alt.TopLevelMixin] = [bars]
    if not many_values:
        count_labels = (
            alt.Chart(df)
            .mark_text(
                dy=-7,
                font=tokens["font"],
                fontSize=11,
                fontWeight=600,
                color=tokens["text_secondary"],
            )
            .encode(
                x=alt.X("value_label:N", sort=sorted_labels),
                y=alt.Y("count:Q", scale=y_scale),
                text=alt.Text("count:Q", format=","),
            )
        )
        layers.append(count_labels)
    elif selected_int is not None:
        sel_df = df[df["value"] == selected_int]
        if not sel_df.empty:
            sel_label = (
                alt.Chart(sel_df)
                .mark_text(
                    dy=-8,
                    font=tokens["font"],
                    fontSize=11,
                    fontWeight=600,
                    color=ASTRYX_ORANGE,
                )
                .encode(
                    x=alt.X("value_label:N", sort=sorted_labels),
                    y=alt.Y("count:Q", scale=y_scale),
                    text=alt.Text("count:Q", format=","),
                )
            )
            layers.append(sel_label)

    layered = alt.layer(*layers)
    return _apply_astryx_chrome(
        layered,
        tokens,
        height=390,
    )


__all__ = [
    "ASTRYX_BLUE",
    "ASTRYX_DATA_NEUTRAL",
    "ASTRYX_GREEN",
    "ASTRYX_ORANGE",
    "ASTRYX_PURPLE",
    "build_domain_distribution_chart",
    "build_cv_stability_chart",
    "build_metrics_comparison_chart",
    "build_price_sensitivity_chart",
]
