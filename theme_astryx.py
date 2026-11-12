"""Astryx Design System (`@astryxdesign/theme-neutral`) integration for Streamlit and Altair 6.3."""

from __future__ import annotations

import html
from typing import Literal

import altair as alt
import streamlit as st

ThemeMode = Literal["auto", "light", "dark"]
KpiTone = Literal["neutral", "success", "warning", "info"]

ASTRYX_FONT_BODY = (
    "'Figtree', -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif"
)
ASTRYX_FONT_CODE = (
    "ui-monospace, \"SF Mono\", Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace"
)

ASTRYX_LIGHT: dict[str, str] = {
    "background_body": "#f1f1f1",
    "background_surface": "#ffffff",
    "background_card": "#ffffff",
    "background_popover": "#ffffff",
    "background_muted": "#f1f1f1",
    "background_gray": "#e2e2e2",
    "border": "#00000014",
    "border_emphasized": "#d4d4d4",
    "text_primary": "#000000",
    "icon_primary": "#000000",
    "text_secondary": "#474747",
    "text_selected_secondary": "#474747",
    "text_disabled": "#919191",
    "accent": "#1b1b1b",
    "text_accent": "#1b1b1b",
    "on_accent": "#ffffff",
    "accent_muted": "#d4d4d4",
    "overlay_hover": "#0000000D",
    "overlay_pressed": "#0000001A",
    "info": "#0074e2",
    "info_muted": "#b9d7ff",
    "text_info": "#0067ce",
    "on_info_muted": "#003978",
    "success": "#198100",
    "success_muted": "#bce0bb",
    "text_success": "#2f7d33",
    "on_success_muted": "#00490b",
    "warning": "#ffce2f",
    "warning_muted": "#fae19e",
    "text_warning": "#836700",
    "on_warning_muted": "#4b3900",
    "error": "#c9303a",
    "error_muted": "#ffc4be",
    "text_error": "#c0262c",
    "on_error_muted": "#76000c",
    "data_neutral": "#8494A3",
    "inset_selected": "#0074e280",
    "inset_hover": "#0074e24D",
}

ASTRYX_DARK: dict[str, str] = {
    "background_body": "#1b1b1b",
    "background_surface": "#262626",
    "background_card": "#1b1b1b",
    "background_popover": "#1b1b1b",
    "background_muted": "#1b1b1b",
    "background_gray": "#303030",
    "border": "#ffffff1A",
    "border_emphasized": "#525252",
    "text_primary": "#ffffff",
    "icon_primary": "#ffffff",
    "text_secondary": "#9e9e9e",
    "text_selected_secondary": "#c6c6c6",
    "text_disabled": "#525252",
    "accent": "#f1f1f1",
    "text_accent": "#f1f1f1",
    "on_accent": "#111111",
    "accent_muted": "#474747",
    "overlay_hover": "#ffffff0D",
    "overlay_pressed": "#ffffff1A",
    "info": "#6d9cfe",
    "info_muted": "#88bcff3D",
    "text_info": "#a1caff",
    "on_info_muted": "#a1caff",
    "success": "#64af4c",
    "success_muted": "#90ca903D",
    "text_success": "#90ca90",
    "on_success_muted": "#90ca90",
    "warning": "#ffce2f",
    "warning_muted": "#e2b6233D",
    "text_warning": "#e3c36c",
    "on_warning_muted": "#e3c36c",
    "error": "#ff705d",
    "error_muted": "#ff98903D",
    "text_error": "#ffaea7",
    "on_error_muted": "#ffaea7",
    "data_neutral": "#8C939B",
    "inset_selected": "#0074e280",
    "inset_hover": "#0074e24D",
}

ASTRYX_DATA_CATEGORICAL: dict[str, str] = {
    "blue": "#0171E3",
    "orange": "#EB6E00",
    "purple": "#6B1EFD",
    "green": "#0B991F",
    "pink": "#F351C0",
    "cyan": "#0171A4",
    "red": "#F5394F",
    "teal": "#08A3A3",
    "brown": "#965E03",
    "indigo": "#6F8AFF",
}

# Five-stop sequential ramps ordered 5 (darkest) -> 1 (lightest), matching @astryxdesign/tokens-data.
ASTRYX_DATA_SEQUENTIAL: dict[str, list[str]] = {
    "blue": ["#02165E", "#004CBC", "#2694FE", "#78BEFF", "#DBECFF"],
    "shamrock": ["#0B603D", "#138546", "#24BB5E", "#8EF7AA", "#D6FEE4"],
    "orange": ["#A13F04", "#D66100", "#FD9537", "#FDB876", "#FFE6CF"],
    "red": ["#9D0519", "#D31130", "#FB7D87", "#FFB2B8", "#FEE4E6"],
    "gray": ["#25363F", "#5D6C7B", "#AFB9C4", "#CCD3DB", "#F1F4F7"],
}

ASTRYX_DATA_SEQUENTIAL_GRAY_DARK: list[str] = [
    "#333338",
    "#666A72",
    "#B2B8BE",
    "#D0D3D6",
    "#F2F4F6",
]

ASTRYX_DATA_NEUTRAL: dict[str, str] = {
    "light": "#8494A3",
    "dark": "#8C939B",
}

_CSS_COLOR_TOKEN_MAP: tuple[tuple[str, str], ...] = (
    ("background-body", "background_body"),
    ("background-surface", "background_surface"),
    ("background-card", "background_card"),
    ("background-popover", "background_popover"),
    ("background-muted", "background_muted"),
    ("background-gray", "background_gray"),
    ("border", "border"),
    ("border-emphasized", "border_emphasized"),
    ("text-primary", "text_primary"),
    ("icon-primary", "icon_primary"),
    ("text-secondary", "text_secondary"),
    ("text-selected-secondary", "text_selected_secondary"),
    ("text-disabled", "text_disabled"),
    ("accent", "accent"),
    ("text-accent", "text_accent"),
    ("on-accent", "on_accent"),
    ("accent-muted", "accent_muted"),
    ("overlay-hover", "overlay_hover"),
    ("overlay-pressed", "overlay_pressed"),
    ("info", "info"),
    ("info-muted", "info_muted"),
    ("text-info", "text_info"),
    ("on-info-muted", "on_info_muted"),
    ("success", "success"),
    ("success-muted", "success_muted"),
    ("text-success", "text_success"),
    ("on-success-muted", "on_success_muted"),
    ("warning", "warning"),
    ("warning-muted", "warning_muted"),
    ("text-warning", "text_warning"),
    ("on-warning-muted", "on_warning_muted"),
    ("error", "error"),
    ("error-muted", "error_muted"),
    ("text-error", "text_error"),
    ("on-error-muted", "on_error_muted"),
    ("data-neutral", "data_neutral"),
    ("inset-selected", "inset_selected"),
    ("inset-hover", "inset_hover"),
)

_GOOGLE_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com" />'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />'
    '<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap" '
    'rel="stylesheet" />'
)


def resolve_tokens(mode: str = "light") -> dict[str, str]:
    """Return resolved Astryx Neutral tokens for `'light'` or `'dark'`."""
    if mode == "light":
        return dict(ASTRYX_LIGHT)
    if mode == "dark":
        return dict(ASTRYX_DARK)
    raise ValueError(f"Expected mode 'light' or 'dark', got {mode!r}")


def get_sequential_ramp(
    name: str = "blue",
    *,
    mode: str = "light",
    low_to_high: bool = False,
) -> list[str]:
    """Return a 5-stop Astryx sequential palette ramp for charts and heatmaps."""
    if name not in ASTRYX_DATA_SEQUENTIAL:
        raise ValueError(f"Unknown sequential ramp {name!r}; expected one of {tuple(ASTRYX_DATA_SEQUENTIAL)}")
    if mode not in ("light", "dark"):
        raise ValueError(f"Expected mode 'light' or 'dark', got {mode!r}")
    ramp = (
        list(ASTRYX_DATA_SEQUENTIAL_GRAY_DARK)
        if name == "gray" and mode == "dark"
        else list(ASTRYX_DATA_SEQUENTIAL[name])
    )
    return list(reversed(ramp)) if low_to_high else ramp


def _color_value(key: str, mode: ThemeMode) -> str:
    light_val = ASTRYX_LIGHT[key]
    dark_val = ASTRYX_DARK[key]
    if mode == "light":
        return light_val
    if mode == "dark":
        return dark_val
    return f"light-dark({light_val}, {dark_val})"


def _shadow_vars(mode: ThemeMode) -> str:
    if mode == "light":
        low = "0 2px 4px oklch(0 0 0 / 5%), 0 4px 8px oklch(0 0 0 / 10%)"
        med = "0 4px 8px oklch(0 0 0 / 8%), 0 8px 16px oklch(0 0 0 / 14%)"
        high = "0 8px 16px oklch(0 0 0 / 12%), 0 16px 32px oklch(0 0 0 / 18%)"
    elif mode == "dark":
        low = (
            "0 2px 4px oklch(0 0 0 / 25%), 0 4px 8px oklch(0 0 0 / 40%), "
            "inset 0 0 0 1px oklch(1 0 0 / 8%)"
        )
        med = (
            "0 4px 8px oklch(0 0 0 / 35%), 0 8px 16px oklch(0 0 0 / 50%), "
            "inset 0 0 0 1px oklch(1 0 0 / 12%)"
        )
        high = (
            "0 8px 16px oklch(0 0 0 / 50%), 0 16px 32px oklch(0 0 0 / 70%), "
            "inset 0 0 0 1px oklch(1 0 0 / 15%)"
        )
    else:
        low = (
            "0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), "
            "0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), "
            "inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))"
        )
        med = (
            "0 4px 8px light-dark(oklch(0 0 0 / 8%), oklch(0 0 0 / 35%)), "
            "0 8px 16px light-dark(oklch(0 0 0 / 14%), oklch(0 0 0 / 50%)), "
            "inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))"
        )
        high = (
            "0 8px 16px light-dark(oklch(0 0 0 / 12%), oklch(0 0 0 / 50%)), "
            "0 16px 32px light-dark(oklch(0 0 0 / 18%), oklch(0 0 0 / 70%)), "
            "inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))"
        )
    return "\n".join(
        [
            f"--shadow-low: {low};",
            f"--shadow-med: {med};",
            f"--shadow-high: {high};",
            "--shadow-inset-selected: inset 0px 0px 0px 2px var(--color-inset-selected);",
            "--shadow-inset-hover: inset 0px 0px 0px 2px var(--color-inset-hover);",
        ]
    )


def build_astryx_css(mode: ThemeMode = "auto") -> str:
    """Build the single Astryx Neutral CSS stylesheet string for `:root` and Streamlit chrome."""
    if mode not in ("auto", "light", "dark"):
        raise ValueError(f"Expected mode 'auto', 'light', or 'dark', got {mode!r}")

    color_scheme = "light dark" if mode == "auto" else mode
    color_vars = "\n".join(
        f"--color-{css_name}: {_color_value(py_key, mode)};"
        for css_name, py_key in _CSS_COLOR_TOKEN_MAP
    )
    categorical_vars = "\n".join(
        f"--color-data-categorical-{name}: {hex_val};"
        for name, hex_val in ASTRYX_DATA_CATEGORICAL.items()
    )
    shadow_vars = _shadow_vars(mode)

    # Keep lines unindented so CommonMark never parses CSS/HTML inside st.markdown as a code block.
    return f"""<style>
:root {{
color-scheme: {color_scheme};
{color_vars}
{categorical_vars}
--font-family-body: {ASTRYX_FONT_BODY};
--font-family-heading: {ASTRYX_FONT_BODY};
--font-family-code: {ASTRYX_FONT_CODE};
--font-size-xs: 0.625rem;
--font-size-sm: 0.75rem;
--font-size-base: 0.875rem;
--font-size-lg: 1.0625rem;
--font-size-xl: 1.25rem;
--font-size-2xl: 1.5rem;
--font-size-3xl: 1.8125rem;
--font-size-4xl: 2.1875rem;
--font-size-5xl: 2.625rem;
--font-weight-normal: 400;
--font-weight-medium: 500;
--font-weight-semibold: 600;
--font-weight-bold: 700;
--spacing-1: 0.25rem;
--spacing-2: 0.5rem;
--spacing-3: 0.75rem;
--spacing-4: 1rem;
--spacing-6: 1.5rem;
--spacing-8: 2rem;
--size-element-sm: 28px;
--size-element-md: 32px;
--size-element-lg: 36px;
--border-width: 1px;
--radius-none: 0px;
--radius-inner: 0.375rem;
--radius-element: 0.625rem;
--radius-container: 0.75rem;
--radius-page: 1.75rem;
--radius-full: 9999px;
{shadow_vars}
--motion-fast: 125ms;
--motion-medium: 300ms;
--motion-slow: 700ms;
--ease-standard: cubic-bezier(0.24, 1, 0.4, 1);
--ease-exit: cubic-bezier(0.3, 0, 0.6, 0.6);
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
background-color: var(--color-background-body);
color: var(--color-text-primary);
font-family: var(--font-family-body);
font-size: var(--font-size-base);
line-height: 1.4286;
}}

[data-testid="stHeader"] {{
background: transparent;
}}

[data-testid="stMainBlockContainer"] {{
padding-top: var(--spacing-6);
padding-bottom: var(--spacing-8);
}}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
background-color: var(--color-background-surface);
border-right: var(--border-width) solid var(--color-border);
}}

h1, .stMarkdown h1, [data-testid="stHeading"] h1 {{
font-family: var(--font-family-heading);
font-size: var(--font-size-2xl);
font-weight: var(--font-weight-semibold);
line-height: 1.3333;
letter-spacing: -0.015em;
color: var(--color-text-primary);
}}

h2, .stMarkdown h2, [data-testid="stHeading"] h2 {{
font-family: var(--font-family-heading);
font-size: var(--font-size-xl);
font-weight: var(--font-weight-semibold);
line-height: 1.4;
letter-spacing: -0.01em;
color: var(--color-text-primary);
}}

h3, .stMarkdown h3, [data-testid="stHeading"] h3 {{
font-family: var(--font-family-heading);
font-size: var(--font-size-lg);
font-weight: var(--font-weight-bold);
line-height: 1.4118;
color: var(--color-text-primary);
}}

h4, .stMarkdown h4, [data-testid="stHeading"] h4 {{
font-family: var(--font-family-heading);
font-size: var(--font-size-base);
font-weight: var(--font-weight-bold);
line-height: 1.4286;
color: var(--color-text-primary);
}}

code, pre, [data-testid="stCode"] {{
font-family: var(--font-family-code);
}}

[data-testid="stTabs"] [role="tablist"] {{
border-bottom: var(--border-width) solid var(--color-border);
gap: var(--spacing-4);
}}

[data-testid="stTabs"] button[role="tab"] {{
font-family: var(--font-family-body);
font-size: var(--font-size-base);
font-weight: var(--font-weight-medium);
color: var(--color-text-secondary);
padding: var(--spacing-2) var(--spacing-1);
transition: color var(--motion-fast) var(--ease-standard);
}}

[data-testid="stTabs"] button[role="tab"]:hover {{
color: var(--color-text-primary);
}}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
color: var(--color-text-primary);
font-weight: var(--font-weight-semibold);
}}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
background-color: var(--color-accent);
height: 2px;
}}

[data-testid="stForm"] {{
background-color: var(--color-background-surface);
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
padding: var(--spacing-4);
}}

[data-testid="stMetric"] {{
background-color: var(--color-background-card);
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
padding: var(--spacing-3) var(--spacing-4);
}}

[data-testid="stMetricLabel"] {{
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
color: var(--color-text-secondary);
}}

[data-testid="stMetricValue"] {{
font-size: var(--font-size-2xl);
font-weight: var(--font-weight-semibold);
color: var(--color-text-primary);
font-variant-numeric: tabular-nums;
}}

[data-testid="stAlert"] {{
border-radius: var(--radius-container);
border-width: var(--border-width);
font-family: var(--font-family-body);
font-size: var(--font-size-base);
}}

[data-testid="stFormSubmitButton"] button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"] {{
background-color: var(--color-accent);
color: var(--color-on-accent);
border: var(--border-width) solid var(--color-accent);
border-radius: var(--radius-element);
min-height: var(--size-element-lg);
font-family: var(--font-family-body);
font-size: var(--font-size-base);
font-weight: var(--font-weight-medium);
transition: background-color var(--motion-fast) var(--ease-standard), opacity var(--motion-fast) var(--ease-standard);
}}

[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover {{
opacity: 0.9;
}}

[data-testid="stBaseButton-secondary"],
.stButton > button[kind="secondary"] {{
background-color: var(--color-background-surface);
color: var(--color-text-primary);
border: var(--border-width) solid var(--color-border-emphasized);
border-radius: var(--radius-element);
min-height: var(--size-element-md);
font-family: var(--font-family-body);
font-size: var(--font-size-base);
font-weight: var(--font-weight-medium);
}}

button:focus-visible,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {{
outline: 2px solid var(--color-accent);
outline-offset: 3px;
}}

[data-testid="stPills"] button,
[data-testid="stSegmentedControl"] button {{
font-family: var(--font-family-body);
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
min-height: var(--size-element-sm);
border-radius: var(--radius-element);
border-color: var(--color-border-emphasized);
color: var(--color-text-secondary);
background-color: var(--color-background-surface);
transition: background-color var(--motion-fast) var(--ease-standard), color var(--motion-fast) var(--ease-standard);
}}

[data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stPills"] button[kind="pillsActive"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {{
background-color: var(--color-accent);
color: var(--color-on-accent);
border-color: var(--color-accent);
}}

[data-testid="stDataFrame"] {{
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
overflow: hidden;
font-variant-numeric: tabular-nums;
}}

.astryx-hero {{
padding: var(--spacing-2) 0 var(--spacing-4) 0;
border-bottom: var(--border-width) solid var(--color-border);
margin-bottom: var(--spacing-4);
}}

.astryx-hero-title {{
margin: 0 0 var(--spacing-1) 0;
font-family: var(--font-family-heading);
font-size: var(--font-size-2xl);
font-weight: var(--font-weight-semibold);
line-height: 1.3333;
letter-spacing: -0.015em;
color: var(--color-text-primary);
}}

.astryx-hero-subtitle {{
margin: 0;
font-size: var(--font-size-base);
font-weight: var(--font-weight-normal);
line-height: 1.4286;
color: var(--color-text-secondary);
}}

.astryx-badge-row {{
display: flex;
flex-wrap: wrap;
gap: var(--spacing-2);
margin-top: var(--spacing-3);
}}

.astryx-badge {{
display: inline-flex;
align-items: center;
padding: 2px 10px;
border-radius: var(--radius-full);
background-color: var(--color-background-gray);
border: var(--border-width) solid var(--color-border);
color: var(--color-text-secondary);
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
line-height: 1.5;
}}

.astryx-section-hdr {{
margin: var(--spacing-4) 0 var(--spacing-3) 0;
}}

.astryx-section-title {{
margin: 0;
font-family: var(--font-family-heading);
font-size: var(--font-size-lg);
font-weight: var(--font-weight-bold);
line-height: 1.4118;
color: var(--color-text-primary);
}}

.astryx-section-subtitle {{
margin: var(--spacing-1) 0 0 0;
font-size: var(--font-size-sm);
font-weight: var(--font-weight-normal);
line-height: 1.6667;
color: var(--color-text-secondary);
}}

.astryx-kpi-grid {{
display: grid;
grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
gap: var(--spacing-3);
margin: var(--spacing-3) 0 var(--spacing-4) 0;
}}

.astryx-kpi-card {{
background-color: var(--color-background-card);
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
padding: var(--spacing-3) var(--spacing-4);
display: flex;
flex-direction: column;
gap: var(--spacing-1);
}}

.astryx-kpi-label {{
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
line-height: 1.4;
color: var(--color-text-secondary);
}}

.astryx-kpi-value {{
font-size: var(--font-size-2xl);
font-weight: var(--font-weight-semibold);
line-height: 1.25;
color: var(--color-text-primary);
font-variant-numeric: tabular-nums;
}}

.astryx-kpi-delta {{
display: inline-flex;
align-self: flex-start;
align-items: center;
margin-top: var(--spacing-1);
padding: 2px 8px;
border-radius: var(--radius-inner);
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
line-height: 1.4;
}}

.astryx-tone-neutral {{
background-color: var(--color-background-gray);
color: var(--color-text-secondary);
}}

.astryx-tone-success {{
background-color: var(--color-success-muted);
color: var(--color-on-success-muted);
}}

.astryx-tone-warning {{
background-color: var(--color-warning-muted);
color: var(--color-on-warning-muted);
}}

.astryx-tone-info {{
background-color: var(--color-info-muted);
color: var(--color-on-info-muted);
}}

.astryx-pred-card {{
background-color: var(--color-background-card);
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
box-shadow: var(--shadow-low);
padding: var(--spacing-4);
margin: var(--spacing-3) 0;
display: flex;
flex-direction: column;
gap: var(--spacing-3);
}}

.astryx-pred-feasible {{
border-left: 4px solid var(--color-success);
}}

.astryx-pred-infeasible {{
border-left: 4px solid var(--color-warning);
}}

.astryx-pred-top {{
display: flex;
flex-wrap: wrap;
align-items: center;
justify-content: space-between;
gap: var(--spacing-2);
}}

.astryx-pred-outcome {{
font-family: var(--font-family-heading);
font-size: var(--font-size-xl);
font-weight: var(--font-weight-semibold);
line-height: 1.3;
color: var(--color-text-primary);
margin: 0;
}}

.astryx-pred-chips {{
display: flex;
flex-wrap: wrap;
gap: var(--spacing-2);
}}

.astryx-pred-chip {{
display: inline-flex;
align-items: center;
padding: 3px 10px;
border-radius: var(--radius-full);
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
}}

.astryx-pred-meta {{
display: grid;
grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
gap: var(--spacing-3);
padding-top: var(--spacing-3);
border-top: var(--border-width) solid var(--color-border);
}}

.astryx-pred-meta-label {{
font-size: var(--font-size-sm);
font-weight: var(--font-weight-medium);
color: var(--color-text-secondary);
margin-bottom: 2px;
}}

.astryx-pred-meta-val {{
font-size: var(--font-size-base);
font-weight: var(--font-weight-semibold);
color: var(--color-text-primary);
font-variant-numeric: tabular-nums;
}}

.astryx-pred-meta-val code {{
font-family: var(--font-family-code);
font-size: var(--font-size-sm);
background-color: var(--color-background-muted);
padding: 2px 6px;
border-radius: var(--radius-inner);
border: var(--border-width) solid var(--color-border);
}}

.astryx-term-list {{
background-color: var(--color-background-surface);
border: var(--border-width) solid var(--color-border);
border-radius: var(--radius-container);
overflow: hidden;
margin: var(--spacing-2) 0 var(--spacing-4) 0;
}}

.astryx-term-row {{
display: grid;
grid-template-columns: minmax(170px, 1fr) minmax(240px, 2.2fr) minmax(160px, 1.3fr);
gap: var(--spacing-3);
padding: var(--spacing-3) var(--spacing-4);
border-bottom: var(--border-width) solid var(--color-border);
align-items: baseline;
}}

.astryx-term-row:last-child {{
border-bottom: none;
}}

@media (max-width: 720px) {{
.astryx-term-row {{
grid-template-columns: 1fr;
gap: var(--spacing-1);
}}
}}

.astryx-term-name {{
font-size: var(--font-size-base);
font-weight: var(--font-weight-semibold);
color: var(--color-text-primary);
}}

.astryx-term-def {{
font-size: var(--font-size-base);
font-weight: var(--font-weight-normal);
color: var(--color-text-secondary);
}}

.astryx-term-avoid {{
font-size: var(--font-size-sm);
font-weight: var(--font-weight-normal);
color: var(--color-text-error);
}}
</style>"""


def inject_astryx_theme(mode: str = "auto") -> None:
    """Inject Figtree Google Fonts and Astryx Neutral CSS variables/chrome overrides into Streamlit."""
    css_block = build_astryx_css(mode=mode)  # validates mode
    st.markdown(f"{_GOOGLE_FONTS_LINK}\n{css_block}", unsafe_allow_html=True)


def register_astryx_altair_theme(mode: str = "light") -> str:
    """Register and enable the `'astryx_neutral'` Altair 6.3 theme for the given mode."""
    tokens = resolve_tokens(mode)
    theme_name = "astryx_neutral"
    categorical = list(ASTRYX_DATA_CATEGORICAL.values())
    heatmap_ramp = get_sequential_ramp("blue", mode=mode, low_to_high=True)

    @alt.theme.register(theme_name, enable=True)
    def _astryx_neutral_theme() -> alt.theme.ThemeConfig:
        return {
            "background": "transparent",
            "config": {
                "font": ASTRYX_FONT_BODY,
                "view": {
                    "stroke": "transparent",
                },
                "title": {
                    "color": tokens["text_primary"],
                    "subtitleColor": tokens["text_secondary"],
                    "font": ASTRYX_FONT_BODY,
                    "subtitleFont": ASTRYX_FONT_BODY,
                    "fontSize": 17,
                    "fontWeight": 700,
                    "subtitleFontSize": 12,
                    "anchor": "start",
                    "offset": 10,
                },
                "axis": {
                    "domain": False,
                    "ticks": False,
                    "gridColor": tokens["border"],
                    "gridWidth": 1,
                    "labelColor": tokens["text_secondary"],
                    "titleColor": tokens["text_secondary"],
                    "labelFont": ASTRYX_FONT_BODY,
                    "titleFont": ASTRYX_FONT_BODY,
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                    "titleFontWeight": 500,
                    "labelPadding": 8,
                    "titlePadding": 10,
                },
                "axisX": {
                    "grid": False,
                },
                "axisY": {
                    "grid": True,
                },
                "legend": {
                    "labelColor": tokens["text_secondary"],
                    "titleColor": tokens["text_primary"],
                    "labelFont": ASTRYX_FONT_BODY,
                    "titleFont": ASTRYX_FONT_BODY,
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                    "titleFontWeight": 600,
                },
                "header": {
                    "labelColor": tokens["text_secondary"],
                    "titleColor": tokens["text_primary"],
                    "labelFont": ASTRYX_FONT_BODY,
                    "titleFont": ASTRYX_FONT_BODY,
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "bar": {
                    "cornerRadiusEnd": 4,
                },
                "range": {
                    "category": categorical,
                    "heatmap": heatmap_ramp,
                    "ramp": heatmap_ramp,
                    "ordinal": heatmap_ramp,
                },
            },
        }

    alt.theme.enable(theme_name)
    return theme_name


def render_hero_header(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    """Render an Astryx Neutral hero header with escaped title, subtitle, and optional badges."""
    safe_title = html.escape(title, quote=True)
    safe_subtitle = html.escape(subtitle, quote=True)
    badge_html = ""
    if badges:
        pills = "".join(
            f'<span class="astryx-badge">{html.escape(badge, quote=True)}</span>'
            for badge in badges
            if badge
        )
        if pills:
            badge_html = f'<div class="astryx-badge-row">{pills}</div>'
    markup = (
        '<div class="astryx-hero">'
        f'<h1 class="astryx-hero-title">{safe_title}</h1>'
        f'<p class="astryx-hero-subtitle">{safe_subtitle}</p>'
        f"{badge_html}"
        "</div>"
    )
    st.markdown(markup, unsafe_allow_html=True)


def render_kpi_strip(items: list[dict[str, str]]) -> None:
    """Render a reflowable Astryx KPI strip from dicts with label, value, and optional delta/tone."""
    if not items:
        return
    valid_tones = {"neutral", "success", "warning", "info"}
    cards: list[str] = []
    for item in items:
        safe_label = html.escape(str(item.get("label", "")), quote=True)
        safe_value = html.escape(str(item.get("value", "")), quote=True)
        delta_raw = item.get("delta")
        tone_raw = str(item.get("tone", "neutral")).lower()
        tone = tone_raw if tone_raw in valid_tones else "neutral"
        delta_html = ""
        if delta_raw:
            safe_delta = html.escape(str(delta_raw), quote=True)
            delta_html = f'<span class="astryx-kpi-delta astryx-tone-{tone}">{safe_delta}</span>'
        cards.append(
            '<div class="astryx-kpi-card">'
            f'<span class="astryx-kpi-label">{safe_label}</span>'
            f'<span class="astryx-kpi-value">{safe_value}</span>'
            f"{delta_html}"
            "</div>"
        )
    st.markdown(f'<div class="astryx-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Render a typographically ranked Astryx section header with optional supporting subtitle."""
    safe_title = html.escape(title, quote=True)
    sub_html = (
        f'<p class="astryx-section-subtitle">{html.escape(subtitle, quote=True)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        '<div class="astryx-section-hdr">'
        f'<h3 class="astryx-section-title">{safe_title}</h3>'
        f"{sub_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_prediction_card(
    outcome_label: str,
    is_feasible: bool,
    is_seen: bool,
    decision_margin: float,
    config_key: str,
) -> None:
    """Render an Astryx outcome summary card complementing the AppTest st.success/st.warning banner."""
    safe_outcome = html.escape(outcome_label, quote=True)
    safe_key = html.escape(config_key, quote=True)
    accent_cls = "astryx-pred-feasible" if is_feasible else "astryx-pred-infeasible"
    outcome_tone = "astryx-tone-success" if is_feasible else "astryx-tone-warning"
    outcome_badge = "Predicted Feasible (class 1)" if is_feasible else "Predicted Infeasible (class 0)"
    novelty_tone = "astryx-tone-neutral" if is_seen else "astryx-tone-info"
    novelty_badge = "In-Domain · Seen in Training" if is_seen else "In-Domain · Novel Configuration"
    margin_text = html.escape(f"{float(decision_margin):+.4f}", quote=True)

    markup = (
        f'<div class="astryx-pred-card {accent_cls}">'
        '<div class="astryx-pred-top">'
        f'<h3 class="astryx-pred-outcome">{safe_outcome}</h3>'
        '<div class="astryx-pred-chips">'
        f'<span class="astryx-pred-chip {outcome_tone}">{outcome_badge}</span>'
        f'<span class="astryx-pred-chip {novelty_tone}">{novelty_badge}</span>'
        "</div>"
        "</div>"
        '<div class="astryx-pred-meta">'
        "<div>"
        '<div class="astryx-pred-meta-label">SVM RBF Hyperplane Margin (distance, not probability)</div>'
        f'<div class="astryx-pred-meta-val">{margin_text}</div>'
        "</div>"
        "<div>"
        '<div class="astryx-pred-meta-label">Auction Configuration Key</div>'
        f'<div class="astryx-pred-meta-val"><code>{safe_key}</code></div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(markup, unsafe_allow_html=True)


def render_terminology_grid(terms: list[tuple[str, str, str]]) -> None:
    """Render CONTEXT.md terminology pairs in a clean hairline-divided table without nested cards."""
    if not terms:
        return
    rows: list[str] = []
    for term, definition, avoid in terms:
        safe_term = html.escape(term, quote=True)
        safe_def = html.escape(definition, quote=True)
        safe_avoid = html.escape(avoid, quote=True)
        rows.append(
            '<div class="astryx-term-row">'
            f'<div class="astryx-term-name">{safe_term}</div>'
            f'<div class="astryx-term-def">{safe_def}</div>'
            f'<div class="astryx-term-avoid">Avoid: {safe_avoid}</div>'
            "</div>"
        )
    st.markdown(f'<div class="astryx-term-list">{"".join(rows)}</div>', unsafe_allow_html=True)
