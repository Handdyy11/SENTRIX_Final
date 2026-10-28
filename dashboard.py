"""SENTRIX dashboard.  Run from the project root:  streamlit run dashboard.py"""
import base64
import json
import time
from html import escape as esc

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(
    page_title="SENTRIX",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Replace these placeholders with your own team details
TEAM = [
    {"name": "Team Member 1", "role": "Role / responsibility"},
    {"name": "Team Member 2", "role": "Role / responsibility"},
    {"name": "Team Member 3", "role": "Role / responsibility"},
    {"name": "Team Member 4", "role": "Role / responsibility"},
]

# Theme colours. Every colour used by the page comes from here.
THEMES = {
    "dark": {
        "bg": "#0e1117",
        "surface": "rgba(255,255,255,.04)",
        "solid": "#161c2a",
        "border": "#27303f",
        "text": "#f1f5f9",
        "muted": "#9aa4b2",
        "sub": "#cbd5e1",
        "input": "#1a2233",
        "nav": "rgba(14,17,23,.88)",
        "thead": "#1b2333",
        "alt": "rgba(255,255,255,.025)",
        "accent": "#38bdf8",
        "soft": "rgba(56,189,248,.14)",
        "hover": "rgba(255,255,255,.06)",
        "shadow": "0 6px 24px rgba(0,0,0,.35)",
        "code": "#0b1220",
        "grid": "rgba(255,255,255,.08)",
        "panel": "linear-gradient(135deg,#111827,#0f172a)",
        "plot_bg": "#0e1627",
    },
    "light": {
        "bg": "#f3f6fb",
        "surface": "#ffffff",
        "solid": "#ffffff",
        "border": "#e2e8f0",
        "text": "#0f172a",
        "muted": "#64748b",
        "sub": "#475569",
        "input": "#ffffff",
        "nav": "rgba(255,255,255,.9)",
        "thead": "#f1f5f9",
        "alt": "#f8fafc",
        "accent": "#0284c7",
        "soft": "rgba(2,132,199,.10)",
        "hover": "rgba(15,23,42,.05)",
        "shadow": "0 4px 20px rgba(15,23,42,.07)",
        "code": "#f1f5f9",
        "grid": "#e2e8f0",
        "panel": "linear-gradient(135deg,#ffffff,#eef2ff)",
        "plot_bg": "#ffffff",
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, p, label, input, textarea, button, h1, h2, h3, h4, h5, h6,
.stMarkdown, [data-testid="stMetricValue"] {font-family: 'Inter', 'Segoe UI', Arial, sans-serif;}
[data-testid="stIconMaterial"], .material-symbols-rounded {font-family: 'Material Symbols Rounded' !important;}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--bg) !important; color: var(--text);}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display: none !important;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
.block-container {padding: .8rem 2rem 2rem; max-width: 1280px;}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {color: var(--text) !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p {color: var(--text);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}

[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important;}
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] div,
[data-baseweb="select"] span, [data-baseweb="select"] input {color: var(--text) !important; -webkit-text-fill-color: var(--text);}
[data-baseweb="select"] svg {fill: var(--muted) !important;}
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"] {background: var(--solid) !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--text) !important;}
[data-baseweb="popover"] li:hover {background: var(--hover) !important;}
[data-testid="stForm"] [data-testid="InputInstructions"] {display: none !important;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display: none !important;}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color: var(--muted) !important;}
[data-testid="stSlider"] [data-testid="stThumbValue"] {color: var(--accent) !important;}

[data-testid="stExpander"] details {background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px;}
[data-testid="stExpander"] summary svg {color: var(--muted);}

[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stAlert"] p {color: var(--text) !important;}

[data-testid="stFileUploaderDropzone"] {background: var(--input); border: 1px dashed var(--border); border-radius: 12px;}
[data-testid="stFileUploaderDropzone"] * {color: var(--sub) !important;}

[data-testid="stCode"], [data-testid="stCode"] pre {background: var(--code) !important; border-radius: 10px;}
[data-testid="stCode"] code, [data-testid="stCode"] span {color: var(--text) !important;}

[data-baseweb="tab-list"] {gap: 6px;}
button[role="tab"] {color: var(--muted) !important; background: transparent !important; font-weight: 600;}
button[role="tab"] p {color: inherit !important;}
button[role="tab"][aria-selected="true"] {color: var(--accent) !important;}
[data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}
[data-baseweb="tab-border"] {background-color: var(--border) !important;}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, [data-testid^="stBaseButton"] {
  border-radius: 10px; border: 1px solid var(--border); background: var(--solid); color: var(--text); font-weight: 600;}
.stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p,
[data-testid^="stBaseButton"] p {color: inherit !important; font-weight: 600;}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover,
[data-testid^="stBaseButton"]:hover {border-color: var(--accent); color: var(--accent);}
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; border: none !important; color: #ffffff !important;}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  color: #ffffff !important; filter: brightness(1.1);}

.st-key-topbar {position: sticky; top: .5rem; z-index: 999; background: var(--nav); backdrop-filter: blur(12px);
  border: 1px solid var(--border); border-radius: 16px; padding: .45rem 1rem; box-shadow: var(--shadow); margin-bottom: 1.2rem;}
.st-key-nav [role="radiogroup"] {gap: .25rem; flex-wrap: nowrap; justify-content: center;}
.st-key-nav label {padding: 8px 16px; border-radius: 10px; cursor: pointer; margin: 0;}
.st-key-nav label > div:first-child {display: none;}
.st-key-nav label p {font-weight: 600; font-size: 14px; white-space: nowrap; color: var(--muted) !important;}
.st-key-nav label:hover {background: var(--hover);}
.st-key-nav label:has(input:checked) {background: var(--soft); box-shadow: inset 0 -2px 0 var(--accent);}
.st-key-nav label:has(input:checked) p {color: var(--accent) !important;}
.st-key-theme_btn button {border-radius: 999px; font-size: 13px;}

.brand {display: flex; align-items: center; gap: 10px; position: relative; top: -10px;}
.logo {display: inline-flex; width: 34px; height: 34px; align-items: center; justify-content: center;
       border-radius: 10px; background: linear-gradient(135deg, #0ea5e9, #6366f1); font-size: 18px;}
.brand-t {font-weight: 800; letter-spacing: .14em; font-size: 18px; color: var(--text);}

[class*="st-key-card_"] {background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 1rem 1.2rem; box-shadow: var(--shadow);}

.ph {margin: 4px 0 18px;}
.ph-t {font-size: 28px; font-weight: 800; color: var(--text);}
.ph-s {color: var(--muted); font-size: 14px; margin-top: 2px;}

.kpi {background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
      padding: 16px 18px; margin-bottom: 12px;}
.kl {font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600;}
.kv {font-size: 27px; font-weight: 700; line-height: 1.3; color: var(--text);}
.ks {font-size: 12px; color: var(--muted);}

.section {font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
          color: var(--muted); margin: 8px 0 10px 2px;}

.verdict {border: 2px solid; border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 14px;}
.vl {font-size: 11px; letter-spacing: .14em; color: var(--muted); font-weight: 600;}
.vv {font-size: 40px; font-weight: 800; line-height: 1.2;}
.vs {font-size: 13px; color: var(--sub);}
.pill {display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 12px;
       font-weight: 700; letter-spacing: .1em; color: #0b1220; margin-top: 10px;}
.action {background: var(--surface); border: 1px solid var(--border); color: var(--text);
         border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;}
.flowhdr {background: var(--panel); border: 1px solid var(--border); padding: 18px 20px;
          border-radius: 16px; margin-bottom: 16px; color: var(--text);}
.flowhdr-l {font-size: 11px; letter-spacing: .14em; color: var(--muted); text-transform: uppercase;}
.flowhdr-v {font-size: 20px; font-weight: 700; margin-top: 8px;}
.flowhdr-s {margin-top: 8px; color: var(--sub);}

.stApp .tbl-wrap {max-height: 520px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface);}
.stApp .tbl {width: 100%; border-collapse: collapse; font-size: 13px; color: var(--text); margin: 0;}
.stApp .tbl th {position: sticky; top: 0; background: var(--thead); color: var(--muted); text-transform: uppercase;
                font-size: 11px; letter-spacing: .08em; text-align: left; padding: 10px 14px; font-weight: 700;
                border: 0; border-bottom: 1px solid var(--border); white-space: nowrap;}
.stApp .tbl td {padding: 9px 14px; border: 0; border-bottom: 1px solid var(--border); white-space: nowrap;
                font-variant-numeric: tabular-nums; color: var(--text);}
.stApp .tbl tbody tr:nth-child(even) td {background: var(--alt);}
.stApp .tbl tbody tr:hover td {background: var(--hover);}
.stApp .tbl td.idx {font-weight: 600;}
.tbl-note {font-size: 12px; color: var(--muted); margin-top: 6px;}
.bar {position: relative; min-width: 140px; height: 22px; background: var(--hover); border-radius: 6px; overflow: hidden;}
.bar .fill {position: absolute; top: 0; bottom: 0; left: 0; background: linear-gradient(90deg, #0ea5e9, #6366f1); opacity: .55; border-radius: 6px;}
.bar .num {position: relative; padding: 0 8px; line-height: 22px; font-weight: 600; font-size: 12px;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: .04em;}

.hero2 {display: flex; align-items: center; gap: 28px; padding: 34px 38px; border-radius: 22px; margin-bottom: 20px;
        background: radial-gradient(900px 400px at 85% 10%, rgba(99,102,241,.35), transparent 60%),
                    linear-gradient(135deg, #0b1220 0%, #0f1d3a 55%, #1e1b4b 100%);
        border: 1px solid rgba(255,255,255,.08); box-shadow: 0 18px 50px rgba(30,27,75,.35);}
.hero2-text {flex: 1 1 50%;}
.hero2-art {flex: 1 1 46%;}
.hero2-art img {width: 100%; display: block; border-radius: 16px;}
.hero2-title {font-size: 66px; font-weight: 800; letter-spacing: .12em; line-height: 1.05;
              background: linear-gradient(90deg, #ffffff, #7dd3fc); -webkit-background-clip: text; background-clip: text;
              -webkit-text-fill-color: transparent; color: transparent;}
.hero2-tag {color: #7dd3fc; font-weight: 600; font-size: 16px; margin-top: 8px;}
.hero2-sub {color: rgba(226,232,240,.92); font-size: 16px; margin-top: 14px; line-height: 1.65; max-width: 520px;}
.hchip {display: inline-block; padding: 5px 12px; border-radius: 999px; background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14); color: #e2e8f0; font-size: 12px; font-weight: 600; margin: 16px 8px 0 0;}

.grid {display: grid; gap: 14px; margin: 6px 0 18px;}
.g4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
.g3 {grid-template-columns: repeat(3, minmax(0, 1fr));}
.g2 {grid-template-columns: repeat(2, minmax(0, 1fr));}
.fcard {background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--c, #38bdf8);
        border-radius: 14px; padding: 18px; box-shadow: var(--shadow);}
.fi {font-size: 26px;}
.ft {font-weight: 700; font-size: 16px; margin-top: 6px; color: var(--text);}
.fd {font-size: 13.5px; color: var(--sub); margin-top: 4px; line-height: 1.55;}
.fa {font-size: 12.5px; color: var(--muted); margin-top: 10px; font-weight: 600;}
.stepn {display: inline-flex; width: 28px; height: 28px; align-items: center; justify-content: center; border-radius: 50%;
        background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #ffffff; font-weight: 700; font-size: 13px;}
.chip {display: inline-block; padding: 6px 14px; margin: 0 8px 8px 0; border-radius: 999px; border: 1px solid var(--border);
       background: var(--surface); font-size: 13px; font-weight: 600; color: var(--text);}
.prose {color: var(--sub); font-size: 15px; line-height: 1.75; max-width: 900px;}
.avatar {width: 54px; height: 54px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
         background: linear-gradient(135deg, #0ea5e9, #8b5cf6); color: #ffffff; font-weight: 700; font-size: 18px;}
.footer {text-align: center; color: var(--muted); font-size: 12px; margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--border);}

@media (max-width: 1000px) {
  .g4, .g3 {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .hero2-title {font-size: 48px;}
}
@media (max-width: 700px) {
  .g4, .g3, .g2 {grid-template-columns: 1fr;}
  .hero2 {flex-direction: column; padding: 24px;}
  .block-container {padding: .6rem 1rem 2rem;}
}
</style>
"""

FAMILY_COLORS = {
    "Benign": "#22c55e",
    "PortScan": "#f59e0b",
    "BruteForce": "#f97316",
    "DoS": "#ef4444",
}

# What the autonomous defense does for each predicted family
ACTIONS = {
    "Benign": "Allow the traffic",
    "PortScan": "Raise an alert and log the source IP",
    "BruteForce": "Block the source IP",
    "DoS": "Rate-limit or block the source IP",
}
ACTION_TAG = {"Benign": "ALLOW", "PortScan": "ALERT", "BruteForce": "BLOCK", "DoS": "RATE-LIMIT"}
ACTION_COLORS = {
    "ALLOW": "#22c55e",
    "ALERT": "#f59e0b",
    "BLOCK": "#f97316",
    "RATE-LIMIT": "#ef4444",
    "DROPPED (BLACKLISTED)": "#ef4444",
    "INSPECT": "#a78bfa",
    "REVIEW": "#94a3b8",
}

FAMILY_INFO = {
    "Benign": "Normal everyday traffic such as browsing, email and file transfers.",
    "PortScan": "An attacker probing many ports to find open services.",
    "BruteForce": "Repeated login guesses against services like SSH or FTP.",
    "DoS": "A flood of requests meant to make a service unavailable.",
}

# Features that only take a few fixed values -> shown as a dropdown
CHOICES = {
    "Protocol": {6: "6 (TCP)", 17: "17 (UDP)"},
    "PSH Flag Count": {0: "0 (no PSH flag)", 1: "1 (PSH flag set)"},
}

# Display name: (key in metrics_knn.json, saved model file)
MODELS = {
    "k-NN": ("knn", "knn_model.joblib"),
    "Decision tree": ("decision_tree", "tree_model.joblib"),
    "Random Forest": ("random_forest", "forest_model.joblib"),
    "XGBoost": ("xgboost", "xgboost_model.joblib"),
}

PORT_SERVICES = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
    445: "SMB", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
    8080: "HTTP-Alt", 8443: "HTTPS-Alt",
}


def format_service(port):
    try:
        port = int(port)
    except (TypeError, ValueError):
        return "Unknown"
    if port in PORT_SERVICES:
        return f"{port} ({PORT_SERVICES[port]})"
    return f"{port} (Custom)"


def format_attack_route(row):
    if isinstance(row, pd.Series):
        src_ip = row.get("Source IP", "unknown")
        src_port = row.get("Source Port", 0)
        dst_ip = row.get("Destination IP", "unknown")
        dst_port = row.get("Destination Port", 80)
    else:
        src_ip = row.get("Source IP", row.get("src_ip", "unknown"))
        src_port = row.get("Source Port", row.get("src_port", 0))
        dst_ip = row.get("Destination IP", row.get("dst_ip", "unknown"))
        dst_port = row.get("Destination Port", row.get("dst_port", 80))
    return f"{src_ip}:{src_port} ──► {dst_ip}:{dst_port} ({format_service(dst_port)})"


def get_model_classes(model, fallback_labels):
    fallback_labels = list(fallback_labels)
    classes = getattr(model, "classes_", None)
    if classes is None:
        return fallback_labels
    classes = list(classes)
    if not classes:
        return fallback_labels
    arr = np.asarray(classes)
    if arr.dtype.kind in {"i", "u", "f"} and arr.size and np.all(np.isfinite(arr.astype(float))):
        ref = sorted(fallback_labels)
        mapped = []
        for item in classes:
            idx = int(float(item))
            if 0 <= idx < len(ref):
                mapped.append(ref[idx])
            else:
                mapped.append(str(item))
        return mapped
    return [str(label) for label in classes]


# Theme helpers
def theme():
    return THEMES[st.session_state.get("theme", "dark")]


def theme_vars():
    body = "".join(f"--{key}:{value};" for key, value in theme().items())
    return f"<style>:root{{{body}}}</style>"


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.get("theme", "dark") == "dark" else "dark"


def go(page):
    st.session_state.nav = page


# Data loaders
def read_json(name):
    path = config.RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def feature_columns():
    return json.loads((config.DATA_PROCESSED / "feature_columns.json").read_text())


@st.cache_data
def load_family_counts():
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet", columns=["family"])
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet", columns=["family"])
    return pd.concat([train, test])["family"].value_counts()


@st.cache_data
def load_test():
    path = config.DATA_PROCESSED / "test.parquet"
    try:
        from pyarrow.parquet import ParquetFile
        schema_columns = ParquetFile(path).schema.names
    except Exception:
        schema_columns = list(pd.read_parquet(path).columns)
    cols = feature_columns()
    metadata_cols = [
        "Source IP", "Destination IP", "Source Port", "Destination Port",
        "Protocol", "Timestamp", "Label", "family",
    ]
    available = [c for c in metadata_cols if c in schema_columns]
    wanted = list(dict.fromkeys(cols + available))
    df = pd.read_parquet(path, columns=wanted)

    for c in metadata_cols:
        if c not in df.columns:
            df[c] = np.nan

    if "Source IP" in df.columns:
        mask = df["Source IP"].isna() | (df["Source IP"].astype(str).str.strip() == "")

        def _src_default(row):
            fam = row.get("family")
            label = str(row.get("Label", "")).upper()
            if fam == "Benign" or label == "BENIGN":
                return "192.168.10.12"
            return "172.16.0.1"

        df.loc[mask, "Source IP"] = df.loc[mask].apply(_src_default, axis=1)

    if "Destination IP" in df.columns:
        df["Destination IP"] = df["Destination IP"].fillna("192.168.10.50")

    if "Destination Port" in df.columns:
        df["Destination Port"] = pd.to_numeric(df["Destination Port"], errors="coerce")
        df["Destination Port"] = df["Destination Port"].fillna(df["Destination Port"].dropna().median())
        df["Destination Port"] = df["Destination Port"].fillna(80).astype(int)
    else:
        df["Destination Port"] = 80

    if "Source Port" in df.columns:
        df["Source Port"] = pd.to_numeric(df["Source Port"], errors="coerce")
        df["Source Port"] = df["Source Port"].fillna(df["Source Port"].dropna().median())
        df["Source Port"] = df["Source Port"].fillna(49152).astype(int)
    else:
        df["Source Port"] = 49152

    if "Protocol" in df.columns:
        df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(6)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if "Label" in df.columns:
        df["Label"] = df["Label"].fillna(df["family"])

    return df


@st.cache_resource
def load_models():
    bundles = {}
    for name, (_, filename) in MODELS.items():
        path = config.DATA_PROCESSED / filename
        if path.exists():
            bundle = joblib.load(path)
            model = bundle.get("model", bundle)
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            bundles[name] = bundle
    scaler = joblib.load(config.DATA_PROCESSED / "scaler.joblib")
    return bundles, scaler


def available_results(res):
    return {name: res[key] for name, (key, _) in MODELS.items() if key in res}


# UI helpers
def md(markup):
    """Render one block of HTML. Line breaks are removed so markdown never turns it into a code block."""
    st.markdown("".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


_card_count = 0


def card():
    """A bordered card. Each card gets its own key so the CSS can style it."""
    global _card_count
    _card_count += 1
    return st.container(key=f"card_{_card_count}")


def kpi(label, value, sub="", accent="#38bdf8"):
    md(
        f"<div class='kpi' style='border-left:4px solid {accent}'>"
        f"<div class='kl'>{esc(str(label))}</div><div class='kv'>{esc(str(value))}</div>"
        f"<div class='ks'>{esc(str(sub))}</div></div>"
    )


def section(text):
    md(f"<div class='section'>{text}</div>")


def page_header(title, sub):
    md(f"<div class='ph'><div class='ph-t'>{title}</div><div class='ph-s'>{sub}</div></div>")


def format_value(col, value, formats, float_fmt):
    fmt = formats.get(col)
    if fmt is not None:
        try:
            return fmt(value) if callable(fmt) else fmt.format(value)
        except (ValueError, TypeError):
            pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return "-" if pd.isna(value) else float_fmt.format(value)
    return str(value)


def table_cell(col, value, bars, formats, badges, float_fmt):
    text = esc(format_value(col, value, formats, float_fmt))
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if col in bars and is_number and pd.notna(value):
        width = max(0.0, min(1.0, float(value))) * 100
        return f"<div class='bar'><div class='fill' style='width:{width:.1f}%'></div><div class='num'>{text}</div></div>"
    colour = None
    if col in badges:
        try:
            colour = badges[col].get(value)
        except TypeError:
            colour = None
    if colour:
        return f"<span class='badge' style='color:{colour};background:{colour}22;border:1px solid {colour}55'>{text}</span>"
    return text


def html_table(df, bars=(), formats=None, badges=None, show_index=True, max_rows=500, float_fmt="{:,.3f}"):
    """A table drawn as HTML so it follows the light and dark theme."""
    formats = formats or {}
    badges = badges or {}
    shown = df.head(max_rows)
    cols = list(shown.columns)
    head = f"<th>{esc(str(shown.index.name or ''))}</th>" if show_index else ""
    head += "".join(f"<th>{esc(str(c))}</th>" for c in cols)
    body = []
    for idx, values in zip(shown.index, shown.itertuples(index=False, name=None)):
        cells = f"<td class='idx'>{esc(str(idx))}</td>" if show_index else ""
        for col, value in zip(cols, values):
            cells += "<td>" + table_cell(col, value, bars, formats, badges, float_fmt) + "</td>"
        body.append(f"<tr>{cells}</tr>")
    note = ""
    if len(df) > max_rows:
        note = f"<div class='tbl-note'>Showing the first {max_rows:,} of {len(df):,} rows</div>"
    return (
        f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>{note}"
    )


def show_table(df, container=None, **kwargs):
    target = container if container is not None else st
    target.markdown(html_table(df, **kwargs), unsafe_allow_html=True)


def show_chart(chart):
    t = theme()
    styled = (
        chart.configure(background="rgba(0,0,0,0)")
        .configure_axis(
            labelColor=t["muted"], titleColor=t["sub"], gridColor=t["grid"],
            domainColor=t["border"], tickColor=t["border"],
        )
        .configure_legend(labelColor=t["sub"], titleColor=t["sub"])
        .configure_view(stroke=None)
    )
    st.altair_chart(styled, width="stretch", theme=None)


def feature_input(feat, row, key, guide=None):
    """Dropdown for features in CHOICES (Protocol, PSH Flag Count), number box for the rest."""
    val = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else 0.0
    bounds = None
    if guide is not None and feat in guide.index:
        lower = float(guide.loc[feat, "Min"])
        upper = float(guide.loc[feat, "Max"])
        if np.isfinite(lower) and np.isfinite(upper) and lower <= upper:
            bounds = (lower, upper)

    if feat in CHOICES:
        options = list(CHOICES[feat].keys())
        if bounds is not None:
            options = [option for option in options if bounds[0] <= option <= bounds[1]]
        if not options:
            st.error(f"No valid choices for {feat} within its observed data range.")
            return val
        default = int(val) if int(val) in options else options[0]
        value = st.selectbox(
            feat,
            options,
            index=options.index(default),
            format_func=lambda o, f=feat: CHOICES[f][o],
            key=key,
        )
    elif bounds is not None:
        value = st.number_input(
            feat,
            min_value=bounds[0],
            max_value=bounds[1],
            value=val,
            key=key,
        )
    else:
        st.error(f"Could not determine a valid data range for {feat}.")
        return val

    if bounds is not None:
        st.caption(f"Allowed range in data: {bounds[0]:,.2f} to {bounds[1]:,.2f}")
    return value


def plot_confusion(cm, labels, title):
    import matplotlib.pyplot as plt

    t = theme()
    cm = np.array(cm)
    shares = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(t["plot_bg"])
    ax.set_facecolor(t["plot_bg"])
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(colors=t["sub"])
    ax.set_xlabel("Predicted family", color=t["sub"])
    ax.set_ylabel("True family", color=t["sub"])
    ax.set_title(title, color=t["text"])
    for s in ax.spines.values():
        s.set_color(t["border"])
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "#0b1220"
            ax.text(j, i, f"{shares[i, j]:.1%}\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=7, color=colour)
    fig.tight_layout()
    return fig


def family_donut(counts):
    df = counts.rename_axis("family").reset_index(name="rows")
    df["share"] = df["rows"] / df["rows"].sum()
    df["label"] = df["family"] + " (" + (df["share"] * 100).round(1).astype(str) + "%)"
    colors = [FAMILY_COLORS.get(f, "#38bdf8") for f in df["family"]]

    base = alt.Chart(df).encode(theta=alt.Theta("rows:Q", stack=True))
    arc = base.mark_arc(innerRadius=60, outerRadius=105).encode(
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=df["label"].tolist(), range=colors),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            "family",
            alt.Tooltip("rows:Q", format=",", title="Rows"),
            alt.Tooltip("share:Q", format=".1%", title="Share"),
        ],
    )
    text = (
        base.transform_filter(alt.datum.share >= 0.05)
        .mark_text(radius=83, color="white", fontWeight="bold", size=13)
        .encode(text=alt.Text("share:Q", format=".1%"))
    )
    return (arc + text).properties(height=270)


def proba_bars(probs):
    families = list(FAMILY_COLORS) + [family for family in probs if family not in FAMILY_COLORS]
    df = pd.DataFrame(
        {
            "family": families,
            "probability": [probs.get(family, 0.0) for family in families],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y(
                "family:N",
                scale=alt.Scale(domain=families),
                sort=families,
                title=None,
                axis=alt.Axis(labelLimit=120),
            ),
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
                legend=None,
            ),
            tooltip=["family", alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=240)
    )


def _model_key_for_name(name):
    mapping = {
        "k-NN": "knn",
        "Decision tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    return mapping.get(name, name)


def _predict_with_bundle(bundle, X_scaled):
    model = bundle.get("model", bundle)
    selector = bundle.get("selector")
    X_work = selector.transform(X_scaled) if selector is not None else X_scaled
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_work)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_work)
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        return scores
    return np.zeros((X_work.shape[0], 1))


def run_consensus(bundles, scaler, df_flows, cols, weights_by_model):
    if df_flows.empty:
        return df_flows.copy()
    X_scaled = scaler.transform(df_flows[cols]).astype("float32")
    all_labels = sorted({str(label) for label in list(FAMILY_COLORS.keys())})
    model_outputs = {}
    for model_name, bundle in bundles.items():
        proba = _predict_with_bundle(bundle, X_scaled)
        model = bundle.get("model", bundle)
        labels = list(get_model_classes(model, all_labels))
        if proba.ndim == 1:
            proba = np.asarray([proba])
        model_outputs[model_name] = {"proba": proba, "labels": labels}
        all_labels = sorted(set(all_labels) | set(labels))

    rows = []
    for idx in range(len(df_flows)):
        ensemble = {}
        winners = {}
        for model_name, payload in model_outputs.items():
            proba = np.asarray(payload["proba"][idx])
            labels = payload["labels"]
            scaled = {str(label): float(prob) for label, prob in zip(labels, proba)}
            winners[model_name] = max(scaled, key=lambda k: scaled[k], default="Benign")
            weight = float(weights_by_model.get(model_name, weights_by_model.get(_model_key_for_name(model_name), 1.0)))
            for label, prob in scaled.items():
                ensemble[label] = ensemble.get(label, 0.0) + weight * float(prob)
        if not ensemble:
            pred = "Benign"
            conf = 0.0
        else:
            total = sum(ensemble.values())
            pred = max(ensemble, key=lambda k: ensemble[k])
            conf = ensemble[pred] / total if total else 0.0
        votes = sum(1 for v in winners.values() if v == pred)
        total_models = max(1, len(model_outputs))
        consensus_ratio = votes / total_models
        if votes == total_models:
            label = f"{votes}/{total_models} Unanimous"
        elif votes > total_models / 2:
            label = f"{votes}/{total_models} Majority"
        else:
            label = f"{votes}/{total_models} Split Vote"
        rows.append({
            "ensemble_pred": pred,
            "ensemble_confidence": float(conf),
            "consensus_count": int(votes),
            "consensus_ratio": float(consensus_ratio),
            "consensus_label": label,
            "model_votes": winners,
        })

    out = df_flows.copy()
    for key in ["ensemble_pred", "ensemble_confidence", "consensus_count", "consensus_ratio", "consensus_label"]:
        out[key] = [row[key] for row in rows]
    out["model_votes"] = [row["model_votes"] for row in rows]
    return out


def resolve_action(pred_family, confidence, consensus_ratio, src_ip, blacklisted_ips, conf_threshold=0.70):
    ipset = set(blacklisted_ips or [])
    if src_ip in ipset:
        return (
            "DROPPED (BLACKLISTED)",
            " Drop packet immediately — Source IP is already on the active firewall blacklist",
        )
    if pred_family == "Benign" and (confidence < conf_threshold or consensus_ratio < 0.75):
        return (
            "INSPECT",
            "Hold for deep packet inspection & rate-limit (low confidence / model disagreement)",
        )
    action_map = {
        "Benign": ("ALLOW", "Allow the traffic"),
        "PortScan": ("ALERT", "Raise an alert, log Source → Destination scan, and throttle"),
        "BruteForce": ("BLOCK", "Block the Source IP at firewall and terminate session"),
        "DoS": ("RATE-LIMIT", "Rate-limit or block the Source IP"),
    }
    return action_map.get(pred_family, ("REVIEW", "Review manually"))


@st.cache_data
def benign_baseline(features):
    test = load_test()
    test = test[test["family"] == "Benign"].copy()
    rows = []
    for feature in features:
        s = test[feature].astype(float)
        std = float(s.std(ddof=0))
        if std < 1e-6:
            std = 1e-6
        rows.append({
            "Feature": feature,
            "mean": float(s.mean()),
            "std": std,
            "median": float(s.median()),
            "q05": float(s.quantile(0.05)),
            "q95": float(s.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def explain_flow(flow_series, features):
    baseline = benign_baseline(tuple(features))
    frame = []
    for feature in features:
        row = baseline[baseline["Feature"] == feature].iloc[0]
        value = float(flow_series.get(feature, 0.0))
        benign_std = float(row["std"])
        z_score = (value - float(row["mean"])) / benign_std
        abs_z = abs(z_score)
        direction = "above Benign normal" if z_score >= 0 else "below Benign normal"
        normal_range = f"{row['q05']:.2f} to {row['q95']:.2f}"
        frame.append({
            "Feature": feature,
            "Flow Value": value,
            "Normal Benign Range (5%-95%)": normal_range,
            "Z-Score": float(z_score),
            "Deviation Summary": f"{direction} ({abs_z:.2f}σ away from Benign median)",
        })
    out = pd.DataFrame(frame).sort_values("Z-Score", key=lambda s: s.abs(), ascending=False).head(6)
    out["abs_z"] = out["Z-Score"].abs()
    return out[["Feature", "Flow Value", "Normal Benign Range (5%-95%)", "Z-Score", "Deviation Summary"]]


@st.cache_data
def plot_sample(features):
    test = load_test()
    return test.sample(min(5000, len(test)), random_state=42)[list(features) + ["family"]]


def importance_chart(model, features, top=15):
    df = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    df = df.nlargest(top, "importance")
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color="#38bdf8")
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("importance:Q", title="Importance"),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=420)
    )


def family_boxplot(df, feature):
    d = df[["family", feature]].rename(columns={feature: "value"})
    lo, hi = d["value"].quantile([0.01, 0.99])  # hide extreme outliers
    d = d[(d["value"] >= lo) & (d["value"] <= hi)]

    stats = d.groupby("family")["value"].agg(
        lo="min",
        q1=lambda s: s.quantile(0.25),
        med="median",
        q3=lambda s: s.quantile(0.75),
        hi="max",
    ).reset_index()

    base = alt.Chart(stats).encode(
        x=alt.X("family:N", title=None, axis=alt.Axis(labelAngle=0))
    )
    color = alt.Color(
        "family:N",
        scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
        legend=None,
    )
    tips = [
        "family",
        alt.Tooltip("lo:Q", title="Min", format=",.2f"),
        alt.Tooltip("q1:Q", title="25%", format=",.2f"),
        alt.Tooltip("med:Q", title="Median", format=",.2f"),
        alt.Tooltip("q3:Q", title="75%", format=",.2f"),
        alt.Tooltip("hi:Q", title="Max", format=",.2f"),
    ]

    whisker = base.mark_rule(color=theme()["sub"], clip=True).encode(
        y=alt.Y("lo:Q", title=feature), y2="hi:Q", tooltip=tips)
    box = base.mark_bar(size=60, clip=True).encode(
        y="q1:Q", y2="q3:Q", color=color, tooltip=tips)
    median = base.mark_tick(color="white", size=60, thickness=2, clip=True).encode(
        y="med:Q", tooltip=tips)

    # Scroll wheel = zoom, drag = pan (up and down only)
    zoom = alt.selection_interval(bind="scales", encodings=["y"])

    return (whisker + box + median).add_params(zoom).properties(height=420)


def correlation_heatmap(df, features):
    corr = df[list(features)].corr().fillna(0)  # constant features give NaN, show them as 0
    long = corr.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="corr")
    long["val"] = long["corr"].map(lambda v: f"{v:.2f}")

    base = alt.Chart(long).encode(
        x=alt.X("b:N", title=None, sort=list(features), axis=alt.Axis(labelLimit=260)),
        y=alt.Y("a:N", title=None, sort=list(features), axis=alt.Axis(labelLimit=260)),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                        title="Correlation"),
        tooltip=["a", "b", alt.Tooltip("val:N", title="Correlation")],
    )
    text = base.mark_text(fontSize=10).encode(
        text="val:N",
        color=alt.condition("abs(datum.corr) > 0.5", alt.value("white"), alt.value("#0b1220")),
    )
    return (rect + text).properties(height=560)


def per_family_table(metrics):
    df = pd.DataFrame(metrics["per_family"]).T.rename(columns={"f1-score": "f1"})
    df.columns = [str(c).capitalize() for c in df.columns]
    df.index.name = "Family"
    return df


def run_model(bundle, scaler, flow, cols):
    X = scaler.transform(flow[cols]).astype("float32")
    model = bundle.get("model", bundle)
    selector = bundle.get("selector")
    X = selector.transform(X) if selector is not None else X
    proba = model.predict_proba(X)[0]
    classes = get_model_classes(model, ["Benign", "BruteForce", "DoS", "PortScan"])
    if len(classes) != len(proba):
        classes = [str(c) for c in model.classes_]
    k = int(np.argmax(proba))
    probs = {str(c): float(p) for c, p in zip(classes, proba)}
    return str(classes[k]), float(proba[k]), probs


def to_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


@st.cache_data
def feature_guide(features):
    test = load_test()
    rows = []
    for f in features:
        r = {"Feature": f, "Min": test[f].min(), "Max": test[f].max()}
        for fam, g in test.groupby("family")[f]:
            r[fam] = f"{g.quantile(0.05):,.2f} to {g.quantile(0.95):,.2f}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("Feature")


def random_flow_position(test):
    """A random flow from a random family. The family is never shown before predicting."""
    family = np.random.choice(test["family"].unique())
    return int(np.random.choice(np.where(test["family"] == family)[0]))


# Page: Home
@st.cache_data
def hero_art():
    """The cybersecurity banner, drawn as an SVG so no image file is needed."""
    nodes = [
        (90, 80, "#ef4444"), (130, 400, "#22c55e"), (700, 90, "#f59e0b"),
        (720, 390, "#22c55e"), (60, 240, "#ef4444"), (745, 240, "#22c55e"),
    ]
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500'>"
        "<defs>"
        "<linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
        "<stop offset='0' stop-color='#38bdf8'/><stop offset='1' stop-color='#8b5cf6'/></linearGradient>"
        "<radialGradient id='glow'><stop offset='0' stop-color='#6366f1' stop-opacity='.55'/>"
        "<stop offset='1' stop-color='#6366f1' stop-opacity='0'/></radialGradient>"
        "<pattern id='grid' width='40' height='40' patternUnits='userSpaceOnUse'>"
        "<path d='M40 0H0V40' fill='none' stroke='#38bdf8' stroke-opacity='.08'/></pattern>"
        "</defs>"
        "<rect width='800' height='500' rx='24' fill='#0a1020'/>"
        "<rect width='800' height='500' rx='24' fill='url(#grid)'/>"
        "<circle cx='400' cy='250' r='230' fill='url(#glow)'/>"
    )
    for i, (x, y, c) in enumerate(nodes):
        svg += (
            f"<line x1='{x}' y1='{y}' x2='400' y2='250' stroke='{c}' stroke-opacity='.4' stroke-dasharray='5 6'/>"
            f"<circle cx='{x}' cy='{y}' r='9' fill='{c}'>"
            f"<animate attributeName='r' values='7;11;7' dur='{2 + i * 0.4:.1f}s' repeatCount='indefinite'/></circle>"
            f"<circle r='4' fill='{c}'>"
            f"<animateMotion dur='{2.5 + i * 0.5:.1f}s' repeatCount='indefinite' path='M{x} {y} L400 250'/></circle>"
        )
    svg += (
        "<circle cx='400' cy='250' r='165' fill='none' stroke='#38bdf8' stroke-opacity='.3' stroke-dasharray='3 10'>"
        "<animateTransform attributeName='transform' type='rotate' from='0 400 250' to='360 400 250' "
        "dur='40s' repeatCount='indefinite'/></circle>"
        "<path d='M400 110 L520 150 V250 C520 330 465 385 400 410 C335 385 280 330 280 250 V150 Z' "
        "fill='url(#g)' fill-opacity='.2' stroke='url(#g)' stroke-width='4'/>"
        "<rect x='365' y='244' width='70' height='56' rx='9' fill='url(#g)'/>"
        "<path d='M378 244 V226 a22 22 0 0 1 44 0 V244' fill='none' stroke='#e0f2fe' stroke-width='6' stroke-linecap='round'/>"
        "<circle cx='400' cy='265' r='7' fill='#0a1020'/><rect x='397' y='268' width='6' height='16' rx='3' fill='#0a1020'/>"
        "<text x='400' y='470' text-anchor='middle' fill='#7dd3fc' font-size='15' letter-spacing='6' "
        "font-family='Arial, sans-serif'>THREAT DETECTION ACTIVE</text>"
        "</svg>"
    )
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def home_stats():
    stats = read_json("preprocessing_stats.json")
    res = read_json("metrics_knn.json")
    items = []
    if stats:
        items.append(("Clean flows", f"{stats['rows_clean']:,}", "after cleaning", "#38bdf8"))
    items.append(("Traffic families", "4", "Benign, PortScan, BruteForce, DoS", "#22c55e"))
    if res:
        results = available_results(res)
        if results:
            best = max(results, key=lambda n: results[n]["macro_f1"])
            items.append(("Detection models", str(len(results)), ", ".join(results), "#a78bfa"))
            items.append(("Best macro F1", f"{results[best]['macro_f1']:.3f}", best, "#f59e0b"))
    return items


def page_home():
    art = hero_art()
    md(
        "<div class='hero2'>"
        "<div class='hero2-text'>"
        "<div class='hero2-title'>SENTRIX</div>"
        "<div class='hero2-tag'>Autonomous Cyber-Defense &amp; Intrusion Detection System</div>"
        "<div class='hero2-sub'>SENTRIX studies network traffic with machine learning, recognises attacks such as "
        "port scans, brute force and denial of service, and decides how to respond on its own.</div>"
        "<span class='hchip'>CICIDS2017 dataset</span><span class='hchip'>4 ML models</span>"
        "<span class='hchip'>Ensemble voting</span><span class='hchip'>Automatic response</span>"
        "</div>"
        f"<div class='hero2-art'><img src='data:image/svg+xml;base64,{art}' alt='Cyber security illustration'/></div>"
        "</div>"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Try live detection", type="primary", width="stretch", key="cta_live",
                  on_click=go, args=("Live Detection",))
    with c2:
        st.button("Run a simulation", width="stretch", key="cta_sim", on_click=go, args=("Simulation",))
    with c3:
        st.button("Explore the data", width="stretch", key="cta_data", on_click=go, args=("Overview",))

    items = home_stats()
    st.write("")
    k = st.columns(len(items))
    for col, (label, value, sub, accent) in zip(k, items):
        with col:
            kpi(label, value, sub, accent)

    section("What you can do")
    md(
        "<div class='grid g4'>"
        "<div class='fcard' style='--c:#38bdf8'><div class='fi'>📊</div><div class='ft'>Explore the data</div>"
        "<div class='fd'>See the traffic mix, which features matter most, and how they relate to each other.</div></div>"
        "<div class='fcard' style='--c:#22c55e'><div class='fi'>🤖</div><div class='ft'>Compare the models</div>"
        "<div class='fd'>Check k-NN, decision tree, random forest and XGBoost on macro F1, accuracy and confusion matrices.</div></div>"
        "<div class='fcard' style='--c:#f59e0b'><div class='fi'>🚨</div><div class='ft'>Detect a flow</div>"
        "<div class='fd'>Load or edit one network flow and get the predicted family, confidence and recommended action.</div></div>"
        "<div class='fcard' style='--c:#a78bfa'><div class='fi'>🛰️</div><div class='ft'>Simulate traffic</div>"
        "<div class='fd'>Run a batch or a live stream through the defense, or upload your own flow file.</div></div>"
        "</div>"
    )

    section("How SENTRIX works")
    md(
        "<div class='grid g4'>"
        "<div class='fcard'><span class='stepn'>1</span><div class='ft'>Clean the traffic</div>"
        "<div class='fd'>Raw CICIDS2017 flows are cleaned, de-duplicated and split into train and test sets.</div></div>"
        "<div class='fcard'><span class='stepn'>2</span><div class='ft'>Pick the best features</div>"
        "<div class='fd'>The most informative flow features are selected and scaled for the models.</div></div>"
        "<div class='fcard'><span class='stepn'>3</span><div class='ft'>Classify with voting</div>"
        "<div class='fd'>Four models each give probabilities. They are combined, weighted by their macro F1.</div></div>"
        "<div class='fcard'><span class='stepn'>4</span><div class='ft'>Respond automatically</div>"
        "<div class='fd'>Each prediction maps to an action, and repeat attackers are blocked by IP.</div></div>"
        "</div>"
    )

    section("Traffic families and responses")
    cards = ""
    for family, colour in FAMILY_COLORS.items():
        cards += (
            f"<div class='fcard' style='--c:{colour}'><div class='ft' style='color:{colour}'>{family}</div>"
            f"<div class='fd'>{FAMILY_INFO[family]}</div>"
            f"<div class='fa'>Response: {ACTIONS[family]}</div></div>"
        )
    md(f"<div class='grid g4'>{cards}</div>")


# Page: Overview
def page_data():
    page_header("Overview", "The cleaned dataset, the traffic mix and what the models learn from it.")
    stats = read_json("preprocessing_stats.json")
    if stats is None:
        st.error("Run `python -m src.preprocess.clean` first.")
        return
    res = read_json("metrics_knn.json")
    n_total = len(feature_columns())
    n_used = len(res["features_used"]) if res else "-"

    section("Dataset summary")
    k = st.columns(4)
    with k[0]:
        kpi("Raw rows", f"{stats['rows_raw']:,}", "4 CICIDS2017 files", "#38bdf8")
    with k[1]:
        kpi("Clean rows", f"{stats['rows_clean']:,}", "after cleaning", "#22c55e")
    with k[2]:
        kpi("Our sample", f"{stats['rows_sample']:,}", f"{stats['sample_fraction']:.1%} of clean rows", "#a78bfa")
    with k[3]:
        kpi("Features", f"{n_used} of {n_total}", "selected for the models", "#f59e0b")

    k = st.columns(4)
    with k[0]:
        kpi("Missing / inf rows removed", f"{stats['missing_removed']:,}", "dropped during cleaning", "#ef4444")
    with k[1]:
        kpi("Duplicates removed", f"{stats['duplicates_removed']:,}", "repeated rows", "#ef4444")
    with k[2]:
        kpi("Train / test split", "80 / 20", "same family share in both", "#38bdf8")
    with k[3]:
        kpi("Attack families", "4", "Benign, BruteForce, DoS, PortScan", "#22c55e")

    section("Data details")
    left, right = st.columns([3, 2])
    with left:
        with card():
            st.markdown("##### Files used")
            files = pd.DataFrame(
                [{"File": n, "Rows": v["rows"], "Columns": v["columns"]}
                 for n, v in stats["files"].items()]
            )
            show_table(files, show_index=False)
    with right:
        with card():
            st.markdown("##### Traffic by family")
            show_chart(family_donut(load_family_counts()))

    if res:
        features = res["features_used"]
        bundles, _ = load_models()
        sample = plot_sample(tuple(features))

        section("Data and model insights")
        with card():
            t1, t2, t3 = st.tabs(["Feature importance", "Feature by family", "Correlation"])

            with t1:
                options = [n for n in ("Random Forest", "XGBoost") if n in bundles]
                if options:
                    pick = st.radio("Model", options, horizontal=True)
                    model = bundles[pick]["model"]
                    if len(model.feature_importances_) == len(features):
                        show_chart(importance_chart(model, features))
                        st.caption("Longer bar = the model relies on that feature more.")
                    else:
                        st.warning("Feature count does not match the saved model.")
                else:
                    st.info("No Random Forest or XGBoost model file found.")

            with t2:
                feat = st.selectbox("Feature", features)
                show_chart(family_boxplot(sample, feat))
                st.caption("Scroll over the chart to zoom, drag to move up or down, "
                           "double-click to reset.")

            with t3:
                show_chart(correlation_heatmap(sample, features))
                st.caption("Dark red or blue = strongly related features, which carry similar information.")


# Page: Models
def page_models():
    page_header("Models", "How each classifier performs on the held-out test set.")
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run `python -m src.supervised.models` first.")
        return
    labels = res.get("labels", ["Benign", "BruteForce", "DoS", "PortScan"])
    results = available_results(res)
    ranked = sorted(results, key=lambda n: results[n]["macro_f1"], reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else best

    section("Best model")
    k = st.columns(4)
    with k[0]:
        kpi("Best model", best, "highest macro F1", "#22c55e")
    with k[1]:
        kpi("Macro F1", f"{results[best]['macro_f1']:.4f}", "all families weighted equally", "#38bdf8")
    with k[2]:
        kpi("Accuracy", f"{results[best]['accuracy']:.2%}", "on the test set", "#a78bfa")
    with k[3]:
        kpi("Models compared", f"{len(results)} classifiers + baseline", f"Runner-up: {runner}", "#f59e0b")

    section("Test-set scores")
    rows = {"Always say Benign": res["majority_baseline"], **results}
    comp = pd.DataFrame(
        [{"Macro F1": m["macro_f1"], "Accuracy": m["accuracy"], "model": n} for n, m in rows.items()]
    ).set_index("model")
    comp.index.name = "Model"
    with card():
        show_table(comp, bars=("Macro F1", "Accuracy"), float_fmt="{:.4f}")
        st.caption("Most traffic is Benign, so accuracy alone is misleading. "
                   "Macro F1 gives every family equal weight.")

    section("Per-family results")
    model_names = list(MODELS) if all(name in results for name in MODELS) else list(results)
    with card():
        tabs = st.tabs(model_names)
        for tab, name in zip(tabs, model_names):
            with tab:
                metrics = results.get(name)
                if not metrics:
                    continue
                c1, c2 = st.columns(2)
                with c1:
                    df = per_family_table(metrics)
                    bar_cols = [c for c in ("Precision", "Recall", "F1") if c in df.columns]
                    show_table(df, bars=bar_cols, formats={"Support": "{:,.0f}"})
                with c2:
                    if "confusion" in metrics:
                        fig = plot_confusion(metrics["confusion"], labels, name)
                        st.pyplot(fig)
                        import matplotlib.pyplot as plt
                        plt.close(fig)

    if "tuning" in res:
        with st.expander("k-NN: choosing k (macro F1 on validation data)"):
            tune = pd.DataFrame(res["tuning"])
            pivot = tune.pivot(index="k", columns="weights", values="macro_f1")
            show_table(pivot, float_fmt="{:.4f}")
            st.success(f"Best: k = {res['knn']['k']}, weights = {res['knn']['weights']}")

    st.info(
        "Limitations: only 4 families because we used 4 files (no web attacks, bot or "
        "infiltration). The train/test split is random. The weakest family is BruteForce."
    )


# Page: Live Detection
def page_predict():
    page_header("Live Detection", "Load or edit a single network flow and see how SENTRIX classifies it.")
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run the k-NN script first.")
        return

    chosen = res["features_used"]
    cols = feature_columns()
    test = load_test()
    bundles, scaler = load_models()
    guide = feature_guide(tuple(chosen))
    results = available_results(res)
    usable = [n for n in results if n in bundles]
    if not usable:
        st.error("No saved model files found. Re-run the training scripts.")
        return
    best_name = max(usable, key=lambda n: results[n]["macro_f1"])

    if "pos" not in st.session_state:
        st.session_state.pos = random_flow_position(test)
    pos = st.session_state.pos
    row = test.iloc[pos]

    def load_scenario(family):
        family_rows = test[test["family"] == family]
        if family_rows.empty:
            family_rows = test
        idx = int(np.random.choice(np.arange(len(family_rows))))
        st.session_state.pos = int(family_rows.index[idx])
        flow = test.loc[st.session_state.pos]
        flow_df = flow[cols].astype("float64").to_frame().T
        for c in flow_df.columns:
            if c in flow and pd.notna(flow[c]):
                flow_df[c] = float(flow[c]) if pd.notna(flow[c]) else 0.0
        pred, conf, probs = run_model(bundles[best_name], scaler, flow_df, cols)
        st.session_state.result = {"pos": st.session_state.pos, "pred": pred, "conf": conf, "probs": probs, "outside": []}

    scenario_buttons = [
        ("Random flow", None),
        ("Benign flow", "Benign"),
        ("PortScan flow", "PortScan"),
        ("BruteForce flow", "BruteForce"),
        ("DoS flow", "DoS"),
    ]
    bar = st.columns(len(scenario_buttons))
    for i, (label, family) in enumerate(scenario_buttons):
        with bar[i]:
            if st.button(label, key=f"scenario_{i}", width="stretch"):
                if family is None:
                    st.session_state.pos = random_flow_position(test)
                    row = test.iloc[st.session_state.pos]
                    pred, conf, probs = run_model(bundles[best_name], scaler, row[cols].astype("float64").to_frame().T, cols)
                    st.session_state.result = {"pos": st.session_state.pos, "pred": pred, "conf": conf, "probs": probs, "outside": []}
                else:
                    load_scenario(family)

    pos = st.session_state.pos
    row = test.iloc[pos]
    src_ip = row.get("Source IP", "unknown")
    dst_ip = row.get("Destination IP", "192.168.10.50")
    dst_port = int(row.get("Destination Port", 80))
    src_port = int(row.get("Source Port", 49152))
    protocol = row.get("Protocol", 6)

    section("Selected flow")
    md(
        f"<div class='flowhdr'>"
        f"<div class='flowhdr-l'>Network flow header</div>"
        f"<div class='flowhdr-v'>{esc(str(src_ip))}:{src_port} ──► {esc(str(dst_ip))}:{dst_port} ({format_service(dst_port)})</div>"
        f"<div class='flowhdr-s'>Protocol: {protocol} · PSH Flag: {row.get('PSH Flag Count', 0)}</div></div>"
    )

    k = st.columns(3)
    with k[0]:
        kpi("Source IP", src_ip, "origin of this flow", "#a78bfa")
    with k[1]:
        kpi("Detector", best_name, "highest macro F1", "#22c55e")
    with k[2]:
        kpi("Detector macro F1", f"{results[best_name]['macro_f1']:.4f}", "on the test set", "#38bdf8")

    left, right = st.columns([3, 2])
    with left:
        with card():
            st.markdown("##### Flow features")
            with st.form("flow_form"):
                top_features = chosen[:6]
                grid = st.columns(3)
                inputs = {}
                for i, feat in enumerate(top_features):
                    with grid[i % 3]:
                        inputs[feat] = feature_input(feat, row, f"{pos}_{feat}", guide)
                with st.expander("Advanced flow features (14 more parameters)"):
                    advanced = st.columns(3)
                    for i, feat in enumerate(chosen[6:20]):
                        with advanced[i % 3]:
                            inputs[feat] = feature_input(feat, row, f"adv_{pos}_{feat}", guide)
                submitted = st.form_submit_button("Analyze flow", type="primary")

            if submitted:
                values = {f: float(v) for f, v in inputs.items() if v is not None}
                invalid = []
                for feat, value in values.items():
                    if feat not in guide.index:
                        invalid.append(f"{feat} (range unavailable)")
                        continue
                    lower = float(guide.loc[feat, "Min"])
                    upper = float(guide.loc[feat, "Max"])
                    if (
                        not np.isfinite(value)
                        or not np.isfinite(lower)
                        or not np.isfinite(upper)
                        or value < lower
                        or value > upper
                    ):
                        invalid.append(f"{feat} ({lower:,.2f} to {upper:,.2f})")
                if invalid:
                    st.error(
                        "Values must be within each feature's observed data range. "
                        "Correct: " + ", ".join(invalid)
                    )
                else:
                    flow = row[cols].astype("float64").to_frame().T
                    for f, v in values.items():
                        flow[f] = v
                    pred, conf, probs = run_model(bundles[best_name], scaler, flow, cols)
                    st.session_state.result = {"pos": pos, "pred": pred, "conf": conf, "probs": probs, "outside": []}

    with right:
        with card():
            st.markdown("##### Detection result")
            result = st.session_state.get("result")
            if result is None or result["pos"] != pos:
                st.info("Edit any feature if you like, then press **Analyze flow**.")
            else:
                pred = result["pred"]
                colour = FAMILY_COLORS.get(pred, "#38bdf8")
                md(
                    f"<div class='verdict' style='border-color:{colour}; background:{colour}22'>"
                    f"<div class='vl'>PREDICTED FAMILY</div>"
                    f"<div class='vv' style='color:{colour}'>{pred}</div>"
                    f"<div class='vs'>{result['conf']:.0%} confident · {best_name}</div>"
                    f"<span class='pill' style='background:{colour}'>{ACTION_TAG.get(pred, 'REVIEW')}</span>"
                    f"</div>"
                )
                md(
                    f"<div class='action'><div class='kl'>Autonomous action</div><b>{ACTIONS.get(pred, 'Review manually')}</b>"
                    f"<br><span style='color:var(--muted)'>Attack path: {esc(format_attack_route(row))}</span></div>"
                )
                show_chart(proba_bars(result["probs"]))
                if result["outside"]:
                    st.warning("Outside the range seen in the data (prediction may be unreliable): " + ", ".join(result["outside"]))

                consensus = run_consensus({best_name: bundles[best_name]}, scaler, pd.DataFrame([row]), cols, {best_name: results[best_name]["macro_f1"]})
                if not consensus.empty:
                    st.caption(f"Consensus: {consensus.iloc[0]['consensus_label']}")

    with st.expander("Which features should I change? (typical values per family)"):
        st.caption("Each column shows the usual range (5th to 95th percentile) of that feature for one family.")
        show_table(guide, formats={"Min": "{:,.2f}", "Max": "{:,.2f}"})


# Page: Simulation
def page_simulation():
    page_header("Simulation", "Run traffic through the autonomous defense and watch how it responds.")
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run the k-NN script first.")
        return

    cols = feature_columns()
    test = load_test()
    bundles, scaler = load_models()
    results = available_results(res)
    usable = [n for n in results if n in bundles]
    if not usable:
        st.error("No saved model files found. Re-run the training scripts.")
        return
    best_name = max(usable, key=lambda n: results[n]["macro_f1"])
    weights = {k: v["macro_f1"] for k, v in results.items()}
    family_badges = {"Predicted": FAMILY_COLORS, "Actual": FAMILY_COLORS, "Action": ACTION_COLORS}

    tabs = st.tabs(["Batch simulation", "Real-time stream", "Upload CSV / Parquet"])

    with tabs[0]:
        st.caption(f"The system analyses random flows with all available models (best single model: {best_name}) and applies a response to each one.")
        n = st.slider("Number of flows to analyse", 20, 300, 100, step=20)
        threshold = 0.75
        stateful = st.checkbox("Enable stateful IP auto-blocking (automatically drop future packets from blocked IPs)", value=True)
        run = st.button("Run simulation", type="primary")

        if run:
            with st.spinner("Analysing traffic..."):
                sample = test.sample(n, random_state=42)
                frame = run_consensus(bundles, scaler, sample.copy(), cols, weights)
                blacklisted = set()
                log_rows = []
                for _, row in frame.iterrows():
                    src_ip = row.get("Source IP", "unknown")
                    pred = row["ensemble_pred"]
                    conf = float(row["ensemble_confidence"])
                    consensus = float(row["consensus_ratio"])
                    action = resolve_action(pred, conf, consensus, src_ip, blacklisted, threshold)[0]
                    if pred in {"BruteForce", "DoS"}:
                        blacklisted.add(src_ip)
                    if stateful and src_ip in blacklisted and pred == "Benign":
                        action = "DROPPED (BLACKLISTED)"
                    if pred == "Benign" and conf < threshold:
                        action = "INSPECT"
                    log_rows.append({
                        "Source IP": src_ip,
                        "Destination IP": row.get("Destination IP", "192.168.10.50"),
                        "Dest Port / Service": format_service(row.get("Destination Port", 80)),
                        "Predicted": pred,
                        "Consensus": row["consensus_label"],
                        "Confidence": conf,
                        "Action": action,
                        "Actual": row.get("family", "Unknown"),
                    })
                log = pd.DataFrame(log_rows)
                log["Correct"] = log["Predicted"] == log["Actual"]
                st.session_state.sim = log

        log = st.session_state.get("sim")
        if log is None:
            st.info("Press **Run simulation** to start.")
        else:
            threats = log[log["Predicted"] != "Benign"]
            blocked = log[log["Action"].isin(["BLOCK", "RATE-LIMIT", "DROPPED (BLACKLISTED)"])]
            missed = log[(log["Actual"] != "Benign") & (log["Predicted"] == "Benign")]
            k = st.columns(5)
            with k[0]:
                kpi("Flows Analysed", f"{len(log):,}", "traffic inspected", "#38bdf8")
            with k[1]:
                kpi("Threats Detected", f"{len(threats):,}", "attack flows", "#f59e0b")
            with k[2]:
                kpi("Packets Auto-Dropped", f"{len(blocked):,}", "blacklisted packets", "#ef4444")
            with k[3]:
                kpi("Number of IP", f"{log['Source IP'].nunique():,}", "distinct sources", "#a78bfa")
            with k[4]:
                kpi("Attacks Missed", f"{len(missed):,}", "predicted benign", "#22c55e")

            section("Traffic log")
            show_table(
                log,
                show_index=False,
                formats={"Confidence": "{:.1%}"},
                badges={**family_badges, "Correct": {True: "#22c55e", False: "#94a3b8"}},
            )

            summary = (log[log["Predicted"] != "Benign"].groupby("Source IP").agg(
                AttackType=("Predicted", lambda s: ", ".join(sorted(set(s)))),
                PacketCount=("Predicted", "size"),
                AvgConfidence=("Confidence", "mean"),
                MitigationStatus=("Action", lambda s: ", ".join(sorted(set(s))))
            ).reset_index())
            if not summary.empty:
                section("Attackers by source IP")
                show_table(summary, show_index=False, formats={"AvgConfidence": "{:.1%}"})

            blocked_ips = sorted({ip for ip in log[log["Action"].str.contains("BLACKLISTED|BLOCK|RATE-LIMIT", na=False)]["Source IP"]})
            if blocked_ips:
                section("Suggested firewall rules")
                st.code("\n".join([f"iptables -A INPUT -s {ip} -j DROP" for ip in blocked_ips]))

    with tabs[1]:
        st.caption("Real-time traffic stream that updates in place. Each packet is classified by the model ensemble.")
        packets = st.slider("Packets to stream", 10, 100, 25, step=5)
        speed = st.slider("Stream speed (seconds per packet)", 0.1, 1.5, 0.3, step=0.1)
        start = st.button("Start live traffic stream", type="primary")
        if start:
            stream = test.sample(packets, random_state=42).reset_index(drop=True)
            scored = run_consensus(bundles, scaler, stream.copy(), cols, weights)
            status = st.empty()
            feed = st.empty()
            history = []
            for i, row in scored.iterrows():
                src_ip = row.get("Source IP", "unknown")
                dst_ip = row.get("Destination IP", "192.168.10.50")
                dst_port = int(row.get("Destination Port", 80))
                pred = row["ensemble_pred"]
                tag = ACTION_TAG.get(pred, "REVIEW")
                route = f"[{i + 1}] {src_ip}:{int(row.get('Source Port', 49152))} ──► {dst_ip}:{dst_port} ({format_service(dst_port)})"
                status.markdown(f"**Current packet:** {route}")
                history.append({
                    "Packet": i + 1,
                    "Source": src_ip,
                    "Destination": f"{dst_ip}:{dst_port}",
                    "Service": format_service(dst_port),
                    "Predicted": pred,
                    "Action": tag,
                })
                recent = pd.DataFrame(history[-8:][::-1])
                feed.markdown(
                    html_table(recent, show_index=False, formats={"Packet": "{}"}, badges=family_badges),
                    unsafe_allow_html=True,
                )
                time.sleep(speed)

    with tabs[2]:
        st.caption("Upload a CSV or parquet file and run the ensemble against it.")
        sample_csv = test.head(50).copy()
        sample_csv = sample_csv.drop(columns=[c for c in ["Label", "family"] if c in sample_csv.columns])
        csv_bytes = sample_csv.to_csv(index=False).encode("utf-8")
        st.download_button("Download sample unlabeled traffic CSV (50 rows)", data=csv_bytes, file_name="sample_network_traffic.csv", mime="text/csv")
        uploaded = st.file_uploader("Upload network flow file (.csv or .parquet)", type=["csv", "parquet"])
        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_parquet(uploaded)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()].copy()
            if "Fwd Header Length.1" in df.columns:
                df = df.drop(columns=["Fwd Header Length.1"])
            df = df.replace([np.inf, -np.inf], np.nan)
            for feature in cols:
                if feature not in df.columns:
                    df[feature] = test[feature].median()
            for feature in cols:
                df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
            df = df.fillna(0.0)
            results_df = run_consensus(bundles, scaler, df[cols].copy(), cols, weights)
            k = st.columns(4)
            with k[0]:
                kpi("Rows Uploaded", f"{len(df):,}", "input records", "#38bdf8")
            with k[1]:
                kpi("Benign Flows", f"{(results_df['ensemble_pred'] == 'Benign').sum():,}", "low-risk traffic", "#22c55e")
            with k[2]:
                kpi("Attack Flows Detected", f"{(results_df['ensemble_pred'] != 'Benign').sum():,}", "flagged records", "#f59e0b")
            with k[3]:
                kpi("Unique Source → Destination Attack Routes", f"{results_df[results_df['ensemble_pred'] != 'Benign'].shape[0]:,}", "attack paths", "#ef4444")

            attack_rows = results_df[results_df["ensemble_pred"] != "Benign"].copy()
            if not attack_rows.empty:
                shown = attack_rows[["ensemble_pred", "ensemble_confidence", "consensus_label"]].rename(
                    columns={"ensemble_pred": "Predicted", "ensemble_confidence": "Confidence", "consensus_label": "Consensus"}
                )
                shown.index.name = "Row"
                show_table(shown, formats={"Confidence": "{:.1%}"}, badges={"Predicted": FAMILY_COLORS})
                st.download_button("Download labeled predictions CSV", data=results_df.to_csv(index=False).encode("utf-8"), file_name="labeled_predictions.csv", mime="text/csv")


# Page: About Us
def page_about():
    page_header("About Us", "What SENTRIX is, how it works and who built it.")

    section("The project")
    with card():
        md(
            "<div class='prose'>SENTRIX is a machine-learning intrusion detection system. It reads network flows, "
            "decides whether each one is normal traffic or an attack, and chooses a defensive response without waiting "
            "for a person. It is trained and tested on the CICIDS2017 dataset, using four of its capture files, and covers "
            "four traffic families: Benign, PortScan, BruteForce and DoS.</div>"
        )

    section("What it does and how")
    left, right = st.columns(2)
    with left:
        md(
            "<div class='fcard' style='--c:#22c55e'><div class='ft'>What it does</div>"
            "<div class='fd'>Classifies each network flow into one of four families.<br>"
            "Shows the confidence and how much the models agree.<br>"
            "Maps every prediction to an action: allow, alert, block or rate-limit.<br>"
            "Blocks repeat offenders by source IP and suggests firewall rules.</div></div>"
        )
    with right:
        md(
            "<div class='fcard' style='--c:#38bdf8'><div class='ft'>How it works</div>"
            "<div class='fd'>Flows are cleaned, de-duplicated and scaled, then the most informative features are chosen.<br>"
            "k-NN, decision tree, random forest and XGBoost each predict probabilities.<br>"
            "The probabilities are combined, weighted by each model's macro F1.<br>"
            "The vote count shows whether the decision was unanimous, a majority or split.</div></div>"
        )

    section("Built with")
    chips = ["Python", "Streamlit", "scikit-learn", "XGBoost", "pandas", "NumPy", "Altair", "Matplotlib", "joblib", "CICIDS2017"]
    md("<div>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div>")

    section("Limits to keep in mind")
    with card():
        md(
            "<div class='prose'>Only four families are covered because only four dataset files were used, so web attacks, "
            "bots and infiltration are not detected. The train and test split is random, which can make scores look "
            "better than they would on traffic from a different network. BruteForce is the hardest family to separate.</div>"
        )

    section("Our team")
    cards = ""
    for member in TEAM:
        initials = "".join(part[0] for part in member["name"].split()[:2]).upper()
        cards += (
            f"<div class='fcard'><div class='avatar'>{esc(initials)}</div>"
            f"<div class='ft'>{esc(member['name'])}</div><div class='fd'>{esc(member['role'])}</div></div>"
        )
    md(f"<div class='grid g4'>{cards}</div>")


PAGES = {
    "Home": page_home,
    "Overview": page_data,
    "Models": page_models,
    "Live Detection": page_predict,
    "Simulation": page_simulation,
}


# Main
def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"

    st.markdown(theme_vars() + CSS, unsafe_allow_html=True)

    with st.container(key="topbar"):
        brand, nav, toggle = st.columns([1.6, 6, 1.5], vertical_alignment="center")
        with brand:
            md("<div class='brand'><span class='logo'>🛡️</span><span class='brand-t'>SENTRIX</span></div>")
        with nav:
            st.radio("Navigation", list(PAGES), horizontal=True, key="nav", label_visibility="collapsed")
        with toggle:
            is_dark = st.session_state.theme == "dark"
            st.button(
                "☀️  Light mode" if is_dark else "🌙  Dark mode",
                key="theme_btn",
                on_click=toggle_theme,
                width="stretch",
            )

    PAGES[st.session_state.nav]()

    md(
        "<div class='footer'>SENTRIX · Autonomous Cyber-Defense &amp; Intrusion Detection System · "
        "Trained on CICIDS2017</div>"
    )


main()