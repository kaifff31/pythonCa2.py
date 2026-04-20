# -*- coding: utf-8 -*-
"""
Analyzer — Advanced MLR Analytics Dashboard
Theme: Deep Slate & Emerald · Amber Accents
Universal Dataset Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Analyzer · ML Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;700&family=Manrope:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: #07100f;
    color: #b8cdc8;
}
.main .block-container {
    background: #07100f;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091512 0%, #0c1a17 100%) !important;
    border-right: 1px solid rgba(52, 211, 153, 0.15) !important;
}
section[data-testid="stSidebar"] * { color: #8ab8ad !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #3d7a6a !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}

.hero-banner {
    background: linear-gradient(135deg, #091512 0%, #0d1e1a 40%, #0f2218 100%);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 20px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(52, 211, 153, 0.07), inset 0 1px 0 rgba(255,255,255,0.03);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(251, 191, 36, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #e8f5f0;
    margin: 0;
    line-height: 1;
    letter-spacing: -0.05em;
}
.hero-accent {
    background: linear-gradient(90deg, #34d399, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { color: #3d7a6a; font-size: 0.88rem; margin-top: 0.5rem; font-weight: 500; }
.hero-pill-row { display: flex; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap; }
.hero-pill {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.2);
    color: #5ecda8 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.68rem;
    padding: 0.28rem 0.85rem;
    border-radius: 6px;
    letter-spacing: 0.04em;
}
.hero-badge {
    background: linear-gradient(135deg, #059669, #0d9488);
    color: #e8f5f0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.65rem 1.4rem;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.35);
    white-space: nowrap;
}

.metric-card {
    background: linear-gradient(135deg, #0b1915 0%, #0e1e19 100%);
    border: 1px solid rgba(52, 211, 153, 0.14);
    border-radius: 14px;
    padding: 1.4rem 1.3rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.02);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.4), transparent);
}
.metric-label {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #2d5c4f;
    margin-bottom: 0.6rem;
    font-weight: 700;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #34d399;
    line-height: 1;
}
.metric-value.green { color: #34d399; }
.metric-value.amber { color: #fbbf24; }
.metric-value.red   { color: #f87171; }
.metric-value.blue  { color: #22d3ee; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8f5f0;
    padding: 0.5rem 0 0.5rem 1rem;
    margin: 2rem 0 1.2rem 0;
    border-left: 3px solid #34d399;
    letter-spacing: -0.01em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: rgba(52, 211, 153, 0.05);
    padding: 5px;
    border-radius: 12px;
    border: 1px solid rgba(52, 211, 153, 0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.82rem;
    color: #3d6b5e;
    padding: 0.48rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: #e8f5f0 !important;
    box-shadow: 0 2px 12px rgba(5, 150, 105, 0.35) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: #e8f5f0 !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(5, 150, 105, 0.5) !important;
    transform: translateY(-2px) !important;
}

.stTextInput input, .stNumberInput input {
    background: #0b1915 !important;
    border: 1px solid rgba(52, 211, 153, 0.18) !important;
    color: #c8e8e0 !important;
    border-radius: 9px !important;
    font-family: 'Manrope', sans-serif !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(52, 211, 153, 0.25) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    background: rgba(52, 211, 153, 0.03) !important;
}

.stDataFrame {
    border-radius: 12px !important;
    border: 1px solid rgba(52, 211, 153, 0.12) !important;
    overflow: hidden;
}

hr { border-color: rgba(52, 211, 153, 0.1) !important; }
p, li, label, .stMarkdown { color: #7aada0 !important; }
h1, h2, h3 { color: #e8f5f0 !important; font-family: 'Syne', sans-serif !important; }
.stMetric label { color: #2d5c4f !important; font-size: 0.72rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #34d399 !important; font-family: 'IBM Plex Mono', monospace !important; }
.stCaption { color: #2a4a40 !important; font-size: 0.78rem !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #e8f5f0;
    letter-spacing: -0.04em;
    padding: 1rem 0 0.2rem;
}
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(52, 211, 153, 0.4), transparent);
    margin: 1rem 0 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── CREDENTIALS ──────────────────────────────────────────────
VALID_USER_ID = "ABCD"
VALID_PASSWORD = "12345678"

for key, default in [("logged_in", False), ("login_error", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── LOGIN ─────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.05, 1])
    with col_m:
        st.markdown("""
        <div style='background:linear-gradient(160deg,#091512,#0d1e1a,#0b1a16);
                    border:1px solid rgba(52,211,153,0.18);border-radius:20px;
                    padding:2.8rem 2.6rem 2.2rem;position:relative;overflow:hidden;
                    box-shadow:0 0 80px rgba(5,150,105,0.12),0 0 0 1px rgba(52,211,153,0.15);'>
            <div style='font-family:Syne,sans-serif;font-size:2.8rem;font-weight:800;
                        color:#e8f5f0;line-height:1;letter-spacing:-0.05em;'>
                Ana<span style='background:linear-gradient(90deg,#34d399,#fbbf24);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;'>lyzer</span>
            </div>
            <div style='color:#3d7a6a;font-size:0.85rem;margin-top:0.5rem;font-weight:500;'>
                ML Analytics · Regression Intelligence Platform
            </div>
            <div style='display:flex;gap:0.5rem;margin-top:1rem;flex-wrap:wrap;'>
                <span style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);
                    color:#34d399;font-size:0.67rem;padding:0.25rem 0.75rem;border-radius:6px;
                    font-family:IBM Plex Mono,monospace;font-weight:500;'>RIDGE</span>
                <span style='background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);
                    color:#fbbf24;font-size:0.67rem;padding:0.25rem 0.75rem;border-radius:6px;
                    font-family:IBM Plex Mono,monospace;font-weight:500;'>LASSO</span>
                <span style='background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.2);
                    color:#22d3ee;font-size:0.67rem;padding:0.25rem 0.75rem;border-radius:6px;
                    font-family:IBM Plex Mono,monospace;font-weight:500;'>LINEAR</span>
            </div>
            <div style='height:1px;background:linear-gradient(90deg,rgba(52,211,153,0.35),transparent);margin-top:1.5rem;'></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        user_id  = st.text_input("User ID",  placeholder="Enter your User ID")
        password = st.text_input("Password", placeholder="Enter your Password", type="password")

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        if st.button("Sign In →", use_container_width=True):
            if user_id == VALID_USER_ID and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Invalid credentials. Please try again."
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🔐 Authorized access only · Analyzer v1.0")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        Ana<span style='background:linear-gradient(90deg,#34d399,#fbbf24);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;'>lyzer</span>
    </div>
    <div style='color:#2d5c4f;font-size:0.72rem;margin-bottom:0.5rem;font-weight:600;letter-spacing:0.02em;'>
        ML Analytics Platform
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#2d5c4f;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.7rem;font-weight:700;'>⚙️ Model Config</div>", unsafe_allow_html=True)
    model_type   = st.selectbox("Regression Model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
    test_size    = st.slider("Test Split (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random Seed", 0, 999, 42)

    alpha_val = None
    if model_type == "Ridge Regression":
        alpha_val = st.slider("Ridge Alpha (λ)", 0.01, 10.0, 1.0, 0.01)
    elif model_type == "Lasso Regression":
        alpha_val = st.slider("Lasso Alpha (λ)", 0.01, 10.0, 1.0, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='color:#2d5c4f;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.7rem;font-weight:700;'>🎨 Chart Palette</div>", unsafe_allow_html=True)
    chart_palette = st.selectbox("Color Palette", ["Emerald & Amber", "Teal Tide", "Solar Flare", "Arctic Moss"])
    palette_map = {
        "Emerald & Amber": ["#34d399", "#fbbf24", "#22d3ee", "#f87171"],
        "Teal Tide":       ["#2dd4bf", "#818cf8", "#38bdf8", "#fb923c"],
        "Solar Flare":     ["#fb923c", "#facc15", "#f472b6", "#34d399"],
        "Arctic Moss":     ["#a3e635", "#22d3ee", "#818cf8", "#fb7185"],
    }
    colors = palette_map[chart_palette]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:linear-gradient(90deg,rgba(52,211,153,0.3),transparent);margin-bottom:1rem;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    st.caption("Analyzer v1.0 · Slate & Emerald")

# ── HERO ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div>
        <div class="hero-title">Ana<span class="hero-accent">lyzer</span></div>
        <div class="hero-sub">Regression Intelligence Platform · Universal ML Analytics</div>
        <div class="hero-pill-row">
            <span class="hero-pill">🔬 {model_type}</span>
            <span class="hero-pill">📊 Multi-Variable</span>
            <span class="hero-pill">⚡ Real-Time</span>
        </div>
    </div>
    <div class="hero-badge">🔬 ML Analytics</div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────
st.markdown('<div class="section-label">📂 Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("⬆️  Upload any CSV file to begin. After uploading, you'll select your target (Y) column.")
    st.stop()

data_raw = pd.read_csv(uploaded_file)
all_cols = list(data_raw.columns)
numeric_cols_all = data_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data_raw.select_dtypes(include="object").columns.tolist()

if len(numeric_cols_all) < 2:
    st.error("Your dataset needs at least 2 numeric columns to run regression.")
    st.stop()

st.success(f"✅ Dataset loaded: **{data_raw.shape[0]:,}** rows × **{data_raw.shape[1]}** columns")

# ── TARGET SELECTOR ───────────────────────────────────────────
st.markdown('<div class="section-label">🎯 Select Target Column</div>', unsafe_allow_html=True)
st.markdown("Choose the column you want to **predict** (your dependent / output variable):")

profit_col = st.selectbox(
    "Target Column (Y)",
    options=numeric_cols_all,
    index=len(numeric_cols_all) - 1,
    help="This is the column the model will learn to predict."
)

feature_cols_numeric = [c for c in numeric_cols_all if c != profit_col]
st.caption(f"Features (X): {feature_cols_numeric + categorical_cols}  →  Target (Y): **{profit_col}**")
st.divider()

# ── PREPROCESS ────────────────────────────────────────────────
data_enc = pd.get_dummies(data_raw, drop_first=True)
if profit_col not in data_enc.columns:
    st.error(f"Target column '{profit_col}' not found after encoding. Please select a numeric column.")
    st.stop()

X_full = data_enc.drop(profit_col, axis=1)
y_full = data_enc[profit_col]

# ── DARK FIGURE HELPER ────────────────────────────────────────
def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    bg = "#0b1915"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#162e25")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#2d6050", labelsize=8)
    ax.xaxis.label.set_color("#3d7a6a")
    ax.yaxis.label.set_color("#3d7a6a")
    ax.grid(True, linestyle="--", alpha=0.1, color="#2d6050")
    return fig, ax

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Explorer", "📊 Visualisations",
    "🤖 Model & Results", "🔮 Predict", "📈 Model Comparison"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Raw Dataset</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, cls in [
        (c1, "Rows",        str(data_raw.shape[0]),    ""),
        (c2, "Columns",     str(data_raw.shape[1]),     ""),
        (c3, "Numeric",     str(len(numeric_cols_all)), "blue"),
        (c4, "Categorical", str(len(categorical_cols)), "amber"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{val}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(data_raw, use_container_width=True, height=280)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label" style="font-size:1rem;">📐 Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(data_raw.describe().round(2), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-label" style="font-size:1rem;">🧹 Missing Values</div>', unsafe_allow_html=True)
        missing = data_raw.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["Status"] = missing["Missing"].apply(lambda x: "✅ Clean" if x == 0 else "⚠️ Nulls")
        st.dataframe(missing, use_container_width=True)

    if categorical_cols:
        st.markdown('<div class="section-label" style="font-size:1rem;">🏷️ Categorical Distribution</div>', unsafe_allow_html=True)
        cat_choice = st.selectbox("Select categorical column", categorical_cols)
        cat_counts = data_raw[cat_choice].value_counts()
        fig_s, ax_s = dark_fig(6, 3)
        ax_s.barh(cat_counts.index.astype(str), cat_counts.values,
                  color=colors[0], edgecolor="none", height=0.6)
        ax_s.set_xlabel("Count")
        st.pyplot(fig_s)
        plt.close(fig_s)

# ══════════════════════════════════════════════════════════════
# TAB 2 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Feature vs Target · Scatter**")
        x_scatter = st.selectbox("Pick X feature", feature_cols_numeric, index=0, key="scatter_x")
        fig1, ax1 = dark_fig()
        sc = ax1.scatter(data_raw[x_scatter], data_raw[profit_col],
                         c=data_raw[profit_col], cmap="cool",
                         alpha=0.85, s=60, edgecolors="#0b1915", linewidths=0.4)
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.ax.yaxis.set_tick_params(color="#2d6050")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#2d6050", fontsize=7)
        ax1.set_xlabel(x_scatter)
        ax1.set_ylabel(profit_col)
        st.pyplot(fig1)
        plt.close(fig1)

    with v2:
        st.markdown("**Correlation Heatmap**")
        fig2, ax2 = dark_fig()
        sns.heatmap(data_enc.corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="YlGn", ax=ax2, linewidths=0.5,
                    annot_kws={"size": 8, "color": "#c8e8d8"},
                    linecolor="#0d201a")
        ax2.tick_params(colors="#2d6050", labelsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

    v3, v4 = st.columns(2)

    with v3:
        st.markdown("**Second Feature vs Target**")
        default_idx = min(1, len(feature_cols_numeric) - 1)
        x_scatter2 = st.selectbox("Pick X feature", feature_cols_numeric,
                                   index=default_idx, key="scatter_x2")
        fig3, ax3 = dark_fig()
        ax3.scatter(data_raw[x_scatter2], data_raw[profit_col],
                    color=colors[2], alpha=0.82, s=60,
                    edgecolors="#0b1915", linewidths=0.4)
        ax3.set_xlabel(x_scatter2)
        ax3.set_ylabel(profit_col)
        st.pyplot(fig3)
        plt.close(fig3)

    with v4:
        st.markdown(f"**{profit_col} · Distribution**")
        fig4, ax4 = dark_fig()
        n, bins, patches = ax4.hist(data_raw[profit_col], bins=16,
                                    edgecolor="#0b1915", linewidth=0.8, alpha=0.9)
        for patch in patches:
            patch.set_facecolor(colors[0])
        ax4.axvline(data_raw[profit_col].mean(), color=colors[1],
                    linestyle="--", linewidth=1.5,
                    label=f"Mean: {data_raw[profit_col].mean():,.2f}")
        ax4.set_xlabel(profit_col)
        ax4.set_ylabel("Frequency")
        ax4.legend(fontsize=8, labelcolor="#7aada0",
                   facecolor="#0b1915", edgecolor="#1a3a2a")
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown('<div class="section-label" style="font-size:1rem;">🔗 Custom Pair Plot</div>', unsafe_allow_html=True)
    px_col, py_col = st.columns(2)
    x_feat = px_col.selectbox("X Axis", numeric_cols_all, index=0, key="pair_x")
    y_feat = py_col.selectbox("Y Axis", numeric_cols_all,
                               index=len(numeric_cols_all) - 1, key="pair_y")
    fig5, ax5 = dark_fig(10, 4)
    if categorical_cols:
        cat_for_color = categorical_cols[0]
        palette = [colors[0], colors[2], colors[3], colors[1], "#a3e635"]
        for i, val in enumerate(data_raw[cat_for_color].unique()):
            mask = data_raw[cat_for_color] == val
            ax5.scatter(data_raw.loc[mask, x_feat], data_raw.loc[mask, y_feat],
                        label=str(val), alpha=0.82, s=60,
                        color=palette[i % len(palette)],
                        edgecolors="#0b1915", linewidths=0.4)
        ax5.legend(fontsize=8, labelcolor="#7aada0", facecolor="#0b1915",
                   edgecolor="#1a3a2a", title=cat_for_color, title_fontsize=7)
    else:
        ax5.scatter(data_raw[x_feat], data_raw[y_feat],
                    color=colors[0], alpha=0.8, s=60)
    ax5.set_xlabel(x_feat)
    ax5.set_ylabel(y_feat)
    st.pyplot(fig5)
    plt.close(fig5)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL & RESULTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Model Training & Evaluation</div>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size / 100, random_state=int(random_state)
    )

    if model_type == "Ridge Regression":
        model = Ridge(alpha=alpha_val)
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=alpha_val)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X_full, y_full, cv=5, scoring="r2")

    r2_cls = "green" if r2 >= 0.9 else "amber" if r2 >= 0.75 else "red"

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, cls in [
        (m1, "R² Score",   f"{r2:.4f}",       r2_cls),
        (m2, "MAE",        f"{mae:,.2f}",      "blue"),
        (m3, "RMSE",       f"{rmse:,.2f}",     "blue"),
        (m4, "CV Mean R²", f"{cv.mean():.4f}", ""),
        (m5, "CV Std",     f"{cv.std():.4f}",  "amber"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{val}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    r3a, r3b = st.columns(2)

    with r3a:
        st.markdown("**Actual vs Predicted**")
        fig6, ax6 = dark_fig()
        ax6.scatter(y_test, y_pred, color=colors[0], alpha=0.82, s=60,
                    edgecolors="#0b1915", linewidths=0.4)
        mn = min(float(y_test.min()), float(y_pred.min()))
        mx = max(float(y_test.max()), float(y_pred.max()))
        ax6.plot([mn, mx], [mn, mx], color=colors[1],
                 linestyle="--", linewidth=1.5, label="Perfect Fit")
        ax6.set_xlabel(f"Actual {profit_col}")
        ax6.set_ylabel(f"Predicted {profit_col}")
        ax6.legend(fontsize=8, labelcolor="#7aada0",
                   facecolor="#0b1915", edgecolor="#1a3a2a")
        st.pyplot(fig6)
        plt.close(fig6)

    with r3b:
        st.markdown("**Residuals Distribution**")
        residuals = y_test - y_pred
        fig7, ax7 = dark_fig()
        ax7.hist(residuals, bins=15, color=colors[1],
                 edgecolor="#0b1915", linewidth=0.8, alpha=0.85)
        ax7.axvline(0, color=colors[3], linestyle="--",
                    linewidth=1.5, label="Zero Error")
        ax7.set_xlabel("Residuals")
        ax7.set_ylabel("Frequency")
        ax7.legend(fontsize=8, labelcolor="#7aada0",
                   facecolor="#0b1915", edgecolor="#1a3a2a")
        st.pyplot(fig7)
        plt.close(fig7)

    st.markdown('<div class="section-label" style="font-size:1rem;">🧬 Feature Coefficients</div>',
                unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": X_full.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)

    fig8, ax8 = dark_fig(10, max(3, len(coef_df) * 0.4))
    bar_colors = [colors[0] if v >= 0 else colors[3] for v in coef_df["Coefficient"]]
    ax8.barh(coef_df["Feature"], coef_df["Coefficient"],
             color=bar_colors, edgecolor="none", height=0.6)
    ax8.axvline(0, color="#2d6050", linewidth=1)
    ax8.set_xlabel("Coefficient Value")
    st.pyplot(fig8)
    plt.close(fig8)

    st.markdown('<div class="section-label" style="font-size:1rem;">📋 Coefficient Table</div>',
                unsafe_allow_html=True)
    st.dataframe(coef_df.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">🔮 Make a Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter values for each feature to get a real-time prediction from the trained model.")

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_full, y_full, test_size=test_size / 100, random_state=int(random_state)
    )
    if model_type == "Ridge Regression":
        model2 = Ridge(alpha=alpha_val)
    elif model_type == "Lasso Regression":
        model2 = Lasso(alpha=alpha_val)
    else:
        model2 = LinearRegression()
    model2.fit(X_tr2, y_tr2)

    input_vals = {}
    cols_input = st.columns(3)
    for i, feat in enumerate(X_full.columns):
        def_val = float(X_full[feat].mean())
        with cols_input[i % 3]:
            input_vals[feat] = st.number_input(
                feat, value=round(def_val, 4), key=f"pred_{feat}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Generate Prediction", use_container_width=True):
        input_arr = np.array([[input_vals[f] for f in X_full.columns]])
        pred_val  = model2.predict(input_arr)[0]
        st.markdown("<br>", unsafe_allow_html=True)
        pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
        with pcol2:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#091512,#0d1e1a);
                        border:1px solid rgba(52,211,153,0.25);border-radius:18px;
                        padding:2rem;text-align:center;
                        box-shadow:0 0 40px rgba(5,150,105,0.18);'>
                <div style='font-family:Manrope,sans-serif;font-size:0.78rem;
                            text-transform:uppercase;letter-spacing:0.15em;
                            color:#2d5c4f;margin-bottom:0.8rem;font-weight:700;'>
                    Predicted {profit_col}
                </div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:3rem;
                            font-weight:700;background:linear-gradient(90deg,#34d399,#fbbf24);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;line-height:1;'>
                    {pred_val:,.2f}
                </div>
                <div style='color:#2a4a40;font-size:0.75rem;margin-top:0.8rem;
                            font-family:IBM Plex Mono,monospace;'>
                    Model: {model_type}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">📈 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Comparing Linear, Ridge, and Lasso regression across key metrics.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=test_size / 100, random_state=int(random_state)
    )

    results = {}
    for name, mdl in [
        ("Linear", LinearRegression()),
        ("Ridge",  Ridge(alpha=alpha_val or 1.0)),
        ("Lasso",  Lasso(alpha=alpha_val or 1.0)),
    ]:
        mdl.fit(X_tr, y_tr)
        yp = mdl.predict(X_te)
        cv_scores = cross_val_score(mdl, X_full, y_full, cv=5, scoring="r2")
        results[name] = {
            "R²":      r2_score(y_te, yp),
            "MAE":     mean_absolute_error(y_te, yp),
            "RMSE":    np.sqrt(mean_squared_error(y_te, yp)),
            "CV Mean": cv_scores.mean(),
            "CV Std":  cv_scores.std(),
        }

    res_df = pd.DataFrame(results).T.round(4)
    st.dataframe(res_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_a, c_b = st.columns(2)

    with c_a:
        st.markdown("**R² Score Comparison**")
        fig9, ax9 = dark_fig(5, 3)
        r2_vals = [results[n]["R²"] for n in ["Linear", "Ridge", "Lasso"]]
        bars9 = ax9.bar(["Linear", "Ridge", "Lasso"], r2_vals,
                        color=[colors[0], colors[1], colors[2]],
                        edgecolor="none", width=0.5)
        for bar, val in zip(bars9, r2_vals):
            ax9.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom',
                     color="#7aada0", fontsize=9)
        ax9.set_ylabel("R² Score")
        ax9.set_ylim(0, min(1.15, max(r2_vals) * 1.2 + 0.05))
        st.pyplot(fig9)
        plt.close(fig9)

    with c_b:
        st.markdown("**MAE Comparison**")
        fig10, ax10 = dark_fig(5, 3)
        mae_vals = [results[n]["MAE"] for n in ["Linear", "Ridge", "Lasso"]]
        bars10 = ax10.bar(["Linear", "Ridge", "Lasso"], mae_vals,
                          color=[colors[0], colors[1], colors[2]],
                          edgecolor="none", width=0.5)
        for bar, val in zip(bars10, mae_vals):
            ax10.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(mae_vals) * 0.01,
                      f'{val:,.2f}', ha='center', va='bottom',
                      color="#7aada0", fontsize=9)
        ax10.set_ylabel("MAE")
        st.pyplot(fig10)
        plt.close(fig10)

    st.markdown("**Cross-Validation R² (5-Fold) — All Models**")
    fig11, ax11 = dark_fig(10, 3.5)
    for i, (name, mdl) in enumerate([
        ("Linear", LinearRegression()),
        ("Ridge",  Ridge(alpha=alpha_val or 1.0)),
        ("Lasso",  Lasso(alpha=alpha_val or 1.0)),
    ]):
        mdl.fit(X_tr, y_tr)
        cv_s = cross_val_score(mdl, X_full, y_full, cv=5, scoring="r2")
        ax11.plot(range(1, 6), cv_s, marker="o", markersize=6,
                  color=colors[i], linewidth=2, label=name, alpha=0.9)
        ax11.fill_between(range(1, 6), cv_s, alpha=0.08, color=colors[i])
    ax11.set_xlabel("Fold")
    ax11.set_ylabel("R² Score")
    ax11.set_xticks(range(1, 6))
    ax11.legend(fontsize=9, labelcolor="#7aada0",
                facecolor="#0b1915", edgecolor="#1a3a2a")
    st.pyplot(fig11)
    plt.close(fig11)
