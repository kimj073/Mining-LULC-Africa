import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="MineWatch IVC Dashboard", layout="wide")
st.title("🕵️ MineWatch IVC Dashboard")
st.subheader("Artisanal mine segmentation from Sentinel-1/2 (Côte d'Ivoire)")

# Find the directory of this script
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    required_files = [
        "bundle.npz",
        "thresholds.json",
        "leaderboard.csv",
    ]
    missing_files = [name for name in required_files if not (BASE_DIR / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required dashboard assets: " + ", ".join(missing_files)
        )

    zb = np.load(BASE_DIR / "bundle.npz", allow_pickle=True)
    B = {k: zb[k] for k in zb.files}
    THR = json.load(open(BASE_DIR / "thresholds.json"))
    LBD = pd.read_csv(BASE_DIR / "leaderboard.csv")
    try:
        SPATIAL = pd.read_csv(BASE_DIR / "random_vs_spatial.csv")
    except Exception:
        SPATIAL = None
    return B, THR, LBD, SPATIAL

try:
    B, THR, LBD, SPATIAL = load_data()
except (FileNotFoundError, json.JSONDecodeError, pd.errors.ParserError, OSError, ValueError) as exc:
    st.error("The MineWatch dashboard could not load its data.")
    st.info(
        f"{exc}\n\nPlace the missing dashboard assets beside `app.py` in "
        f"`{BASE_DIR}` and refresh this page."
    )
    st.stop()
TRUTH = B["truth"].astype(bool)
NT = len(TRUTH)
MODELS = [k[2:] for k in B if k.startswith("p_")]

# Sidebar Controls
st.sidebar.header("Dashboard Parameters")
selected_model = st.sidebar.selectbox("Select Model", MODELS, index=MODELS.index("unet_hybrid") if "unet_hybrid" in MODELS else 0)
def_thr = float(THR.get(selected_model, 0.5))
threshold = st.sidebar.slider("Classification Threshold", 0.01, 0.99, def_thr, 0.01)

# Base map choice (switch if one provider is blocked on your network)
BASEMAPS = {
    "Light (Carto)": "carto-positron",
    "Dark (Carto)": "carto-darkmatter",
    "Satellite (Esri)": "satellite",
    "OpenStreetMap": "open-street-map",
}
basemap_choice = st.sidebar.selectbox("Base Map", list(BASEMAPS.keys()), index=0)

PIXEL_HA = 0.01
PRED = B[f"p_{selected_model}"].astype(np.float32) >= threshold

tabs = st.tabs(["Map Overview", "Model Leaderboard", "Tile Explorer", "Threshold Lab"])

# --- 1. OVERVIEW MAP ---
with tabs[0]:
    st.header("Geographical Performance Map")

    # Calculate counts
    tp = int((PRED & TRUTH).sum())
    fp = int((PRED & ~TRUTH).sum())
    fn = int((~PRED & TRUTH).sum())
    iou = tp / max(tp + fp + fn, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tiles Checked", f"{NT}")
    col2.metric("Actual Mine Area", f"{TRUTH.sum() * PIXEL_HA:.1f} ha")
    col3.metric("Predicted Mine Area", f"{PRED.sum() * PIXEL_HA:.1f} ha")
    col4.metric("Pixel-Level IoU", f"{iou:.4f}")

    df = pd.DataFrame({
        "tile_id": B["tile_id"],
        "zone": B["zone"],
        "lon": B["lon"],
        "lat": B["lat"],
        "predicted_ha": PRED.sum((1, 2)) * PIXEL_HA,
        "true_ha": TRUTH.sum((1, 2)) * PIXEL_HA
    })
    df["status"] = np.select(
        [(df.predicted_ha > 0) & (df.true_ha > 0), df.predicted_ha > 0, df.true_ha > 0],
        ["Correct Detection", "False Alarm", "Missed Mine"],
        default="Clear Background"
    )
    df["size"] = df[["predicted_ha", "true_ha"]].max(axis=1) + 0.3

    style = BASEMAPS[basemap_choice]
    is_satellite = style == "satellite"

    # Put the tile counts directly in the legend (colour key) labels
    STATUS_COLORS = {
        "Correct Detection": "#2a9d8f",
        "False Alarm": "#e63946",
        "Missed Mine": "#1d4ed8",
        "Clear Background": "#9aa5b1",
    }
    counts = df["status"].value_counts()
    df["status_label"] = df["status"].map(lambda s: f"{s} ({int(counts.get(s, 0))})")
    label_colors = {f"{k} ({int(counts.get(k, 0))})": v for k, v in STATUS_COLORS.items()}

    # px.scatter_mapbox -> px.scatter_map, mapbox_style -> map_style
    fig = px.scatter_map(
        df,
        lat="lat", lon="lon", color="status_label", size="size", size_max=18, zoom=6,
        hover_data=["tile_id", "zone", "status", "predicted_ha", "true_ha"],
        color_discrete_map=label_colors,
        category_orders={"status_label": list(label_colors.keys())},
        labels={"status_label": f"Tile counts ({selected_model}, τ={threshold:.2f})"},
        map_style="white-bg" if is_satellite else style,
        height=600
    )

    if is_satellite:
        fig.update_layout(
            map_layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            }]
        )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 2. LEADERBOARD ---
with tabs[1]:
    st.header("Model Benchmark Leaderboard")
    st.dataframe(LBD.style.format(precision=4))

    long_lbd = LBD.melt(id_vars="model", value_vars=["iou", "f1", "precision", "recall"], var_name="Metric")
    fig = px.bar(long_lbd, x="model", y="value", color="Metric", barmode="group", height=400, title="Metrics Comparison")
    st.plotly_chart(fig, use_container_width=True)

    if SPATIAL is not None:
        st.subheader("Generalisation Check: Random vs Spatial Splits")
        st.dataframe(SPATIAL.style.format(precision=4))

# --- 3. TILE EXPLORER ---
def to_uint8(a):
    """Float array -> uint8 image (0-255). Handles NaN/inf and constant arrays."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)

def upscale(a, target=384):
    """Nearest-neighbour upscale so tiny tiles (e.g. 64x64) are clearly visible."""
    h = a.shape[0]
    f = max(1, int(round(target / max(h, 1))))
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)

def show_img(a, caption=None):
    """Display an image, compatible with old and new Streamlit versions."""
    a = upscale(a)
    try:
        st.image(a, caption=caption, width="stretch")
    except TypeError:
        st.image(a, caption=caption, use_container_width=True)

def stretch(a):
    a = np.nan_to_num(a.astype(np.float32), nan=0.0)
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

with tabs[2]:
    st.header("Individual Tile Inspector")

    # Tile Selector
    order = np.argsort(-TRUTH.sum((1, 2)))
    tile_options = [(int(idx), f"{B['tile_id'][idx]} (True Mine: {TRUTH[idx].sum() * PIXEL_HA:.1f} ha)") for idx in order]
    selected_idx = st.selectbox("Select Tile", options=[t[0] for t in tile_options], format_func=lambda x: dict(tile_options)[x])

    bands_choice = st.selectbox("Spectral Band Combination", [
        "True colour (B4,B3,B2)",
        "SWIR false colour (B11,B8,B4)",
        "Vegetation (B8,B4,B3)"
    ])

    # Mapping combinations
    band_map = {
        "True colour (B4,B3,B2)": [3, 2, 1],
        "SWIR false colour (B11,B8,B4)": [9, 7, 3],
        "Vegetation (B8,B4,B3)": [7, 3, 2]
    }

    img = B["img"][selected_idx].astype(np.float32)
    # If the tile is stored channels-last (H, W, C), convert to (C, H, W)
    if img.ndim == 3 and img.shape[-1] <= 20 and img.shape[0] > img.shape[-1]:
        img = np.transpose(img, (2, 0, 1))

    ch = band_map[bands_choice]
    if max(ch) >= img.shape[0]:
        st.warning(
            f"This tile has only {img.shape[0]} bands, but the selected combination "
            f"needs band index {max(ch)}. Showing the first 3 bands instead."
        )
        ch = [0, 1, 2]

    rgb_img = to_uint8(np.dstack([stretch(img[c]) for c in ch]))
    mask_img = to_uint8(TRUTH[selected_idx].astype(np.float32))
    prob_raw = B[f"p_{selected_model}"][selected_idx].astype(np.float32)
    prob_img = to_uint8(prob_raw)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Spectral Image View**")
        show_img(rgb_img)
    with col2:
        st.write("**Ground Truth Mine Mask**")
        show_img(mask_img)
    with col3:
        st.write("**Model Probability Map**")
        show_img(prob_img)

    with st.expander("Diagnostics (open if an image looks blank)"):
        st.write({
            "tile image shape (C,H,W)": tuple(img.shape),
            "image min / max": (float(np.nanmin(img)), float(np.nanmax(img))),
            "NaN pixels in image": int(np.isnan(img).sum()),
            "truth mask pixels (mine)": int(TRUTH[selected_idx].sum()),
            "probability min / max": (float(prob_raw.min()), float(prob_raw.max())),
            "bands used": ch,
        })

# --- 4. THRESHOLD LAB ---
with tabs[3]:
    st.header("Threshold Sensitivity Laboratory")
    st.write("Evaluate how metrics change when moving the classification threshold:")

    p_flat = B[f"p_{selected_model}"].ravel().astype(np.float32)
    t_flat = TRUTH.ravel()

    grid = np.linspace(0.02, 0.98, 49)
    lab_rows = []
    for th in grid:
        pred_t = p_flat >= th
        tp_t = int((pred_t & t_flat).sum())
        fp_t = int((pred_t & ~t_flat).sum())
        fn_t = int((~pred_t & t_flat).sum())

        lab_rows.append({
            "Threshold": th,
            "IoU": tp_t / max(tp_t + fp_t + fn_t, 1),
            "F1": 2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1),
            "Precision": tp_t / max(tp_t + fp_t, 1),
            "Recall": tp_t / max(tp_t + fn_t, 1)
        })

    df_lab = pd.DataFrame(lab_rows)
    fig = px.line(df_lab, x="Threshold", y=["IoU", "F1", "Precision", "Recall"], title=f"Threshold Sweep for {selected_model}")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected: {threshold:.2f}")
    st.plotly_chart(fig, use_container_width=True)