"""
obis_ai_dashboard.py
Single-input version: one text box for both AI and dataset routing.
Enhanced: persistent cached view, date filters, interactive plots,
CSV/Excel export, PDF report export (images via kaleido + reportlab).
Run:
    pip install streamlit requests pandas plotly kaleido reportlab openpyxl
    streamlit run obis_ai_dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.io as pio
import time
from datetime import datetime, date
import json
from io import StringIO, BytesIO
import base64
import math

import sys
print(sys.executable)

import matplotlib.pyplot as plt
import tempfile


# -----------------------------
# Session helpers: persist dataframe as plain Python (records + columns)
# -----------------------------
# -----------------------------
# Session helpers: persist dataframe as plain Python (records + columns)
# -----------------------------
import numpy as np  # ensure numpy available for type checks

def _sanitize_value(v):
    """Convert pandas / numpy / datetime types to plain Python (JSON-friendly) values."""
    try:
        if v is None:
            return None
        # pandas NA
        if pd.isna(v):
            return None
        # pandas Timestamp or datetime -> ISO string
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        if isinstance(v, date):
            return v.isoformat()
        # numpy scalars -> native python
        if isinstance(v, (np.integer, np.int_, np.int32, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float_, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.bool_, np.bool8)):
            return bool(v)
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                return str(v)
        return v
    except Exception:
        return str(v)

def save_obis_df(df: pd.DataFrame):
    """Save dataframe into session_state as plain Python structures (safe across reruns)."""
    records = []
    for r in df.to_dict(orient="records"):
        rec = {k: _sanitize_value(v) for k, v in r.items()}
        records.append(rec)
    st.session_state["obis_df_records"] = records
    st.session_state["obis_df_columns"] = list(df.columns)

def load_obis_df() -> pd.DataFrame | None:
    """Reconstruct DataFrame from session_state. Return None if not present."""
    recs = st.session_state.get("obis_df_records")
    cols = st.session_state.get("obis_df_columns")
    # be explicit: only None means missing (empty list is valid)
    if recs is None or cols is None:
        return None
    try:
        df = pd.DataFrame(recs, columns=cols)
        # attempt to parse common datetime back into datetime dtype
        if "eventDate" in df.columns:
            df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
        return df
    except Exception:
        try:
            return pd.DataFrame(recs)
        except Exception:
            return None



# --- Compatibility helpers (typing, streamlit cache compatibility) ---
from typing import Optional

# Streamlit changed caching API in newer versions. Provide a safe fallback so this file
# runs on older Streamlit installs too.
try:
    cache_data = st.cache_data  # Streamlit 1.18+
except Exception:
    cache_data = getattr(st, "cache", None) or (lambda **kw: (lambda f: f))



def plot_year_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    if "eventDate_parsed" in df.columns:
        df['year'] = df['eventDate_parsed'].dt.year
        df['year'].dropna().astype(int).value_counts().sort_index().plot(ax=ax)
        ax.set_title("Records by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
    return fig

def plot_depth_distribution_plotly(df):
    """
    Depth distribution using plotly.express.histogram.
    Returns a plotly.graph_objects.Figure (or None if no data).
    """
    if "depth" not in df.columns or not df["depth"].notna().any():
        return None
    # coerce to numeric and drop NaNs
    series = pd.to_numeric(df["depth"].dropna(), errors="coerce").dropna()
    if series.empty:
        return None

    fig = px.histogram(series, x=series, nbins=30,
                       title="Depth distribution",
                       labels={"x": "Depth (m)", "count": "Frequency"},
                       height=350)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig



def plot_occurrence_map(df):
    fig, ax = plt.subplots(figsize=(6,4))
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        ax.scatter(df["decimalLongitude"], df["decimalLatitude"], c="red", s=10, alpha=0.6)
        ax.set_title("Occurrence Locations")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    return fig

def safe_rerun():
    """
    Robust replacement for st.experimental_rerun().
    Tries public API first, then the internal RerunException, otherwise sets
    a session flag to force a re-render and asks user to refresh.
    (No use of deprecated st.experimental_set_query_params.)
    """
    try:
        st.experimental_rerun()
        return
    except Exception:
        pass

    try:
        # Streamlit internal rerun exception (works in many versions)
        from streamlit.runtime.scriptrunner import RerunException
        raise RerunException()
    except Exception:
        # Last-ditch: flip a session_state key (forces app to observe a change)
        try:
            st.session_state["_force_rerun_ts"] = time.time()
        except Exception:
            pass
        # Inform the user to refresh if everything else fails
        st.warning("Action completed. If the UI didn't update, please refresh your browser.")




# add near the top, right after imports
import uuid

def _make_dl_key(base: str, filename: str) -> str:
    """
    Create a stable unique widget key for download buttons.
    Keeps a registry in st.session_state to avoid duplicates.
    This version is idempotent across reruns: it returns the same key
    for the same base+filename unless it truly conflicts with an existing key.
    """
    safe = (base + "_" + filename).replace(" ", "_").replace("/", "_")
    safe = safe[:200]

    # ensure _dl_keys exists and is a list
    existing = st.session_state.get("_dl_keys")
    if existing is None:
        st.session_state["_dl_keys"] = []
        existing = st.session_state["_dl_keys"]
    if not isinstance(existing, list):
        existing = list(existing)
        st.session_state["_dl_keys"] = existing

    # If the exact safe key already exists, return it (idempotent).
    if safe in existing:
        return safe

    # If a shorter 'safe' collides with other values, generate a single time suffix
    if any(s.startswith(safe + "_") for s in existing):
        safe = f"{safe}_{uuid.uuid4().hex[:8]}"

    existing.append(safe)
    st.session_state["_dl_keys"] = existing
    return safe




def dl_button(container, label, data, file_name, mime, base="dl"):
    """
    Wrapper for container.download_button that guarantees a unique key.
    container can be st (top-level) or a column object (left_col, right_col).
    """
    key = _make_dl_key(base, file_name)
    return container.download_button(label, data=data, file_name=file_name, mime=mime, key=key)




# -----------------------------
# CONFIG (your provided key)
# -----------------------------
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.3-8b-instruct:free"

OBIS_API_URL = "https://api.obis.org/v3/occurrence"

# -----------------------------
# Page / CSS (dark theme restored)
# -----------------------------
st.set_page_config(
    page_title="FloatChat | AI-Powered ARGO Ocean Data Discovery & Visualization",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show title and description inside the app
st.title("FloatChat: AI-Powered ARGO Ocean Data Discovery & Visualization")

st.markdown(
    """
    **FloatChat** is an AI-powered interface that makes ARGO ocean data simple to explore.  
    Ask questions in plain language and get instant insights on temperature, salinity, 
    and biogeochemical variables — no complex tools required.
    """
)


st.markdown(
    """
    <style>
    /* page bg and card (dark theme restored) */
    .reportview-container, .stApp {
        background: linear-gradient(180deg,#071022 0%, #081526 45%, #0b2b3a 100%);
        color: #e6f0f6;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }
    /* card look for st.card containers */
    .card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    h1, h2, h3, .css-1v3fvcr {
        color: #e6f0f6;
    }
    /* small subtle button hover */
    button.stButton>button {
        background: linear-gradient(90deg,#0b84ff 0%, #6ee7b7 100%);
        color: #06202a;
        font-weight: 600;
        border-radius: 10px;
    }
    /* loading pulse */
    .pulse {
        display:inline-block;
        width:12px;
        height:12px;
        background:#0b84ff;
        border-radius:50%;
        animation:pulse 1.4s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.9; }
        50% { transform: scale(1.4); opacity: 0.4; }
        100% { transform: scale(0.8); opacity: 0.9; }
    }
    .muted { color: #9fb7c7; font-size: 0.95rem; }
    .small { font-size: 0.85rem; color:#bcd7e6; }
    .stat {
        display:inline-block;
        padding:10px 14px;
        margin-right:8px;
        background: rgba(255,255,255,0.02);
        border-radius:10px;
        border:1px solid rgba(255,255,255,0.02);
    }
    .top-row { display:flex; gap:12px; align-items:center; }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Sidebar controls (non-text only)
# -----------------------------
with st.sidebar:
    st.markdown("## Controls")
    max_records = st.slider("Max records to fetch", min_value=10, max_value=1000, value=200, step=10)
    bbox_enable = st.checkbox("Filter by bounding box (lon/lat)", value=False)
    if bbox_enable:
        lon_min = st.number_input("Lon min", value=68.0, step=0.1, format="%.3f")
        lon_max = st.number_input("Lon max", value=96.0, step=0.1, format="%.3f")
        lat_min = st.number_input("Lat min", value=6.0, step=0.1, format="%.3f")
        lat_max = st.number_input("Lat max", value=24.0, step=0.1, format="%.3f")

    st.markdown("---")
    st.markdown("## Date range (optional)")
    start_date = st.date_input("Start date", value=date(2000, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    if start_date and end_date and start_date > end_date:
        st.warning("Start date is after end date — results will be empty until corrected.")

    st.markdown("---")
    st.markdown("### OpenRouter / LLM")
    st.markdown(f"**Model:** {OPENROUTER_MODEL}")
    st.markdown("<div class='small muted'>Using your OpenRouter key (provided).</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Advanced")
    auto_summary = st.checkbox("Auto-summarize after successful fetch", value=True)
    st.markdown("Built for FloatChat — IndOBIS + AI")

# -----------------------------
# Helper functions
# -----------------------------
@cache_data(ttl=60 * 30)
def fetch_obis_records(species_name: str, size: int = 100, bbox: Optional[dict] = None):

    """
    Fetch OBIS records. Convert bbox into a stable tuple (hashable) so Streamlit's cache can
    safely include the bbox in the cache key.
    """
    # Turn bbox dict into a hashable tuple (lonmin, lonmax, latmin, latmax) if provided
    bbox_tuple = None
    if isinstance(bbox, dict):
        try:
            bbox_tuple = (
                float(bbox.get("lonmin")),
                float(bbox.get("lonmax")),
                float(bbox.get("latmin")),
                float(bbox.get("latmax")),
            )
        except Exception:
            bbox_tuple = None
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_tuple = tuple(bbox)

    params = {"scientificname": species_name, "size": size}
    if bbox_tuple:
        lonmin, lonmax, latmin, latmax = bbox_tuple
        poly = f"POLYGON(({lonmin} {latmin}, {lonmax} {latmin}, {lonmax} {latmax}, {lonmin} {latmax}, {lonmin} {latmin}))"
        params["geometry"] = poly

    r = requests.get(OBIS_API_URL, params=params, timeout=40)
    r.raise_for_status()
    js = r.json()
    results = js.get("results", [])
    return pd.DataFrame(results)



def ask_openrouter(messages: list, model=OPENROUTER_MODEL, timeout=60):
    """
    Safe wrapper for OpenRouter calls.
    Returns: string response (or an error message string) — never raises.
    """
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured. Set OPENROUTER_API_KEY in environment or st.secrets."

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 800, "temperature": 0.2}
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        # Try common response shapes safely
        if isinstance(j, dict):
            # new-style: choices -> [ { "message": {"content": "..."} } ]
            choices = j.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    # OpenRouter (or chat-like) shape
                    msg = first.get("message") or first.get("delta") or first
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("text")
                        if isinstance(content, str):
                            return content
                    # fallback if first has 'text'
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"]
            # older shape: 'text' at top-level
            if "text" in j and isinstance(j["text"], str):
                return j["text"]
        # fallback: return pretty json so user sees API response
        return json.dumps(j, indent=2)
    except requests.exceptions.HTTPError as he:
        return f"AI HTTP error: {he}"
    except requests.exceptions.RequestException as re:
        return f"AI request error: {re}"
    except Exception as e:
        return f"AI unknown error: {e}"



def interpret_input_via_ai(user_text: str, model=OPENROUTER_MODEL, timeout=20):
    system = {
        "role": "system",
        "content": (
            "You are a routing assistant. Given a single-line user input, return ONLY a one-line JSON object with exactly one of the two shapes:\n"
            '{"action":"search","species":"Genus species"} OR {"action":"ai","query":"..."}\n'
            "If you can identify a scientific name (Genus species), put it in 'species'. If unsure, return action 'ai'. Output pure JSON only."
        )
    }
    user = {"role": "user", "content": f"User input: \"{user_text}\""}
    try:
        reply = ask_openrouter([system, user], model=model, timeout=timeout)
        txt = reply.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
        start = txt.find("{"); end = txt.rfind("}")
        if start != -1 and end != -1:
            txt_json = txt[start:end+1]
            parsed = json.loads(txt_json)
            if "action" in parsed and parsed["action"] in ("search","ai"):
                if parsed.get("action") == "search":
                    sp = parsed.get("species", "")
                    if isinstance(sp, str):
                        parsed["species"] = sp.strip()
                return parsed
    except Exception:
        pass

    # fallback: binomial detection
    words = user_text.strip().split()
    if len(words) >= 2 and words[0][0].isupper():
        return {"action":"search", "species":" ".join(words[:2])}
    return {"action":"ai", "query": user_text}


def ai_summarize_records(df: pd.DataFrame, species_name: str):
    sample = df.head(30).to_dict(orient="records")
    system_msg = {
        "role": "system",
        "content": "You are a marine biology data assistant. Interpret species occurrence records and provide concise, non-technical summaries."
    }
    user_msg = {
        "role": "user",
        "content": f"I have {len(df)} occurrence records for '{species_name}'. Sample (up to 30 rows): {sample}\n\nPlease produce 3-6 bullet points summarizing distribution, notable patterns, and one recommended next-step analysis."
    }
    try:
        reply = ask_openrouter([system_msg, user_msg])
        return reply
    except Exception as e:
        return f"AI summarization failed: {e}"


def prepare_csv_download(df: pd.DataFrame):
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def prepare_excel_download(df: pd.DataFrame):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="records")
    return buf.getvalue()

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def prepare_pdf_download(df: pd.DataFrame, title="OBIS Report"):
    """Return PDF bytes for download, including a small sample table."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=(595, 842))  # A4 portrait approx
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"Total records: {len(df)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Build a simple table for the first N rows
    preview = df.head(20)
    if preview.empty:
        story.append(Paragraph("No records to show.", styles["Normal"]))
    else:
        # choose up to 6 columns to keep PDF readable
        cols = [c for c in ["scientificName", "eventDate", "decimalLongitude", "decimalLatitude", "depth", "basisOfRecord"] if c in preview.columns]
        if not cols:
            cols = list(preview.columns[:6])
        data = [cols]
        for _, row in preview.iterrows():
            data.append([("" if pd.isna(row.get(c, "")) else str(row.get(c, ""))[:90]) for c in cols])

        # small table style
        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b84ff")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d1d5db")),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(tbl)

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes



def make_plots_from_df(df: pd.DataFrame, species_name: str):
    """Return a dict of plotly figures (map, timeseries, depth hist, density)."""
    figs = {}
    # Map scatter
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        map_df = df.dropna(subset=["decimalLongitude","decimalLatitude"])
        sample_map = map_df if len(map_df) <= 1000 else map_df.sample(1000, random_state=1)
        figs["map"] = px.scatter_geo(
            sample_map,
            lon="decimalLongitude",
            lat="decimalLatitude",
            hover_name="scientificName" if "scientificName" in sample_map.columns else None,
            hover_data=[c for c in ["eventDate","depth"] if c in sample_map.columns],
            title=f"{species_name} occurrences (sample)",
            projection="natural earth",
            height=550,
        )
        figs["map"].update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Time series (counts per year / month)
    if "eventDate" in df.columns:
        try:
            df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
            times = df.dropna(subset=["eventDate_parsed"]).copy()
            if not times.empty:
                times["year"] = times["eventDate_parsed"].dt.year
                times["month"] = times["eventDate_parsed"].dt.to_period("M").astype(str)
                yearly = times.groupby("year").size().reset_index(name="count")
                figs["yearly"] = px.bar(yearly, x="year", y="count", title="Records per year", height=300)
                monthly = times.groupby("month").size().reset_index(name="count").sort_values("month")
                figs["monthly"] = px.line(monthly, x="month", y="count", title="Records per month (period)", height=300)
        except Exception:
            pass

    # Depth distribution
    if "depth" in df.columns:
        try:
            df_depth = df.dropna(subset=["depth"])
            if not df_depth.empty:
                # ensure numeric
                df_depth = df_depth[df_depth["depth"].apply(lambda x: (isinstance(x, (int,float)) or (isinstance(x,str) and x.replace('.','',1).isdigit())))]

                df_depth["depth_num"] = pd.to_numeric(df_depth["depth"], errors="coerce")
                df_depth = df_depth.dropna(subset=["depth_num"])
                if not df_depth.empty:
                    figs["depth_hist"] = px.histogram(df_depth, x="depth_num", nbins=30, title="Depth distribution", height=300)
        except Exception:
            pass

    # Density heatmap (lon/lat)
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        try:
            heat_df = df.dropna(subset=["decimalLongitude","decimalLatitude"])
            if len(heat_df) >= 20:
                figs["density"] = px.density_heatmap(heat_df, x="decimalLongitude", y="decimalLatitude",
                                                     nbinsx=60, nbinsy=40, title="Density heatmap (lon/lat)", height=400)
                figs["density"].update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        except Exception:
            pass

    return figs


def generate_pdf_report(df: pd.DataFrame, species_name: str, summary_text: str, figs: dict):
    """
    Professional PDF report generator:
    - Uses reportlab + kaleido (plotly -> png)
    - Includes title, metadata, AI summary, images (map/plots), and a clean table
    Returns: bytes of the generated PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, 
        Table, TableStyle, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    # Helper: convert plotly figs -> png bytes
        # Helper: convert plotly figs -> png bytes (try kaleido explicitly, fallback gracefully)
    image_items = []
    for key in ["map", "yearly", "monthly", "depth_hist", "density"]:
        if key in figs:
            try:
                # prefer kaleido explicitly (more reliable if installed)
                img_bytes = pio.to_image(figs[key], format="png", scale=2, engine="kaleido")
                image_items.append((key, img_bytes))
            except Exception as e_k:
                # try without specifying engine (let plotly choose), but capture error
                try:
                    img_bytes = pio.to_image(figs[key], format="png", scale=2)
                    image_items.append((key, img_bytes))
                except Exception as e:
                    # final fallback: no image for this figure, but surface a warning in server logs
                    print(f"[WARN] Could not render {key} with kaleido: {e_k}; fallback also failed: {e}")
                    image_items.append((key, None))


    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#6b7280")))
    styles.add(ParagraphStyle(name="Heading", parent=styles["Heading2"], fontSize=12, leading=14))
    styles.add(ParagraphStyle(name="NormalSmall", parent=styles["Normal"], fontSize=10, leading=12))

    story = []

    # Title & metadata
    report_title = f"OBIS Report — {species_name}" if species_name else "OBIS Report"
    story.append(Paragraph(report_title, styles["ReportTitle"]))
    story.append(Spacer(1, 6))
    meta_lines = [
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total records: {len(df)}"
    ]
    for m in meta_lines:
        story.append(Paragraph(m, styles["Meta"]))
    story.append(Spacer(1, 12))

    # AI Summary
    story.append(Paragraph("AI Summary", styles["Heading"]))
    if summary_text:
        for paragraph in str(summary_text).split("\n\n"):
            story.append(Paragraph(paragraph.replace("\n", "<br/>"), styles["NormalSmall"]))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No AI summary available.", styles["NormalSmall"]))
    story.append(Spacer(1, 12))

    # Images
    max_img_width = doc.width
    for title, img_bytes in image_items:
        if img_bytes:
            img_io = BytesIO(img_bytes)
            img = RLImage(img_io)
            img.drawWidth = max_img_width
            img.drawHeight = max_img_width * (img.imageHeight / float(img.imageWidth))
            story.append(Paragraph(title.replace("_", " ").title(), styles["Heading"]))
            story.append(Spacer(1, 6))
            story.append(img)
            story.append(Spacer(1, 12))

    story.append(PageBreak())

    # Table (first 100 rows, important cols)
    story.append(Paragraph("Sample Records (first 100 rows)", styles["Heading"]))
    story.append(Spacer(1, 6))

    preferred_cols = ["scientificName", "eventDate", "decimalLongitude", "decimalLatitude", "depth", "basisOfRecord", "institutionCode"]
    cols = [c for c in preferred_cols if c in df.columns] or list(df.columns[:6])
    display_rows = df.head(100)

    def fmt(val):
        if pd.isna(val): return ""
        s = str(val)
        return s[:77] + "..." if len(s) > 80 else s

    data_table = [cols] + [[fmt(r.get(c, "")) for c in cols] for _, r in display_rows.iterrows()]
    colWidths = [doc.width / len(cols)] * len(cols)

    table = Table(data_table, colWidths=colWidths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b84ff")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d1d5db")),
    ])
    for i in range(1, len(data_table)):
        if i % 2 == 0:
            style.add("BACKGROUND", (0,i), (-1,i), colors.HexColor("#f8fafc"))
    table.setStyle(style)

    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Generated by FloatChat — IndOBIS + AI", styles["Meta"]))

    def _add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 12, f"Page {page_num}")

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# MAIN UI: single input box only
# -----------------------------
st.markdown(
            "<div class='muted'>Type anything: a species name (e.g., 'Sardinella longiceps') or a question. </div></div>", unsafe_allow_html=True)

# Single-line input inside a form so pressing Enter submits immediately
with st.form(key="single_input_form", clear_on_submit=False):
    user_input = st.text_input("Enter species name or a question ", value="", key="single_input")
    submit = st.form_submit_button("Submit")


# Some Streamlit versions do not accept the `gap` parameter.
try:
    left_col, right_col = st.columns([2.4, 1.0], gap="large")
except TypeError:
    left_col, right_col = st.columns([2.4, 1.0])


# Always render cached OBIS data (so it won't disappear after other interactions)
with left_col:
    # use load_obis_df() to robustly reconstruct the DataFrame from session_state
    df_prev = load_obis_df()
    if df_prev is not None and not df_prev.empty:

        st.markdown("### Previously fetched OBIS records (cached)")
        st.markdown(f"<div class='small muted'>Showing {len(df_prev)} cached records — use 'Clear cached data' to remove.</div>", unsafe_allow_html=True)

        try:
            c1, c2, c3 = left_col.columns([1,1,2])

            with c1:
                st.markdown(f"<div class='stat'><strong>{len(df_prev)}</strong><div class='small muted'>records</div></div>", unsafe_allow_html=True)
            with c2:
                unique_locs = df_prev.dropna(subset=['decimalLongitude','decimalLatitude']).shape[0]
                st.markdown(f"<div class='stat'><strong>{unique_locs}</strong><div class='small muted'>geo points</div></div>", unsafe_allow_html=True)
            with c3:
                range_time = "-"
                if "eventDate" in df_prev.columns:
                    try:
                        if df_prev["eventDate"].notna().any():
                            mn = df_prev["eventDate"].min(); mx = df_prev["eventDate"].max()
                            if hasattr(mn, "date"):
                                range_time = f"{mn.date()} → {mx.date()}"
                            else:
                                range_time = f"{mn} → {mx}"
                    except Exception:
                        pass
                st.markdown(f"<div class='small muted'>Date range: {range_time}</div>", unsafe_allow_html=True)

            if "decimalLongitude" in df_prev.columns and "decimalLatitude" in df_prev.columns:
                map_df = df_prev.dropna(subset=["decimalLongitude","decimalLatitude"])
                if len(map_df) > 500:
                    map_df = map_df.sample(500, random_state=1)
                fig = px.scatter_geo(map_df, lon="decimalLongitude", lat="decimalLatitude",
                                     hover_name="scientificName", hover_data=["eventDate","depth"] if "eventDate" in df_prev.columns else None,
                                     projection="natural earth", height=420)
                fig.update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Sample cached records")
            st.dataframe(df_prev.head(200))

            # show last AI summary if available (persisted)
            if st.session_state.get("last_summary"):
                st.markdown("#### Last AI summary (cached)")
                st.write(st.session_state.get("last_summary"))

            # download CSV & Excel

            try:
                df_for_dl = load_obis_df()
                if df_for_dl is not None:
                    csv_buf = StringIO()
                    df_for_dl.to_csv(csv_buf, index=False)
                    dl_button(left_col, "Download cached records (CSV)", data=csv_buf.getvalue(), file_name="obis_records.csv", mime="text/csv", base="cached_obis")
                    try:
                        buf = BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                            df_for_dl.to_excel(writer, index=False, sheet_name="records")
                        dl_button(left_col, "Download cached records (Excel)", data=buf.getvalue(), file_name="obis_records.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base="cached_obis_xlsx")
                    except Exception:
                        # Excel optional; CSV always available
                        pass
                else:
                    st.info("No cached dataset available for download.")
                
                


            except Exception:
                # If openpyxl (or excel writer) not installed, still allow CSV
                csv_str = prepare_csv_download(df_prev)
                dl_button(left_col, "Download cached records (CSV)", data=csv_str,
                          file_name="obis_records.csv", mime="text/csv", base="cached_obis_fallback")

                st.info("Install openpyxl to enable Excel download: pip install openpyxl")


                        # clear cache (button inside left column)
            if left_col.button("Clear cached data"):
                # remove serialized dataset and other cached UI artifacts
                st.session_state.pop("obis_df_records", None)
                st.session_state.pop("obis_df_columns", None)
                st.session_state.pop("last_species", None)
                st.session_state.pop("last_summary", None)
                # Correct keys used elsewhere in the app:
                st.session_state.pop("last_pdf", None)
                st.session_state.pop("last_pdf_name", None)
                st.session_state.pop("last_pdf_species", None)
                st.session_state.pop("data_ai_history", None)
                safe_rerun()



        except Exception as e:
            st.warning("Could not render cached data preview: " + str(e))

# Persistent dataset-AI UI (renders whenever cached dataset exists)
df_prev = load_obis_df()
if df_prev is not None and not df_prev.empty:
    # Render persistent AI controls in right_col (so they remain after downloads/reruns)
    with right_col:
        st.markdown("###  Ask AI about cached dataset")
        # text area bound to session key so value survives reruns
        ai_data_query = st.text_area(
            "Ask a question about the cached dataset (AI will see a small sample and previous summary):",
            key="data_ai_input",
            height=120
        )

        if st.button("Ask AI about cached data", key="ask_cached_data_ai"):
            # stable read of query
            ai_data_query = st.session_state.get("data_ai_input", "").strip()
            st.session_state["last_data_ai_query"] = ai_data_query

            # prepare compact sample
            sample_for_ai = None
            try:
                df_for_sample = load_obis_df()
                if df_for_sample is not None and not df_for_sample.empty:
                    sample_for_ai = df_for_sample.head(30).to_dict(orient="records")
            except Exception:
                sample_for_ai = None

            # build messages
            system_msg = {
                "role": "system",
                "content": "You are a helpful marine biology data assistant. Analyze the sample and answer the user's question concisely, suggest one next-step analysis."
            }
            parts = []
            if sample_for_ai:
                parts.append(f"Sample records (up to 30 rows): {sample_for_ai}")
            if st.session_state.get("last_summary"):
                parts.append(f"Existing AI summary: {st.session_state.get('last_summary')}")
            parts.append(f"User question: {ai_data_query or '[NO QUESTION]'}")
            user_msg = {"role": "user", "content": "\n\n".join(parts)}

            # show loading pulse
            loading_slot = right_col.empty()
            loading_slot.markdown("<div class='muted'><span class='pulse'></span> AI analyzing dataset...</div>", unsafe_allow_html=True)
            try:
                ai_reply = ask_openrouter([system_msg, user_msg])
                entry = {"question": ai_data_query, "answer": ai_reply, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
                st.session_state.setdefault("data_ai_history", []).append(entry)
                st.session_state["last_data_ai_reply"] = ai_reply
                # optionally keep this as last_summary as well
                st.session_state["last_summary"] = st.session_state.get("last_summary") or ai_reply
                right_col.markdown("**AI answer:**")
                right_col.markdown(ai_reply)
            except Exception as e:
                right_col.error(f"AI request failed: {e}")
            finally:
                loading_slot.empty()

        # show recent dataset-AI Q&A (most recent first)
        if st.session_state.get("data_ai_history"):
            st.markdown("#### Recent dataset queries")
            for item in reversed(st.session_state["data_ai_history"][-6:]):
                with st.expander(f"Q: {item['question'][:60] or '(no question)'} — {item['time']}", expanded=False):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer']}")

# Process the single submit




if submit and user_input.strip():
    with st.spinner("Routing your input via the LLM..."):
        decision = interpret_input_via_ai(user_input)

    if decision.get("action") == "search" and decision.get("species"):
        chosen_species = decision["species"]
        left_col.markdown(f"###  Searching OBIS for species: **{chosen_species}**")
        loader = left_col.empty()
        loader.markdown("<div class='muted'><span class='pulse'></span> Fetching records…</div>", unsafe_allow_html=True)
        try:
            bbox = None
            if bbox_enable:
                bbox = {"lonmin": lon_min, "lonmax": lon_max, "latmin": lat_min, "latmax": lat_max}

            df = fetch_obis_records(chosen_species, size=max_records, bbox=bbox)

            # parse and filter by date range (if eventDate is present)
            if not df.empty and "eventDate" in df.columns:
                try:
                    df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
                    if start_date:
                        df = df[df["eventDate"] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df["eventDate"] <= pd.to_datetime(end_date)]
                except Exception:
                    pass

            loader.empty()
            if df.empty:
                left_col.warning(f"No records found for species: {chosen_species} (after date/filters)")
            else:
                keep_cols = [c for c in ["scientificName","eventDate","decimalLongitude","decimalLatitude","depth","basisOfRecord","institutionCode"] if c in df.columns]
                df_clean = df[keep_cols].copy()
                if "eventDate" in df_clean.columns:
                    try:
                        df_clean["eventDate"] = pd.to_datetime(df_clean["eventDate"], errors="coerce")
                    except Exception:
                        pass

                # store in session for AI use and persistent display (store as plain records+cols)
                save_obis_df(df_clean)
                st.session_state["last_species"] = chosen_species  # save species for cached UI / downloads


                left_col.success(f"Found {len(df_clean)} records for '{chosen_species}'")
                # summary stats
                c1, c2, c3 = left_col.columns([1,1,2])
                with c1:
                    left_col.markdown(f"<div class='stat'><strong>{len(df_clean)}</strong><div class='small muted'>records</div></div>", unsafe_allow_html=True)
                with c2:
                    unique_locs = df_clean.dropna(subset=["decimalLongitude","decimalLatitude"]).shape[0]
                    left_col.markdown(f"<div class='stat'><strong>{unique_locs}</strong><div class='small muted'>geo points</div></div>", unsafe_allow_html=True)
                with c3:
                    range_time = "-"
                    if "eventDate" in df_clean.columns and df_clean["eventDate"].notna().any():
                        mn = df_clean["eventDate"].min(); mx = df_clean["eventDate"].max()
                        range_time = f"{mn.date()} → {mx.date()}"
                    left_col.markdown(f"<div class='small muted'>Date range: {range_time}</div>", unsafe_allow_html=True)

                # map + plots
                figs = make_plots_from_df(df_clean, chosen_species)
                if "map" in figs:
                    left_col.plotly_chart(figs["map"], use_container_width=True)
                # show other figs in two rows
                if "yearly" in figs or "monthly" in figs:
                    row1 = left_col.columns(2)
                    if "yearly" in figs:
                        row1[0].plotly_chart(figs["yearly"], use_container_width=True)
                    if "monthly" in figs:
                        row1[1].plotly_chart(figs["monthly"], use_container_width=True)
                if "depth_hist" in figs or "density" in figs:
                    row2 = left_col.columns(2)
                    if "depth_hist" in figs:
                        row2[0].plotly_chart(figs["depth_hist"], use_container_width=True)
                    if "density" in figs:
                        row2[1].plotly_chart(figs["density"], use_container_width=True)

                left_col.markdown("#### Sample records")
                left_col.dataframe(df_clean.head(200))

                # download CSV & Excel (for fetched results)
                try:
                    csv_str = prepare_csv_download(df_clean)
                    xlsx_bytes = prepare_excel_download(df_clean)
                    species_key = chosen_species.replace(" ", "_").replace("/", "_")
                    dl_button(left_col, "Download fetched records (CSV)",
                              data=csv_str, file_name=f"{species_key}_obis.csv", mime="text/csv", base=f"fetched_{species_key}")
                    dl_button(left_col, "Download fetched records (Excel)",
                              data=xlsx_bytes, file_name=f"{species_key}_obis.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base=f"fetched_{species_key}")
                except Exception:
                    # fallback: CSV only
                    csv_str = prepare_csv_download(df_clean)
                    species_key = chosen_species.replace(" ", "_").replace("/", "_")
                    dl_button(left_col, "Download fetched records (CSV)",
                              data=csv_str, file_name=f"{species_key}_obis.csv", mime="text/csv", base=f"fetched_{species_key}_fallback")
                    


                # Auto-summary if enabled — keep summary in session so it persists across reruns
                if auto_summary:
                    right_col.markdown("###  AI Summary (auto)")
                    loading_slot = right_col.empty()
                    loading_slot.markdown("<div class='muted'><span class='pulse'></span> AI summarizing the fetched records...</div>", unsafe_allow_html=True)
                    try:
                        summary = ai_summarize_records(df_clean, chosen_species)
                        # persist summary so it doesn't vanish on rerun
                        st.session_state["last_summary"] = summary
                        right_col.markdown(summary)
                    except Exception as e:
                        right_col.error(f"Auto-summary failed: {e}")
                    finally:
                        loading_slot.empty()


                # --- DATASET-AWARE AI CHAT (persistent) ---
                right_col.markdown("###  Ask AI about this dataset")
                # Dataset-aware question input — use key only so Streamlit stores value in session_state reliably
                ai_data_query = right_col.text_area(
                    "Ask a question about the current dataset (the AI will see a small sample and previous summary):",
                    key="data_ai_input",
                    height=120
                )

                # When user clicks, send compact context + question to LLM and store the reply in session_state
                if right_col.button("Ask AI about data", key="ask_data_ai"):
                    # Read the latest user question from session (guaranteed to be stable across reruns)
                    ai_data_query = st.session_state.get("data_ai_input", "").strip()
                    st.session_state["last_data_ai_query"] = ai_data_query

                    # prepare compact sample (safe size)
                    sample_for_ai = None
                    try:
                        df_for_sample = load_obis_df()
                        if df_for_sample is not None and not df_for_sample.empty:
                            sample_for_ai = df_for_sample.head(30).to_dict(orient="records")

                    except Exception:
                        sample_for_ai = None

                    # build messages
                    system_msg = {
                        "role": "system",
                        "content": "You are a helpful marine biology data assistant. Analyze the sample and answer the user's question concisely, suggest one next-step analysis."
                    }
                    parts = []
                    if sample_for_ai:
                        parts.append(f"Sample records (up to 30 rows): {sample_for_ai}")
                    if st.session_state.get("last_summary"):
                        parts.append(f"Existing AI summary: {st.session_state.get('last_summary')}")
                    parts.append(f"User question: {ai_data_query or '[NO QUESTION]'}")
                    user_msg = {"role": "user", "content": "\n\n".join(parts)}

                    # show a lightweight loading pulse in the right column while the LLM runs
                    loading_slot = right_col.empty()
                    loading_slot.markdown("<div class='muted'><span class='pulse'></span> AI analyzing dataset...</div>", unsafe_allow_html=True)
                    try:
                        ai_reply = ask_openrouter([system_msg, user_msg])
                        # persist short history
                        entry = {"question": ai_data_query, "answer": ai_reply, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
                        st.session_state.setdefault("data_ai_history", []).append(entry)
                        # store latest reply for UI uses and persist as "last_summary" so it doesn't vanish
                        st.session_state["last_data_ai_reply"] = ai_reply
                        st.session_state["last_summary"] = ai_reply
                        # show answer
                        right_col.markdown("**AI answer:**")
                        right_col.markdown(ai_reply)
                    except Exception as e:
                        right_col.error(f"AI request failed: {e}")
                    finally:
                        loading_slot.empty()


                # show recent dataset-AI Q&A (most recent first)
                if st.session_state.get("data_ai_history"):
                    right_col.markdown("#### Recent dataset queries")
                    # show up to last 6 queries
                    for item in reversed(st.session_state["data_ai_history"][-6:]):
                        with right_col.expander(f"Q: {item['question'][:60] or '(no question)'} — {item['time']}", expanded=False):
                            right_col.markdown(f"**Q:** {item['question']}")
                            right_col.markdown(f"**A:** {item['answer']}")


                # PDF report generation button
                                # PDF report generation button
                right_col.markdown("###  Export")
                if right_col.button("Generate PDF report (maps + summary)"):
                    right_col.info("Generating PDF... this may take a few seconds.")
                    try:
                        # Reconstruct dataset from session
                        df_for_pdf = load_obis_df()
                        if df_for_pdf is None or df_for_pdf.empty:
                            right_col.error("No cached dataset available for PDF generation. Fetch a species first.")
                        else:
                            # prepare figures from the reconstructed df (do not rely on a local 'figs' variable that disappears after rerun)
                            figs_local = make_plots_from_df(df_for_pdf, st.session_state.get("last_species", ""))
                            summary_text = st.session_state.get("last_summary", "")
                            pdf_loading = right_col.empty()
                            pdf_loading.markdown("<div class='muted'><span class='pulse'></span> Rendering PDF...</div>", unsafe_allow_html=True)
                            try:
                                pdf_bytes = generate_pdf_report(df_for_pdf, st.session_state.get("last_species", ""), summary_text, figs_local)
                                # force bytes
                                if not isinstance(pdf_bytes, (bytes, bytearray)):
                                    try:
                                        pdf_bytes = pdf_bytes.getvalue()
                                    except Exception as e:
                                        raise RuntimeError(f"PDF generator returned non-bytes and could not be converted: {e}")

                                # store PDF in session for persistent download
                                fname = f"{(st.session_state.get('last_species') or 'obis_report').replace(' ','_')}_report.pdf"
                                st.session_state["last_pdf"] = pdf_bytes
                                st.session_state["last_pdf_name"] = fname
                                st.session_state["last_pdf_species"] = st.session_state.get("last_species")

                                right_col.success("PDF generated — you can download it below.")
                                right_col.markdown(f"**PDF size:** {len(pdf_bytes):,} bytes")

                                # immediate download (unique key)
                                try:
                                    dl_button(right_col, "Download PDF report (immediate)", data=pdf_bytes, file_name=fname,
                                            mime="application/pdf", base=f"pdf_{fname}")
                                except Exception as e:
                                    right_col.error(f"Could not create immediate download button: {e}")

                                # DO NOT clear the cached dataset here. Keep data for subsequent AI questions/downloads.
                            except Exception as e:
                                right_col.error(f"PDF creation failed: {e}")
                                right_col.info("Common cause: missing kaleido or reportlab in the environment. Install them and restart Streamlit.")
                            finally:
                                pdf_loading.empty()
                    except Exception as e:
                        right_col.error(f"Failed to create PDF: {e}")


                        # if AI summary not present, generate one (non-blocking)
                        summary_text = ""
                        if auto_summary:
                            try:
                                summary_text = ai_summarize_records(df_clean, chosen_species)
                            except Exception as e:
                                summary_text = ""
                                right_col.warning(f"AI summary failed: {e}")
                        else:
                            summary_text = "No AI summary requested (toggle auto-summary in sidebar)."

                                                # Attempt PDF creation and surface debug info
                        # Reconstruct df from session (defensive)
                        df_for_pdf = load_obis_df()
                        if df_for_pdf is None:
                            right_col.error("Dataset not available for PDF generation.")
                        else:
                            pdf_loading = right_col.empty()
                            pdf_loading.markdown("<div class='muted'><span class='pulse'></span> Rendering PDF...</div>", unsafe_allow_html=True)
                            try:
                                pdf_bytes = generate_pdf_report(df_for_pdf, chosen_species, summary_text, figs)
                                # coerce to bytes if necessary
                                if not isinstance(pdf_bytes, (bytes, bytearray)):
                                    try:
                                        pdf_bytes = pdf_bytes.getvalue()
                                    except Exception as e:
                                        raise RuntimeError(f"PDF generator returned non-bytes and could not be converted: {e}")

                                # store PDF in session for persistent download
                                fname = f"{chosen_species.replace(' ','_')}_report.pdf"
                                st.session_state["last_pdf"] = pdf_bytes
                                st.session_state["last_pdf_name"] = fname
                                st.session_state["last_pdf_species"] = chosen_species

                                right_col.success("PDF generated — you can download it below.")
                                right_col.markdown(f"**PDF size:** {len(pdf_bytes):,} bytes")

                                # immediate download button in right column (unique key)
                                try:
                                    dl_button(right_col, "Download PDF report (immediate)", data=pdf_bytes, file_name=fname,
                                              mime="application/pdf", base=f"pdf_{chosen_species.replace(' ','_')}")
                                except Exception as e:
                                    right_col.error(f"Could not create immediate download button: {e}")

                            except Exception as e:
                                right_col.error(f"PDF creation failed: {e}")
                                right_col.info("Common cause: missing kaleido or reportlab in the Python environment that runs Streamlit.")
                                right_col.info("Install in the same interpreter and restart Streamlit (run the install commands in the terminal used to start Streamlit).")
                            finally:
                                pdf_loading.empty()



                    except Exception as e:
                        right_col.error(f"Failed to create PDF: {e}")



        except Exception as e:
            loader.empty()
            left_col.error(f"Failed to fetch OBIS data: {e}")

    else:
        # LLM decided to answer directly
        ai_query = decision.get("query") or user_input
        right_col.markdown("### AI Response")
        right_col.markdown("<div class='muted'>AI interpreted your input as a question — here's the answer.</div>", unsafe_allow_html=True)
        system_msg = {"role":"system","content":"You are a marine biology data assistant. Answer clearly and concisely."}
        user_msg = {"role":"user","content":f"User input: {ai_query}"}
        try:
            with st.spinner("AI thinking..."):
                reply = ask_openrouter([system_msg, user_msg])
            right_col.markdown(reply)
        except Exception as e:
            right_col.error(f"AI request failed: {e}")

# If a PDF was generated earlier in this session, show a persistent download button
if st.session_state.get("last_pdf"):
    try:
        pdf_bytes = st.session_state["last_pdf"]
        # Accept either bytes or BytesIO-like objects
        if isinstance(pdf_bytes, (bytes, bytearray)):
            pdf_data = pdf_bytes
        else:
            # e.g., io.BytesIO
            try:
                pdf_data = pdf_bytes.getvalue()
            except Exception:
                pdf_data = bytes(pdf_bytes)

        pdf_name = st.session_state.get("last_pdf_name", "obis_report.pdf")
        st.markdown("### Last generated PDF")
        # create a stable unique key for the download button to avoid widget collisions
        key = _make_dl_key("last_pdf", pdf_name)
        st.download_button(
            "Download last generated PDF",
            data=pdf_data,
            file_name=pdf_name,
            mime="application/pdf",
            key=key,
        )
    except Exception as e:
        st.warning(f"PDF available but download button failed: {e}")




# Footer
st.markdown("---")
st.markdown("<div class='small muted'>Data: OBIS (api.obis.org) • LLM: OpenRouter • Created by Pushpal Sanyal</div>", unsafe_allow_html=True)
st.markdown("<div class='muted small'>Last updated: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "</div>", unsafe_allow_html=True)
