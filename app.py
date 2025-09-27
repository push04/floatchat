# app.py
# A single, full-featured Streamlit app merging OBIS + ERDDAP + AI + Auto name routing.

import os
import io
import re
import sys
import math
import json
import uuid
import time
import base64
import tempfile
import calendar
import html
from io import StringIO, BytesIO
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List

import requests
import pandas as pd
import numpy as np

import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import xarray as xr
from netCDF4 import Dataset  # noqa: F401  # xarray netcdf backend uses it

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# ----------- CONSTANTS & CONFIG -----------
OBIS_API_URL = "https://api.obis.org/v3/occurrence"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))

# ERDDAP defaults/curation (inlined from your module)
ERDDAP_SERVER_DEFAULT = "https://coastwatch.noaa.gov/erddap"
NRT_CUTOFF_DAYS = 60
CURATED_DATASETS = {
    "global": [
        "noaacwBLENDEDsstDaily",
        "noaacwBLENDEDsstDNDaily",
        "OISSTs_2022_v05_1",
        "jplMURSST41",
        "noaacwecnMURannual",
    ],
    "https://erddap.incois.gov.in/erddap": [
        "NOAA_AVHRR_AMSR_datasets",
        "incois_argo_sst_weekly",
        "incois_valueadded_products_datasets",
        "AMSR2_3day_Global",
        "incois_argo_10d_VAM",
    ],
    "https://erddap.aoml.noaa.gov/hdb/erddap": ["OISSTs_2022_v05_1"],
    "https://erddap.marine.usf.edu/erddap": ["jplMURSST41"],
    "https://coastwatch.pfeg.noaa.gov/erddap": ["jplMURSST41", "noaacwBLENDEDsstDaily"],
}

# ----------- STREAMLIT PAGE -----------
st.set_page_config(
    page_title="FloatChat | AI-Powered Ocean Data Discovery",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- THEME / CSS -----------
THEMES = {
    "Ocean Dark": """
    .stApp { background: linear-gradient(180deg,#071022 0%, #081526 45%, #0b2b3a 100%); color:#e6f0f6; font-family: Inter, Segoe UI, Roboto, sans-serif; }
    .card { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 16px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.04);}
    .stat { display:inline-block; padding:10px 14px; margin-right:8px; background: rgba(255,255,255,0.04); border-radius:12px; border:1px solid rgba(255,255,255,0.06); }
    .muted { color:#9fb7c7; } .small { font-size:0.85rem; color:#bcd7e6; }
    button.stButton>button { background: linear-gradient(90deg,#0b84ff 0%, #6ee7b7 100%); color:#06202a; font-weight:600; border-radius:12px; }
    .chip { display:inline-block; padding:6px 10px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.1); border-radius:999px; font-size:0.8rem; margin-right:6px; }
    .pulse { display:inline-block; width:10px;height:10px;background:#0b84ff;border-radius:50%; animation:pulse 1.2s infinite; }
    @keyframes pulse { 0%{transform:scale(.8);opacity:.95} 50%{transform:scale(1.35);opacity:.35} 100%{transform:scale(.8);opacity:.95} }
    """,
    "Clean Light": """
    .stApp { background:#f7fafc; color:#0f172a; font-family: Inter, Segoe UI, Roboto, sans-serif; }
    .card { background:#ffffff; border-radius:16px; padding:16px; box-shadow: 0 10px 30px rgba(2,6,23,0.08); border:1px solid #e5e7eb; }
    .stat { display:inline-block; padding:10px 14px; margin-right:8px; background:#f8fafc; border-radius:12px; border:1px solid #e5e7eb; }
    .muted { color:#64748b; } .small { font-size:0.85rem; color:#475569; }
    button.stButton>button { background: linear-gradient(90deg,#2563eb 0%, #22c55e 100%); color:white; font-weight:600; border-radius:12px; }
    .chip { display:inline-block; padding:6px 10px; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:999px; font-size:0.8rem; margin-right:6px; }
    .pulse { display:inline-block; width:10px;height:10px;background:#2563eb;border-radius:50%; animation:pulse 1.2s infinite; }
    @keyframes pulse { 0%{transform:scale(.8);opacity:.95} 50%{transform:scale(1.35);opacity:.35} 100%{transform:scale(.8);opacity:.95} }
    """,
}
if "theme" not in st.session_state:
    st.session_state["theme"] = "Ocean Dark"
st.markdown(f"<style>{THEMES[st.session_state['theme']]}</style>", unsafe_allow_html=True)

# ----------- HELPERS (session-safe) -----------
def _style_plotly_light(fig):
    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#06202a", size=11),
            legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#d1d5db", borderwidth=0.5),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e6eef6", zerolinecolor="#e6eef6", tickcolor="#06202a")
        fig.update_yaxes(showgrid=True, gridcolor="#e6eef6", zerolinecolor="#e6eef6", tickcolor="#06202a")
    except Exception:
        pass

def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            from streamlit.runtime.scriptrunner import RerunException
            raise RerunException()
        except Exception:
            st.session_state["_force_rerun_ts"] = time.time()

try:
    cache_data = st.cache_data
except Exception:
    cache_data = getattr(st, "cache", None) or (lambda **kw: (lambda f: f))

def _make_dl_key(base: str, filename: str) -> str:
    safe = (base + "_" + filename).replace(" ", "_").replace("/", "_")[:200]
    st.session_state.setdefault("_dl_keys", [])
    if safe in st.session_state["_dl_keys"]:
        return safe
    if any(s.startswith(safe + "_") for s in st.session_state["_dl_keys"]):
        safe = f"{safe}_{uuid.uuid4().hex[:8]}"
    st.session_state["_dl_keys"].append(safe)
    return safe

def dl_button(container, label, data, file_name, mime, base="dl"):
    key = _make_dl_key(base, file_name)
    return container.download_button(label, data=data, file_name=file_name, mime=mime, key=key)

def _sanitize_value(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if pd.isna(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime, date)):
            return pd.to_datetime(v).isoformat()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (bytes, bytearray)):
            try: return v.decode("utf-8")
            except Exception: return str(v)
        return v
    except Exception:
        return str(v)

def save_df(df: pd.DataFrame, key_prefix="obis"):
    records = [{k: _sanitize_value(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    st.session_state[f"{key_prefix}_records"] = records
    st.session_state[f"{key_prefix}_columns"] = list(df.columns)

def load_df(key_prefix="obis") -> Optional[pd.DataFrame]:
    recs = st.session_state.get(f"{key_prefix}_records")
    cols = st.session_state.get(f"{key_prefix}_columns")
    if recs is None or cols is None:
        return None
    df = pd.DataFrame(recs, columns=cols)
    for dcol in ["eventDate", "time"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df

def _auto_download_pdf_bytes(pdf_bytes: bytes, filename: str):
    try:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        html_snip = f"""
        <a id="dl" href="data:application/pdf;base64,{b64}" download="{filename}"></a>
        <script>setTimeout(()=>document.getElementById('dl').click(), 60);</script>
        """
        components.html(html_snip, height=0)
        return True
    except Exception:
        return False

# ----------- NAME ROUTING (Common → Scientific) -----------
BINOMIAL_RE = re.compile(r"^[A-Z][a-zA-Z-]+ [a-z][a-zA-Z-]+$")

def looks_binomial(name: str) -> bool:
    return bool(BINOMIAL_RE.match(name.strip()))

@cache_data(ttl=60*60)
def gbif_resolve(name: str) -> Optional[str]:
    # Try suggest endpoint first
    try:
        r = requests.get("https://api.gbif.org/v1/species/suggest", params={"q": name, "limit": 1}, timeout=15)
        if r.ok and isinstance(r.json(), list) and r.json():
            sci = r.json()[0].get("scientificName")
            if sci: return sci
    except Exception:
        pass
    # Fallback to match
    try:
        r = requests.get("https://api.gbif.org/v1/species/match", params={"name": name}, timeout=15)
        if r.ok:
            j = r.json()
            if isinstance(j, dict):
                sci = j.get("scientificName")
                if sci: return sci
    except Exception:
        pass
    return None

@cache_data(ttl=60*60)
def worms_resolve(name: str) -> Optional[str]:
    try:
        r = requests.get(f"https://www.marinespecies.org/rest/AphiaRecordsByName/{requests.utils.quote(name)}?like=true&marine_only=true", timeout=15)
        if r.ok and isinstance(r.json(), list) and r.json():
            sci = r.json()[0].get("scientificname")
            if sci: return sci
    except Exception:
        pass
    return None

def resolve_common_to_scientific(query: str) -> Tuple[str, str]:
    """Return (scientific_name, provenance)"""
    q = query.strip()
    if looks_binomial(q):
        return q, "User provided binomial"
    # GBIF first
    sci = gbif_resolve(q)
    if sci: return sci, "Resolved via GBIF"
    # WoRMS fallback
    sci = worms_resolve(q)
    if sci: return sci, "Resolved via WoRMS"
    # Heuristic: title-case first two words
    parts = q.split()
    if len(parts) >= 2:
        return f"{parts[0].capitalize()} {parts[1].lower()}", "Heuristic guess"
    return q, "Unchanged"

# ----------- OBIS INTEGRATION -----------
def _sanitize_bbox(bbox: Dict[str, Any]) -> Dict[str, float]:
    """Validate and coerce a bounding box dictionary."""
    required = ("lonmin", "lonmax", "latmin", "latmax")
    missing = [k for k in required if k not in bbox]
    if missing:
        raise ValueError(f"Missing keys: {', '.join(missing)}")
    try:
        lonmin = float(bbox["lonmin"])
        lonmax = float(bbox["lonmax"])
        latmin = float(bbox["latmin"])
        latmax = float(bbox["latmax"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Bounding box values must be numeric") from exc
    if lonmin >= lonmax:
        raise ValueError("Longitude minimum must be less than maximum")
    if latmin >= latmax:
        raise ValueError("Latitude minimum must be less than maximum")
    if not (-180.0 <= lonmin <= 180.0 and -180.0 <= lonmax <= 180.0):
        raise ValueError("Longitudes must be within -180 to 180")
    if not (-90.0 <= latmin <= 90.0 and -90.0 <= latmax <= 90.0):
        raise ValueError("Latitudes must be within -90 to 90")
    return {"lonmin": lonmin, "lonmax": lonmax, "latmin": latmin, "latmax": latmax}


@cache_data(ttl=60 * 30)
def fetch_obis_records(
    species_name: str,
    size: int = 200,
    bbox: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch OBIS occurrences, returning (dataframe, error_message)."""

    try:
        size_int = max(1, int(size))
    except (TypeError, ValueError):
        return pd.DataFrame(), "Requested record count must be an integer"

    params = {"scientificname": species_name, "size": size_int}

    if bbox:
        try:
            bbox_clean = _sanitize_bbox(bbox)
        except ValueError as exc:
            return pd.DataFrame(), f"Invalid bounding box: {exc}"
        poly = (
            "POLYGON(("
            f"{bbox_clean['lonmin']} {bbox_clean['latmin']}, "
            f"{bbox_clean['lonmax']} {bbox_clean['latmin']}, "
            f"{bbox_clean['lonmax']} {bbox_clean['latmax']}, "
            f"{bbox_clean['lonmin']} {bbox_clean['latmax']}, "
            f"{bbox_clean['lonmin']} {bbox_clean['latmin']}))"
        )
        params["geometry"] = poly

    try:
        r = requests.get(OBIS_API_URL, params=params, timeout=40)
    except requests.RequestException as exc:
        return pd.DataFrame(), f"OBIS request failed: {exc}"

    if r.status_code >= 400:
        detail = None
        try:
            payload = r.json()
            if isinstance(payload, dict):
                for key in ("message", "error", "detail"):
                    if payload.get(key):
                        detail = str(payload[key])
                        break
        except ValueError:
            # Response body is not JSON – fall back to reason/text
            pass

        if not detail:
            detail = r.reason or r.text[:200]
        return pd.DataFrame(), f"OBIS request failed with status {r.status_code}: {detail}"

    try:
        js = r.json()
    except ValueError as exc:
        return pd.DataFrame(), f"OBIS response was not valid JSON: {exc}"

    results = js.get("results", []) if isinstance(js, dict) else []
    if not isinstance(results, list):
        return pd.DataFrame(), "OBIS response format was unexpected"

    return pd.DataFrame(results), None

def make_plots_from_df(df: pd.DataFrame, species_name: str) -> Dict[str, Any]:
    figs = {}
    if {"decimalLongitude", "decimalLatitude"}.issubset(df.columns):
        map_df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
        sample_map = map_df if len(map_df) <= 1500 else map_df.sample(1500, random_state=1)
        f = px.scatter_geo(
            sample_map,
            lon="decimalLongitude",
            lat="decimalLatitude",
            hover_name="scientificName" if "scientificName" in sample_map.columns else None,
            hover_data=[c for c in ["eventDate", "depth"] if c in sample_map.columns],
            title=f"{species_name} occurrences (sample)",
            projection="natural earth",
            height=560,
        )
        f.update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"))
        _style_plotly_light(f)
        figs["map"] = f

    if "eventDate" in df.columns:
        try:
            dft = df.copy()
            dft["eventDate"] = pd.to_datetime(dft["eventDate"], errors="coerce")
            t = dft.dropna(subset=["eventDate"])
            if not t.empty:
                yearly = t.groupby(t["eventDate"].dt.year).size().reset_index(name="count")
                fy = px.bar(yearly, x="eventDate", y="count", title="Records per year", height=300)
                _style_plotly_light(fy)
                figs["yearly"] = fy
                monthly = t.groupby(t["eventDate"].dt.to_period("M")).size().reset_index(name="count")
                monthly["eventMonth"] = monthly["eventDate"].astype(str)
                fm = px.line(monthly, x="eventMonth", y="count", title="Records per month", height=300)
                _style_plotly_light(fm)
                figs["monthly"] = fm
        except Exception:
            pass

    if "depth" in df.columns:
        try:
            d = pd.to_numeric(df["depth"], errors="coerce").dropna()
            if not d.empty:
                fd = px.histogram(d, x=d, nbins=30, title="Depth distribution", labels={"x": "Depth (m)", "count": "Frequency"}, height=300)
                _style_plotly_light(fd)
                figs["depth_hist"] = fd
        except Exception:
            pass

    if {"decimalLongitude", "decimalLatitude"}.issubset(df.columns):
        try:
            heat_df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
            if len(heat_df) >= 20:
                fh = px.density_heatmap(heat_df, x="decimalLongitude", y="decimalLatitude", nbinsx=60, nbinsy=40,
                                        title="Density heatmap (lon/lat)", height=400)
                _style_plotly_light(fh)
                figs["density"] = fh
        except Exception:
            pass

    # Optional: measurement plots (auto-detect)
    def _find(cands):  # case-insensitive match
        lowmap = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns: return c
            if c.lower() in lowmap: return lowmap[c.lower()]
        return None
    temp_col = _find(["temperature", "temp", "water_temperature", "t", "sea_temperature"])
    sal_col = _find(["salinity", "psal", "salt", "psal_ctd"])

    if temp_col is not None:
        s = pd.to_numeric(df[temp_col], errors="coerce").dropna()
        if not s.empty:
            ft = px.histogram(s, x=s, nbins=40, title=f"Temperature distribution ({temp_col})", labels={"x": "Temperature (°C)"}, height=300)
            _style_plotly_light(ft); figs["temperature_hist"] = ft
        if "eventDate" in df.columns:
            tts = df.dropna(subset=[temp_col, "eventDate"]).copy()
            tts[temp_col] = pd.to_numeric(tts[temp_col], errors="coerce")
            tts = tts.dropna(subset=[temp_col])
            if not tts.empty:
                ftts = px.line(tts.sort_values("eventDate"), x="eventDate", y=temp_col, title=f"Temperature time series ({temp_col})", height=300)
                _style_plotly_light(ftts); figs["temperature_ts"] = ftts

    if sal_col is not None:
        s = pd.to_numeric(df[sal_col], errors="coerce").dropna()
        if not s.empty:
            fs = px.histogram(s, x=s, nbins=40, title=f"Salinity distribution ({sal_col})", labels={"x": "Salinity (PSU)"}, height=300)
            _style_plotly_light(fs); figs["salinity_hist"] = fs
        if "eventDate" in df.columns:
            sts = df.dropna(subset=[sal_col, "eventDate"]).copy()
            sts[sal_col] = pd.to_numeric(sts[sal_col], errors="coerce")
            sts = sts.dropna(subset=[sal_col])
            if not sts.empty:
                fsts = px.line(sts.sort_values("eventDate"), x="eventDate", y=sal_col, title=f"Salinity time series ({sal_col})", height=300)
                _style_plotly_light(fsts); figs["salinity_ts"] = fsts

    return figs

def prepare_csv_download(df: pd.DataFrame) -> str:
    buf = StringIO(); df.to_csv(buf, index=False); return buf.getvalue()

def prepare_excel_download(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="records")
    return buf.getvalue()

def generate_pdf_report(df: pd.DataFrame, species_name: str, summary_text: str, figs: Dict[str, Any]) -> bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4

    images = []
    for key in ["map", "yearly", "monthly", "depth_hist", "density", "temperature_hist", "salinity_hist"]:
        if key in figs:
            try:
                images.append((key, pio.to_image(figs[key], format="png", scale=2, engine="kaleido")))
            except Exception:
                try:
                    images.append((key, pio.to_image(figs[key], format="png", scale=2)))
                except Exception:
                    images.append((key, None))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Meta", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#6b7280")))
    styles.add(ParagraphStyle(name="Heading", parent=styles["Heading2"], fontSize=12, leading=14))
    styles.add(ParagraphStyle(name="NormalSmall", parent=styles["Normal"], fontSize=10, leading=12))
    story = []

    title = f"OBIS Report — {species_name}" if species_name else "OBIS Report"
    story.append(Paragraph(title, styles["ReportTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Meta"]))
    story.append(Paragraph(f"Total records: {len(df)}", styles["Meta"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI Summary", styles["Heading"]))
    if summary_text:
        for para in str(summary_text).split("\n\n"):
            story.append(Paragraph(para.replace("\n", "<br/>"), styles["NormalSmall"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No AI summary available.", styles["NormalSmall"]))
    story.append(Spacer(1, 8))

    maxw = 515  # approx doc.width
    for title, img_bytes in images:
        if img_bytes:
            img_io = BytesIO(img_bytes)
            img = RLImage(img_io)
            img.drawWidth = maxw
            img.drawHeight = maxw * (img.imageHeight / float(img.imageWidth))
            story.append(Paragraph(title.replace("_", " ").title(), styles["Heading"]))
            story.append(img)
            story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("Sample Records (first 100)", styles["Heading"]))
    preferred = ["scientificName", "eventDate", "decimalLongitude", "decimalLatitude", "depth", "basisOfRecord", "institutionCode"]
    cols = [c for c in preferred if c in df.columns] or list(df.columns[:6])
    display = df.head(100)
    def fmt(v): 
        if pd.isna(v): return ""
        s = str(v);  return (s[:77] + "...") if len(s) > 80 else s
    data = [cols] + [[fmt(r.get(c, "")) for c in cols] for _, r in display.iterrows()]
    from reportlab.platypus import Table
    table = Table(data, colWidths=[maxw/len(cols)]*len(cols), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b84ff")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d1d5db")),
    ]))
    story.append(table)

    def _pgnum(c, d): c.setFont("Helvetica",8); c.drawRightString(d.pagesize[0]-d.rightMargin, 12, f"Page {c.getPageNumber()}")
    doc.build(story, onFirstPage=_pgnum, onLaterPages=_pgnum)
    buf.seek(0)
    return buf.getvalue()

# ----------- AI (OpenRouter) -----------
def ask_openrouter(messages: List[Dict[str, str]], model=OPENROUTER_MODEL, timeout=60) -> str:
    if not OPENROUTER_API_KEY:
        return "Set OPENROUTER_API_KEY in Streamlit secrets or environment to enable AI."
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 700, "temperature": 0.2}
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or ch.get("delta") or {}
        return msg.get("content") or ch.get("text") or json.dumps(j)[:2000]
    except Exception as e:
        return f"AI error: {e}"

def ai_summarize_records(df: pd.DataFrame, species_name: str) -> str:
    sample = df.head(30).to_dict(orient="records")
    system = {"role": "system", "content": "You are a marine biology data assistant. Summarize non-technically."}
    user = {"role": "user", "content": f"I have {len(df)} records for '{species_name}'. Sample: {sample}. Provide 3–6 bullets on distribution/patterns + one next-step analysis."}
    return ask_openrouter([system, user])

# ----------- ERDDAP (inlined core from your module) -----------
def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def geocode_place(place: str, timeout=10) -> Optional[Tuple[float, float]]:
    if GEOPY_AVAILABLE:
        try:
            loc = Nominatim(user_agent="erddap_integration_geocoder").geocode(place, timeout=timeout)
            if loc: return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": place, "format":"json", "limit":1}, headers={"User-Agent":"erddap_integration/1.0"}, timeout=timeout)
        j = r.json()
        if isinstance(j, list) and j:
            return float(j[0]["lat"]), float(j[0]["lon"])
    except Exception:
        pass
    return None

def erddap_search(server: str, query: str, items_per_page: int = 200) -> pd.DataFrame:
    try:
        url = f"{server.rstrip('/')}/search/index.csv?searchFor={requests.utils.quote(query)}&itemsPerPage={items_per_page}"
        r = requests.get(url, timeout=30); r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

def get_dataset_info(server: str, dataset_id: str) -> Optional[dict]:
    try:
        url = f"{server.rstrip('/')}/info/{requests.utils.quote(dataset_id)}/index.json"
        r = requests.get(url, timeout=20); r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _extract_variable_names_from_info(info_json) -> List[str]:
    if not info_json: return []
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", str(info_json)))
    exclude = {'table','attributes','variable','dataset','dimension','id','units','title','name'}
    return [t for t in tokens if t.lower() not in exclude and len(t) < 60][:200]

def validate_dataset_for_keywords(server, dataset_id, keywords, accept_if_candidates=True):
    info = get_dataset_info(server, dataset_id)
    if not info: return {"valid": False, "variables": [], "info": None}
    txt = str(info).lower()
    found = [k for k in keywords if k.lower() in txt]
    cand = _extract_variable_names_from_info(info)
    valid = bool(found or (accept_if_candidates and cand))
    return {"valid": valid, "variables": found + cand, "info": info}

def discover_dataset(server, friendly_var, search_phrases, var_keywords, curated_fallback=None):
    q = " ".join(search_phrases)
    df = erddap_search(server, q)
    candidates = []
    if not df.empty:
        cols = [c.lower() for c in df.columns]
        for cnd in ["dataset id","dataset","Dataset ID","Dataset"]:
            if cnd.lower() in cols:
                ds_col = df.columns[cols.index(cnd.lower())]
                candidates = list(df[ds_col].dropna().unique()); break
    if curated_fallback:
        for c in curated_fallback:
            if c not in candidates: candidates.append(c)
    for ds in candidates:
        val = validate_dataset_for_keywords(server, ds, var_keywords, accept_if_candidates=True)
        if val["valid"]:
            var = val["variables"][0] if val["variables"] else None
            return ds, var, f"Discovered {ds}; var: {var}"
    return None, None, "No validated dataset found."

def _try_griddap_point(server, dataset_id, variable, start_iso, end_iso, lat, lon, depth=None, timeout=60):
    from pandas.errors import ParserError
    debug = []
    orders_with_depth = [
        ("time","depth","latitude","longitude"),
        ("time","latitude","longitude","depth"),
        ("time","latitude","depth","longitude"),
        ("time","longitude","latitude","depth"),
        ("time","depth","longitude","latitude"),
    ]
    orders_no_depth = [("time","latitude","longitude"), ("time","longitude","latitude")]

    def idx_for(order):
        parts=[]
        for dim in order:
            if dim=="time": parts.append(f"[({start_iso}):1:({end_iso})]")
            elif dim=="latitude": parts.append(f"[({lat}):1:({lat})]")
            elif dim=="longitude": parts.append(f"[({lon}):1:({lon})]")
            elif dim=="depth":
                if depth is None or depth=="ALL": return None
                parts.append(f"[({depth}):1:({depth})]")
        return "".join(parts)

    var_q = requests.utils.quote(variable) if variable else ""
    def build_url(dataset_id, var_q, idx):
        ds_q = requests.utils.quote(dataset_id)
        idx_q = requests.utils.quote(idx, safe="[]():,")
        if var_q: return f"{server.rstrip('/')}/griddap/{ds_q}.csv?{var_q}{idx_q}"
        return f"{server.rstrip('/')}/griddap/{ds_q}.csv?{idx_q}"

    seq = orders_with_depth if (depth is not None and depth!="ALL") else orders_no_depth
    for order in seq:
        idx = idx_for(order)
        if not idx: continue
        url = build_url(dataset_id, var_q, idx)
        try:
            r = requests.get(url, timeout=timeout)
            snippet = r.text[:800] + "..." if isinstance(r.text, str) and len(r.text) > 800 else r.text
            debug.append((getattr(r, "url", url), snippet))
            if r.status_code != 200: continue
            try:
                df = pd.read_csv(io.StringIO(r.text))
            except ParserError as pe:
                debug.append((url, f"PARSER_ERROR:{pe}")); continue
            cols_lower = {c.lower(): c for c in df.columns}
            ren = {}
            for k in ['time','latitude','longitude','depth','z','altitude','depthBelowSeaSurface']:
                if k in cols_lower:
                    ren[cols_lower[k]] = 'depth' if k in ['depth','z','depthBelowSeaSurface'] else k
            if ren: df.rename(columns=ren, inplace=True)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                df = df.dropna(subset=['time'])
            return df, debug
        except Exception as e:
            debug.append((url, f"EXC:{e}"))
    return pd.DataFrame(), debug

def _try_tabledap(server, dataset_id, variable, start_iso, end_iso, lat, lon, timeout=60):
    from pandas.errors import ParserError
    debug = []
    var_part = variable if variable else ""
    base = f"{server.rstrip('/')}/tabledap/{requests.utils.quote(dataset_id)}.csv"
    url = f"{base}?{requests.utils.quote(var_part)}" if var_part else base
    params = { "time>=": start_iso, "time<=": end_iso, "latitude": lat, "longitude": lon }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        snippet = r.text[:800] + "..." if isinstance(r.text, str) and len(r.text) > 800 else r.text
        debug.append((getattr(r, "url", url), snippet))
        if r.status_code != 200: return pd.DataFrame(), debug
        try:
            df = pd.read_csv(io.StringIO(r.text))
        except ParserError as pe:
            debug.append((url, f"PARSER_ERROR:{pe}")); return pd.DataFrame(), debug
        cols_lower = {c.lower(): c for c in df.columns}
        ren = {}
        for k in ['time','latitude','longitude','depth','z','altitude','depthBelowSeaSurface']:
            if k in cols_lower:
                ren[cols_lower[k]] = 'depth' if k in ['depth','z','depthBelowSeaSurface'] else k
        if ren: df.rename(columns=ren, inplace=True)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            df = df.dropna(subset=['time'])
        return df, debug
    except Exception as e:
        debug.append((url, f"EXC:{e}"))
        return pd.DataFrame(), debug

def fetch_with_3d_support(server, dataset_id, variable, lat, lon, start_dt, end_dt, timeout=60):
    cutoff = datetime.now(timezone.utc) - timedelta(days=NRT_CUTOFF_DAYS)
    periods = []
    if end_dt < cutoff: periods.append((start_dt, end_dt))
    elif start_dt >= cutoff: periods.append((start_dt, end_dt))
    else:
        periods.append((start_dt, cutoff - timedelta(seconds=1)))
        periods.append((cutoff, end_dt))
    all_dfs, debug = [], []
    for sdt, edt in periods:
        s_iso, e_iso = _format_iso(sdt), _format_iso(edt)
        df_g, dbg_g = _try_griddap_point(server, dataset_id, variable, s_iso, e_iso, lat, lon, depth=None, timeout=timeout)
        debug.extend(dbg_g)
        if not df_g.empty: all_dfs.append(df_g); continue
        df_t, dbg_t = _try_tabledap(server, dataset_id, variable, s_iso, e_iso, lat, lon, timeout=timeout)
        debug.extend(dbg_t)
        if not df_t.empty: all_dfs.append(df_t); continue
        df_g0, dbg_g0 = _try_griddap_point(server, dataset_id, variable, s_iso, e_iso, lat, lon, depth=0, timeout=timeout)
        debug.extend(dbg_g0)
        if not df_g0.empty: all_dfs.append(df_g0); continue
    if not all_dfs: return pd.DataFrame(), debug
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    if 'time' in df_all.columns: df_all = df_all.sort_values('time').reset_index(drop=True)
    return df_all, debug

def plot_timeseries(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'time' not in df.columns:
        return go.Figure().update_layout(title="No timeseries available")
    fig = px.line(df, x='time', y=varcol, title=title or f"Timeseries: {varcol}")
    fig.update_xaxes(rangeslider_visible=True)
    _style_plotly_light(fig); return fig

def plot_profile_heatmap(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'time' not in df.columns or 'depth' not in df.columns:
        return go.Figure().update_layout(title="No profile/heatmap available")
    pivot = df.pivot_table(index='depth', columns='time', values=varcol, aggfunc='mean')
    pivot = pivot.sort_index(ascending=True)
    fig = go.Figure(data=go.Heatmap(x=[str(t) for t in pivot.columns], y=list(pivot.index), z=pivot.values, colorbar=dict(title=varcol)))
    fig.update_layout(title=title or f"Depth-Time heatmap ({varcol})", yaxis=dict(autorange='reversed'))
    _style_plotly_light(fig); return fig

def plot_profile_scatter(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'depth' not in df.columns:
        return go.Figure().update_layout(title="No profile available")
    prof = df.groupby('depth')[varcol].mean().reset_index().sort_values('depth')
    fig = px.line(prof, x=varcol, y='depth', title=title or f"Vertical profile ({varcol})", markers=True)
    fig.update_yaxes(autorange='reversed'); _style_plotly_light(fig); return fig

def plot_map_latest(df, varcol, title=None):
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return go.Figure().update_layout(title="No spatial data")
    latest = df.loc[[df['time'].idxmax()]] if 'time' in df.columns else df.head(1)
    fig = px.scatter_geo(latest, lat='latitude', lon='longitude', hover_name=varcol if varcol in latest.columns else None,
                         hover_data=[c for c in ['time', varcol] if c in latest.columns], title=title or "Latest location")
    _style_plotly_light(fig); return fig

def render_html_report(output_filename, figures, df_table, caption="ERDDAP Results"):
    frags = [pio.to_html(fig, include_plotlyjs=False, full_html=False) for fig in figures]
    try:
        df_disp = df_table.copy()
        for c in df_disp.select_dtypes(include=["float64", "int64"]).columns:
            df_disp[c] = df_disp[c].round(4)
        html_table = df_disp.to_html(index=False)
    except Exception:
        html_table = df_table.to_html(index=False)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>ERDDAP Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{margin:20px;background:#f8f9fa;font-family:Segoe UI, Roboto, Arial}}</style></head><body>
<div class="container"><h2>{html.escape(caption)}</h2>"""
    for frag in frags:
        html_doc += f'<div class="card"><div class="card-body">{frag}</div></div>\n'
    html_doc += f'<div class="card"><div class="card-body"><h5>Data table</h5>{html_table}</div></div>\n'
    html_doc += '<hr/><p>Generated by app.py</p></div></body></html>'
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_filename

def erddap_streamlit_widget(server_default=ERDDAP_SERVER_DEFAULT):
    st.sidebar.header("ERDDAP 3D Query")
    var_choice = st.sidebar.selectbox("Variable", ["Temperature", "Salinity", "Chlorophyll"])
    place = st.sidebar.text_input("Place (e.g., 'Chennai, India')")
    manual_latlon = st.sidebar.text_input("Or lat,lon (e.g., '13.0827,80.2707')")
    month_year = st.sidebar.text_input("Month-Year (MM-YYYY) optional")
    server_input = st.sidebar.text_input("ERDDAP server", value=server_default)

    if st.sidebar.button("Fetch ERDDAP"):
        latlon = None
        if manual_latlon:
            try:
                p = [x.strip() for x in manual_latlon.split(',')]
                latlon = (float(p[0]), float(p[1]))
            except Exception:
                st.sidebar.error("Invalid lat,lon"); return
        elif place:
            with st.sidebar:
                with st.spinner("Geocoding..."):
                    latlon = geocode_place(place)
            if not latlon:
                st.sidebar.error("Geocoding failed"); return
        else:
            st.sidebar.warning("Provide a place or lat,lon"); return

        lat, lon = latlon
        if month_year:
            try:
                m, y = map(int, month_year.split('-'))
                _, last_day = calendar.monthrange(y, m)
                start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
                end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
            except Exception:
                st.sidebar.error("Invalid Month-Year format"); return
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)
        st.info(f"ERDDAP: {var_choice} @ ({lat:.3f},{lon:.3f}) from {_format_iso(start_dt)} to {_format_iso(end_dt)}")

        heuristics = {
            "Temperature": (["sea surface temperature","sst"], ["analysed_sst","sea_surface_temperature","sst","temperature","sstAnom","sst_anom"]),
            "Salinity": (["salinity","sss"], ["salinity","sea_surface_salinity","sss"]),
            "Chlorophyll": (["chlorophyll","chl"], ["chlor_a","chl","CHL_Weekly","chlorophyll"]),
        }
        search_terms, var_keywords = heuristics[var_choice]
        server_key = server_input.rstrip('/')
        curated = []
        if server_key in CURATED_DATASETS: curated += CURATED_DATASETS[server_key]
        curated += [d for d in CURATED_DATASETS.get("global", []) if d not in curated]

        with st.spinner("Searching dataset..."):
            ds_id, var_guess, note = discover_dataset(server_input, var_choice, search_terms, var_keywords, curated_fallback=curated)
        if ds_id and not var_guess:
            vars_candidates = _extract_variable_names_from_info(get_dataset_info(server_input, ds_id))
            if vars_candidates: var_guess = vars_candidates[0]
        if not ds_id:
            st.error("No dataset found: " + note); return

        with st.spinner("Fetching data..."):
            df, debug = fetch_with_3d_support(server_input, ds_id, var_guess, lat, lon, start_dt, end_dt)
        if df.empty:
            st.error("No data returned. Showing debug attempts:")
            st.write(debug[:8]); return

        data_cols = [c for c in df.columns if c.lower() not in ['time','latitude','longitude','depth']]
        varcol = data_cols[0] if data_cols else None
        st.plotly_chart(plot_timeseries(df, varcol), use_container_width=True)
        if 'depth' in df.columns:
            st.plotly_chart(plot_profile_heatmap(df, varcol), use_container_width=True)
            st.plotly_chart(plot_profile_scatter(df, varcol), use_container_width=True)
        st.plotly_chart(plot_map_latest(df, varcol), use_container_width=True)
        st.markdown("**ERDDAP data (sample):**")
        st.dataframe(df.head(50))

# ----------- SYNTHETIC NETCDF -----------
def generate_ocean_netcdf(outfile_path,
                          lon_min=68.0, lon_max=96.0,
                          lat_min=6.0, lat_max=24.0,
                          nx=40, ny=40, nt=12,
                          start_date=None, depths=None, variables=None):
    if start_date is None:
        start_date = pd.to_datetime(date.today()).normalize()
    if depths is None:
        depths = np.array([0,10,20,50,100,200], dtype=float)
    if variables is None:
        variables = ["temperature", "salinity"]

    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    times = pd.date_range(start=start_date, periods=nt, freq="MS")

    shape = (len(times), len(depths), len(lats), len(lons))
    base_temp = 15.0
    temp = np.zeros(shape, dtype=np.float32)
    salt = np.zeros(shape, dtype=np.float32)
    lon_grad = (lons[np.newaxis, :] - lon_min) / max(1e-9, (lon_max - lon_min))
    lat_grad = (lats[:, np.newaxis] - lat_min) / max(1e-9, (lat_max - lat_min))
    grid = (lat_grad[:,:,None] * lon_grad[None,None,:]).astype(np.float32)
    for ti in range(len(times)):
        seasonal = 2.0 * np.sin(2*np.pi*(ti/max(1, nt)))
        for di, d in enumerate(depths):
            depth_decay = np.exp(-d / 50.0)
            temp[ti, di] = base_temp + seasonal + 8.0*depth_decay + 0.5*grid
            salt[ti, di] = 35.0 + 0.01 * d + 0.2 * grid

    data_vars = {}
    if "temperature" in variables: data_vars["temperature"] = (("time","depth","lat","lon"), temp)
    if "salinity" in variables: data_vars["salinity"] = (("time","depth","lat","lon"), salt)
    ds = xr.Dataset(data_vars=data_vars, coords={"time": times, "depth": depths, "lat": lats, "lon": lons})
    ds.attrs["title"] = "FloatChat generated ocean dataset"
    ds.attrs["created_by"] = "FloatChat"
    ds.to_netcdf(outfile_path)
    return ds

def plot_variable_map_from_ds(ds, var="temperature", time_index=0, depth_index=0):
    if var not in ds: return None
    da = ds[var].isel(time=time_index, depth=depth_index)
    df = da.to_dataframe(name=var).reset_index()
    fig = px.scatter(df, x="lon", y="lat", color=var, size_max=6,
                     title=f"{var} (time={str(ds.time.values[time_index])}, depth={float(ds.depth.values[depth_index])} m)")
    _style_plotly_light(fig); return fig

def plot_variable_profile_at_point(ds, var="temperature", lon_val=None, lat_val=None, time_index=0):
    if var not in ds: return None
    if lon_val is None: lon_val = float(ds.lon.mean())
    if lat_val is None: lat_val = float(ds.lat.mean())
    lon_idx = int(np.abs(ds.lon.values - lon_val).argmin())
    lat_idx = int(np.abs(ds.lat.values - lat_val).argmin())
    da = ds[var].isel(time=time_index, lat=lat_idx, lon=lon_idx)
    prof = pd.DataFrame({"depth": ds.depth.values, var: da.values})
    fig = px.line(prof, x=var, y="depth", title=f"{var} profile at lon={lon_val:.2f}, lat={lat_val:.2f}")
    fig.update_yaxes(autorange="reversed"); _style_plotly_light(fig); return fig

def plot_variable_timeseries_at_point(ds, var="temperature", lon_val=None, lat_val=None, depth_index=0):
    if var not in ds: return None
    if lon_val is None: lon_val = float(ds.lon.mean())
    if lat_val is None: lat_val = float(ds.lat.mean())
    lon_idx = int(np.abs(ds.lon.values - lon_val).argmin())
    lat_idx = int(np.abs(ds.lat.values - lat_val).argmin())
    da = ds[var].isel(depth=depth_index, lat=lat_idx, lon=lon_idx)
    ts = pd.DataFrame({"time": ds.time.values, var: da.values})
    fig = px.line(ts, x="time", y=var, title=f"{var} timeseries at lon={lon_val:.2f}, lat={lat_val:.2f}, depth={float(ds.depth.values[depth_index])} m")
    _style_plotly_light(fig); return fig

# ----------- UI HEADER -----------
st.title("FloatChat: AI-Powered ARGO & OBIS Ocean Data Explorer")
st.markdown(
    "<div class='muted'>Type a <b>common name</b> (e.g., <i>Indian oil sardine</i>) or a <b>scientific name</b> (e.g., <i>Sardinella longiceps</i>). We’ll auto-route and search.</div>",
    unsafe_allow_html=True,
)

# ----------- SIDEBAR -----------
with st.sidebar:
    st.markdown("## Controls")
    theme = st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state["theme"]))
    if theme != st.session_state["theme"]:
        st.session_state["theme"] = theme
        st.markdown(f"<style>{THEMES[theme]}</style>", unsafe_allow_html=True)

    max_records = st.slider("Max OBIS records", 10, 3000, 500, step=10)
    bbox_enable = st.checkbox("Filter by bounding box", value=False)
    if bbox_enable:
        lon_min = st.number_input("Lon min", value=68.0, step=0.1, format="%.3f")
        lon_max = st.number_input("Lon max", value=96.0, step=0.1, format="%.3f")
        lat_min = st.number_input("Lat min", value=6.0, step=0.1, format="%.3f")
        lat_max = st.number_input("Lat max", value=24.0, step=0.1, format="%.3f")

    st.markdown("---")
    st.markdown("## Date range")
    start_date = st.date_input("Start", value=date(2000, 1, 1))
    end_date = st.date_input("End", value=date.today())
    if start_date and end_date and start_date > end_date:
        st.warning("Start date is after end date.")

    st.markdown("---")
    st.markdown("## NetCDF (synthetic)")
    gen_enable = st.checkbox("Enable generator", value=False)
    if gen_enable:
        nc_nx = st.number_input("Longitude points (nx)", 8, 400, 40, step=8)
        nc_ny = st.number_input("Latitude points (ny)", 8, 400, 40, step=8)
        nc_nt = st.number_input("Time steps (nt)", 1, 48, 12)
        nc_depths_str = st.text_input("Depths (m, comma)", value="0,10,20,50,100,200")
        nc_vars = st.multiselect("Variables", ["temperature","salinity","oxygen","nitrate"], default=["temperature","salinity"])

    st.markdown("---")
    st.markdown("## AI")
    st.caption(f"Model: {OPENROUTER_MODEL}")
    auto_summary = st.checkbox("Auto-summarize after fetch", value=True)

    st.markdown("---")
    erddap_streamlit_widget()

# ----------- INPUT BAR -----------
with st.form(key="single_input_form", clear_on_submit=False):
    query = st.text_input("Enter common or scientific name, or ask a question", value="", key="single_input")
    submitted = st.form_submit_button("Submit")

# ----------- SAVED SEARCHES / BOOKMARKS -----------
st.session_state.setdefault("search_history", [])
st.session_state.setdefault("bookmarks", [])

def add_to_history(display, scientific, provenance):
    st.session_state["search_history"].append({
        "t": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "q": display, "sci": scientific, "prov": provenance
    })
    if len(st.session_state["search_history"]) > 20:
        st.session_state["search_history"] = st.session_state["search_history"][-20:]

def bookmark_current(species):
    if species and species not in st.session_state["bookmarks"]:
        st.session_state["bookmarks"].append(species)

# ----------- LAYOUT COLUMNS -----------
try:
    left, right = st.columns([2.4, 1.0], gap="large")
except TypeError:
    left, right = st.columns([2.4, 1.0])

# ----------- PREVIOUS DATA (persistent) -----------
with left:
    df_prev = load_df("obis")
    if df_prev is not None and not df_prev.empty:
        st.markdown("### Previously fetched OBIS records (cached)")
        c1, c2, c3 = st.columns([1,1,2])
        with c1: st.markdown(f"<div class='stat'><b>{len(df_prev)}</b><div class='small muted'>records</div></div>", unsafe_allow_html=True)
        with c2:
            unique_locs = df_prev.dropna(subset=['decimalLongitude','decimalLatitude']).shape[0]
            st.markdown(f"<div class='stat'><b>{unique_locs}</b><div class='small muted'>geo points</div></div>", unsafe_allow_html=True)
        with c3:
            rng = "-"
            if "eventDate" in df_prev.columns and df_prev["eventDate"].notna().any():
                mn, mx = df_prev["eventDate"].min(), df_prev["eventDate"].max()
                rng = f"{mn.date()} → {mx.date()}"
            st.markdown(f"<div class='small muted'>Date range: {rng}</div>", unsafe_allow_html=True)

        if {"decimalLongitude","decimalLatitude"}.issubset(df_prev.columns):
            map_df = df_prev.dropna(subset=["decimalLongitude","decimalLatitude"])
            if len(map_df) > 700: map_df = map_df.sample(700, random_state=1)
            fig = px.scatter_geo(map_df, lon="decimalLongitude", lat="decimalLatitude",
                                 hover_name="scientificName" if "scientificName" in map_df.columns else None,
                                 hover_data=["eventDate","depth"] if "eventDate" in df_prev.columns else None,
                                 projection="natural earth", height=430, title="Cached occurrences")
            fig.update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"))
            _style_plotly_light(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Sample cached records")
        st.dataframe(df_prev.head(200))

        # Downloads
        try:
            dl_button(st, "Download cached CSV", prepare_csv_download(df_prev), "obis_records.csv", "text/csv", base="cached_csv")
            xlsx = prepare_excel_download(df_prev)
            dl_button(st, "Download cached Excel", xlsx, "obis_records.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base="cached_xlsx")
        except Exception:
            dl_button(st, "Download cached CSV", prepare_csv_download(df_prev), "obis_records.csv", "text/csv", base="cached_csv_fallback")
            st.info("Install openpyxl for Excel export.")

        if st.button("Clear cached data"):
            for k in ["obis_records","obis_columns","last_species","last_summary","last_pdf","last_pdf_name","last_pdf_species","data_ai_history"]:
                st.session_state.pop(k, None)
            safe_rerun()

# ----------- RIGHT PANEL: AI/Q&A & EXPORT -----------
if df_prev is not None and not df_prev.empty:
    with right:
        st.markdown("### Ask AI about cached dataset")
        ai_q = st.text_area("Your question", key="data_ai_input", height=120)
        if st.button("Ask AI about cached data", key="ask_cached_ai"):
            sample = df_prev.head(30).to_dict(orient="records")
            parts = [f"Sample (<=30 rows): {sample}"]
            if st.session_state.get("last_summary"):
                parts.append(f"Existing AI summary: {st.session_state['last_summary']}")
            parts.append(f"Question: {ai_q or '[NO QUESTION]'}")
            with st.spinner("AI analyzing..."):
                ans = ask_openrouter([{"role":"system","content":"You are a helpful marine data assistant. Be concise."},
                                      {"role":"user","content":"\n\n".join(parts)}])
            st.session_state.setdefault("data_ai_history", []).append({"q": ai_q, "a": ans, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")})
            st.session_state["last_summary"] = st.session_state.get("last_summary") or ans
            st.markdown("**AI answer:**"); st.markdown(ans)

        if st.session_state.get("data_ai_history"):
            st.markdown("#### Recent dataset queries")
            for item in reversed(st.session_state["data_ai_history"][-6:]):
                with st.expander(f"Q: {item['q'][:60] if item['q'] else '(no question)'} — {item['time']}"):
                    st.markdown(f"**Q:** {item['q']}\n\n**A:** {item['a']}")

        st.markdown("### Export")
        if st.button("Generate PDF report"):
            df_for_pdf = load_df("obis")
            if df_for_pdf is None or df_for_pdf.empty:
                st.error("No dataset to export.")
            else:
                with st.spinner("Rendering PDF..."):
                    figs_local = make_plots_from_df(df_for_pdf, st.session_state.get("last_species",""))
                    pdf = generate_pdf_report(df_for_pdf, st.session_state.get("last_species",""), st.session_state.get("last_summary",""), figs_local)
                fname = f"{(st.session_state.get('last_species') or 'obis_report').replace(' ','_')}_report.pdf"
                st.session_state["last_pdf"] = pdf
                st.session_state["last_pdf_name"] = fname
                st.success(f"PDF ready — {len(pdf):,} bytes")
                auto_ok = _auto_download_pdf_bytes(pdf, fname)
                if not auto_ok:
                    dl_button(st, "Download PDF report", pdf, fname, "application/pdf", base="pdf_manual")

if st.session_state.get("last_pdf"):
    try:
        pdf_bytes = st.session_state["last_pdf"]
        fname = st.session_state.get("last_pdf_name","obis_report.pdf")
        key = _make_dl_key("last_pdf", fname)
        st.download_button("Download last generated PDF", data=pdf_bytes, file_name=fname, mime="application/pdf", key=key)
    except Exception as e:
        st.warning(f"PDF available but download failed: {e}")

# ----------- PROCESS SUBMISSION -----------
if submitted and query.strip():
    # If user asks generic question (not a name), answer via AI and return.
    if len(query.split()) >= 5 and not looks_binomial(query):
        with right:
            st.markdown("### AI Response")
            with st.spinner("AI thinking..."):
                reply = ask_openrouter([{"role":"system","content":"You are a marine biology data assistant. Answer clearly."},
                                        {"role":"user","content":query}])
            st.markdown(reply)
    else:
        with st.spinner("Resolving name..."):
            sci, prov = resolve_common_to_scientific(query)
        left.markdown(f"<div class='chip'>Resolved: <b>{sci}</b> <span class='muted'>(via {prov})</span></div>", unsafe_allow_html=True)
        add_to_history(query, sci, prov)
        if st.button(f"⭐ Bookmark {sci}", key=f"bm_{sci}"):
            bookmark_current(sci); st.success("Bookmarked!")

        df = pd.DataFrame()
        fetch_error: Optional[str] = None

        if start_date and end_date and start_date > end_date:
            fetch_error = "Start date must be on or before end date."
        else:
            bbox_params = None
            if st.session_state.get("bbox_enable", bbox_enable):
                try:
                    bbox_params = _sanitize_bbox({
                        "lonmin": lon_min,
                        "lonmax": lon_max,
                        "latmin": lat_min,
                        "latmax": lat_max,
                    })
                except ValueError as exc:
                    fetch_error = f"Bounding box error: {exc}"

            if fetch_error is None:
                with st.spinner("Fetching OBIS records..."):
                    df, fetch_error = fetch_obis_records(sci, size=max_records, bbox=bbox_params)

        if fetch_error:
            left.error(fetch_error)

        if not fetch_error and not df.empty and "eventDate" in df.columns:
            try:
                df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
                if start_date:
                    df = df[df["eventDate"] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df["eventDate"] <= pd.to_datetime(end_date)]
            except Exception:
                pass

        if df.empty:
            left.warning(f"No records found for '{sci}' with current filters.")
        else:
            keep = [c for c in ["scientificName","eventDate","decimalLongitude","decimalLatitude","depth","basisOfRecord","institutionCode"] if c in df.columns]
            df_clean = df[keep].copy() if keep else df.copy()
            save_df(df_clean, "obis")
            st.session_state["last_species"] = sci

            left.success(f"Found {len(df_clean)} records for '{sci}'")
            c1, c2, c3 = left.columns([1,1,2])
            with c1: left.markdown(f"<div class='stat'><b>{len(df_clean)}</b><div class='small muted'>records</div></div>", unsafe_allow_html=True)
            with c2:
                pts = df_clean.dropna(subset=["decimalLongitude","decimalLatitude"]).shape[0]
                left.markdown(f"<div class='stat'><b>{pts}</b><div class='small muted'>geo points</div></div>", unsafe_allow_html=True)
            with c3:
                rng = "-"
                if "eventDate" in df_clean.columns and df_clean["eventDate"].notna().any():
                    mn, mx = df_clean["eventDate"].min(), df_clean["eventDate"].max()
                    rng = f"{mn.date()} → {mx.date()}"
                left.markdown(f"<div class='small muted'>Date range: {rng}</div>", unsafe_allow_html=True)

            figs = make_plots_from_df(df_clean, sci)
            if "map" in figs: left.plotly_chart(figs["map"], use_container_width=True)
            row1 = left.columns(2)
            if "yearly" in figs: row1[0].plotly_chart(figs["yearly"], use_container_width=True)
            if "monthly" in figs: row1[1].plotly_chart(figs["monthly"], use_container_width=True)
            row2 = left.columns(2)
            if "depth_hist" in figs: row2[0].plotly_chart(figs["depth_hist"], use_container_width=True)
            if "density" in figs: row2[1].plotly_chart(figs["density"], use_container_width=True)

            left.markdown("#### Sample records")
            left.dataframe(df_clean.head(200))

            # Downloads
            try:
                csv_str = prepare_csv_download(df_clean)
                xlsx_bytes = prepare_excel_download(df_clean)
                sp_key = sci.replace(" ", "_").replace("/", "_")
                dl_button(left, "Download fetched CSV", csv_str, f"{sp_key}_obis.csv", "text/csv", base=f"fetched_{sp_key}")
                dl_button(left, "Download fetched Excel", xlsx_bytes, f"{sp_key}_obis.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base=f"fetched_{sp_key}")
            except Exception:
                csv_str = prepare_csv_download(df_clean)
                sp_key = sci.replace(" ", "_").replace("/", "_")
                dl_button(left, "Download fetched CSV", csv_str, f"{sp_key}_obis.csv", "text/csv", base=f"fetched_{sp_key}_fallback")

            if auto_summary:
                with right:
                    st.markdown("### AI Summary (auto)")
                    with st.spinner("Summarizing..."):
                        summary = ai_summarize_records(df_clean, sci)
                    st.session_state["last_summary"] = summary
                    st.markdown(summary)
                    # Auto-generate PDF
                    try:
                        figs_local = make_plots_from_df(df_clean, sci)
                        pdf_bytes = generate_pdf_report(df_clean, sci, summary, figs_local)
                        fname = f"{sci.replace(' ','_')}_auto_report.pdf"
                        st.session_state["last_pdf"] = pdf_bytes
                        st.session_state["last_pdf_name"] = fname
                        _auto_download_pdf_bytes(pdf_bytes, fname)
                    except Exception as e:
                        st.info(f"PDF auto-download skipped: {e}")

            with right:
                st.markdown("### Ask AI about this dataset")
                ai_q = st.text_area("Question about current dataset", key="data_ai_input_current", height=120)
                if st.button("Ask AI about data", key="ask_ai_current"):
                    sample = df_clean.head(30).to_dict(orient="records")
                    parts = [f"Sample (<=30 rows): {sample}"]
                    if st.session_state.get("last_summary"):
                        parts.append(f"Existing AI summary: {st.session_state['last_summary']}")
                    parts.append(f"Question: {ai_q or '[NO QUESTION]'}")
                    with st.spinner("AI analyzing..."):
                        ans = ask_openrouter([{"role":"system","content":"You are a helpful marine data assistant. Be concise."},
                                              {"role":"user","content":"\n\n".join(parts)}])
                    st.session_state.setdefault("data_ai_history", []).append({"q": ai_q, "a": ans, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")})
                    st.session_state["last_summary"] = ans
                    st.markdown("**AI answer:**"); st.markdown(ans)

# ----------- SYNTHETIC NETCDF ACTIONS -----------
if gen_enable:
    depths = [float(x.strip()) for x in nc_depths_str.split(",") if x.strip()]
    if st.button("Generate NetCDF and plots"):
        tmpf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp_path = tmpf.name; tmpf.close()
        ds = generate_ocean_netcdf(tmp_path,
                                   lon_min=locals().get("lon_min", 68.0), lon_max=locals().get("lon_max", 96.0),
                                   lat_min=locals().get("lat_min", 6.0),  lat_max=locals().get("lat_max", 24.0),
                                   nx=nc_nx, ny=nc_ny, nt=nc_nt,
                                   start_date=start_date, depths=np.array(depths), variables=nc_vars)
        st.success(f"NetCDF created: {tmp_path} (dims: {ds.dims})")
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        dl_button(st, "Download NetCDF", data, "floatchat_ocean.nc", "application/x-netcdf")
        f1 = plot_variable_map_from_ds(ds, var=nc_vars[0], time_index=0, depth_index=0); 
        if f1: st.plotly_chart(f1, use_container_width=True)
        f2 = plot_variable_profile_at_point(ds, var=nc_vars[0], time_index=0); 
        if f2: st.plotly_chart(f2, use_container_width=True)
        f3 = plot_variable_timeseries_at_point(ds, var=nc_vars[0], depth_index=0); 
        if f3: st.plotly_chart(f3, use_container_width=True)

# ----------- HISTORY & BOOKMARKS -----------
with st.expander("Saved searches & bookmarks"):
    hist = st.session_state.get("search_history", [])
    if hist:
        st.markdown("**Recent searches:**")
        for h in reversed(hist[-10:]):
            st.markdown(f"- `{h['q']}` → **{h['sci']}** <span class='small muted'>({h['prov']}, {h['t']})</span>", unsafe_allow_html=True)
    if st.session_state.get("bookmarks"):
        st.markdown("**Bookmarks:**")
        st.write(", ".join([f"`{b}`" for b in st.session_state["bookmarks"]]))

# ----------- FOOTER -----------
st.markdown("---")
st.markdown("<div class='small muted'>Data: OBIS (api.obis.org) • ERDDAP • Geocoding: Nominatim • LLM: OpenRouter</div>", unsafe_allow_html=True)
st.markdown("<div class='muted small'>Last updated: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "</div>", unsafe_allow_html=True)
