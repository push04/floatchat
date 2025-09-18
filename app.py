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

# add near other imports at top of file

import re

# --- NetCDF / xarray support (paste here) ---
import xarray as xr
import subprocess
import tempfile

from netCDF4 import Dataset
import os
# xarray uses numpy already imported in file


import streamlit.components.v1 as components


# -----------------------------
# Session helpers: persist dataframe as plain Python (records + columns)
# -----------------------------
# -----------------------------
# Session helpers: persist dataframe as plain Python (records + columns)
# -----------------------------
import numpy as np  # ensure numpy available for type checks


# ------------------ A: general region/bbox helpers (paste after imports) ------------------
import re
from typing import Optional, Dict, Tuple

def extract_bbox_from_text(text: str) -> Optional[Dict[str, float]]:
    """Parse bbox from free text. Accepts patterns like:
       'bbox=lonmin,lonmax,latmin,latmax' or 'bbox: lonmin, lonmax, latmin, latmax'."""
    if not text:
        return None
    m = re.search(r"bbox\s*[:=]\s*([-\d+.]+)\s*,\s*([-\d+.]+)\s*,\s*([-\d+.]+)\s*,\s*([-\d+.]+)", text, flags=re.I)
    if not m:
        return None
    try:
        lonmin, lonmax, latmin, latmax = map(float, m.groups())
        return {"lonmin": lonmin, "lonmax": lonmax, "latmin": latmin, "latmax": latmax}
    except Exception:
        return None

# Small set of region presets (you can add more). These are conservative boxes.
_REGION_PRESETS = {
    "sri lanka": {"lonmin": 79.5, "lonmax": 82.5, "latmin": 5.5, "latmax": 10.0},
    "indian ocean near srilanka": {"lonmin": 79.0, "lonmax": 83.0, "latmin": 4.5, "latmax": 10.5},
    "bay of bengal": {"lonmin": 80.0, "lonmax": 93.0, "latmin": 5.0, "latmax": 22.0},
    "arabian sea": {"lonmin": 50.0, "lonmax": 74.0, "latmin": 6.0, "latmax": 26.0},
}

def parse_region_to_bbox_general(text: str) -> Optional[Dict[str, float]]:
    """Try to match simple region names or keywords in the user's text to a preset bbox.
       If none matched, returns None."""
    if not text:
        return None
    t = text.lower()
    # direct preset match (substring)
    for key, box in _REGION_PRESETS.items():
        if key in t:
            return box
    # look for phrases like "near <placename>" (we won't geocode automatically here)
    # caller can optionally attempt geocoding from the place name (see geocode_place below)
    m = re.search(r"near\s+([A-Za-z0-9 \-,']{3,40})", text, flags=re.I)
    if m:
        place = m.group(1).strip()
        # common trivial variants:
        if "sri lanka" in place or "srilanka" in place:
            return _REGION_PRESETS.get("sri lanka")
        # unknown place -> return None (caller may try geocoding)
    return None

def geocode_place(place: str) -> Optional[Dict[str, float]]:
    """(Optional) Try to geocode a place name to a bbox using Nominatim.
       This performs a network request — keep it optional. If you don't want external calls remove/ignore this."""
    try:
        import requests
        if not place:
            return None
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        # lightweight user-agent
        headers = {"User-Agent": "floatchat/1.0 (use responsibly)"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        bb = data[0].get("boundingbox")  # [latmin, latmax, lonmin, lonmax]
        latmin = float(bb[0]); latmax = float(bb[1]); lonmin = float(bb[2]); lonmax = float(bb[3])
        return {"lonmin": lonmin, "lonmax": lonmax, "latmin": latmin, "latmax": latmax}
    except Exception:
        # network/geocode can fail; caller should handle None
        return None
# ------------------ END A ------------------

# ------------------ B: plotting helper (paste after A) ------------------
def plot_temp_salinity(df, container):
    """Draw temperature & salinity maps + histograms into the given Streamlit container.
       Detects likely column names and falls back gracefully."""
    try:
        import plotly.express as px
    except Exception:
        container.info("Plotly not installed — install plotly to see interactive charts.")
        return

    if df is None or df.empty:
        container.info("No data available to plot.")
        return

    # Attempt to normalize common coordinate column names
    if "longitude" in df.columns and "decimalLongitude" not in df.columns:
        df = df.rename(columns={"longitude": "decimalLongitude"})
    if "latitude" in df.columns and "decimalLatitude" not in df.columns:
        df = df.rename(columns={"latitude": "decimalLatitude"})

    # heuristics for temperature / salinity column names
    temp_cols = [c for c in df.columns if "temp" in c.lower() or c.lower() in ("temperature", "sst")]
    sal_cols = [c for c in df.columns if "salin" in c.lower() or c.lower() in ("salinity", "sss")]

    temp_col = temp_cols[0] if temp_cols else None
    sal_col = sal_cols[0] if sal_cols else None

    # coerce numeric
    if temp_col:
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    if sal_col:
        df[sal_col] = pd.to_numeric(df[sal_col], errors="coerce")

    has_coords = {"decimalLongitude", "decimalLatitude"}.issubset(set(df.columns))

    # Temperature
    if temp_col and df[temp_col].notna().any():
        container.markdown("### Temperature measurements")
        preview_cols = [c for c in ("occurrenceID", "decimalLongitude", "decimalLatitude", "eventDate", temp_col) if c in df.columns]
        if preview_cols:
            container.dataframe(df[preview_cols].head(200))
        if has_coords:
            p_df = df.dropna(subset=[temp_col, "decimalLongitude", "decimalLatitude"])
            if not p_df.empty:
                fig = px.scatter_mapbox(p_df, lon="decimalLongitude", lat="decimalLatitude", color=temp_col,
                                        hover_data=[temp_col, "eventDate"], title=f"Temperature ({temp_col})", height=450)
                fig.update_layout(mapbox_style="open-street-map")
                try:
                    _style_plotly_light(fig)
                except Exception:
                    pass
                container.plotly_chart(fig, use_container_width=True)
        # histogram
        try:
            container.plotly_chart(px.histogram(df[temp_col].dropna(), x=temp_col, nbins=40, title="Temperature distribution"), use_container_width=True)
        except Exception:
            pass
    else:
        container.info("No temperature measurements found in this dataset.")

    # Salinity
    if sal_col and df[sal_col].notna().any():
        container.markdown("### Salinity measurements")
        preview_cols = [c for c in ("occurrenceID", "decimalLongitude", "decimalLatitude", "eventDate", sal_col) if c in df.columns]
        if preview_cols:
            container.dataframe(df[preview_cols].head(200))
        if has_coords:
            p_df = df.dropna(subset=[sal_col, "decimalLongitude", "decimalLatitude"])
            if not p_df.empty:
                fig = px.scatter_mapbox(p_df, lon="decimalLongitude", lat="decimalLatitude", color=sal_col,
                                        hover_data=[sal_col, "eventDate"], title=f"Salinity ({sal_col})", height=450)
                fig.update_layout(mapbox_style="open-street-map")
                try:
                    _style_plotly_light(fig)
                except Exception:
                    pass
                container.plotly_chart(fig, use_container_width=True)
        try:
            container.plotly_chart(px.histogram(df[sal_col].dropna(), x=sal_col, nbins=40, title="Salinity distribution"), use_container_width=True)
        except Exception:
            pass
    else:
        container.info("No salinity measurements found in this dataset.")
# ------------------ END B ------------------



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

from typing import Optional

def load_obis_df() -> Optional[pd.DataFrame]:

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

        fig = px.histogram(
        series,
        x=series,
        nbins=30,
        title="Depth distribution",
        labels={"x": "Depth (m)", "count": "Frequency"},
        height=350,
    )
    _style_plotly_light(fig)
    return fig





def plot_occurrence_map(df):
    fig, ax = plt.subplots(figsize=(6,4))
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        ax.scatter(df["decimalLongitude"], df["decimalLatitude"], c="red", s=10, alpha=0.6)
        ax.set_title("Occurrence Locations")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    return fig


# --- NetCDF plotting helpers (paste after existing plot functions) ---
def plot_variable_map_from_ds(ds, var="temperature", time_index=0, depth_index=0):
    """Return a plotly figure: map (lat/lon) of chosen var at specified time & depth indices."""
    if var not in ds:
        return None
    da = ds[var].isel(time=time_index, depth=depth_index)
    df = da.to_dataframe(name=var).reset_index()
    # use px.scatter or px.density_mapbox; keep it simple with scatter
    fig = px.scatter(df, x="lon", y="lat", color=var, size_max=6,
                     title=f"{var} (time={ds.time.values[time_index]}, depth={float(ds.depth.values[depth_index])} m)")
    _style_plotly_light(fig)
    return fig

def plot_variable_profile_at_point(ds, var="temperature", lon_val=None, lat_val=None, time_index=0):
    """Return profile (var vs depth) at nearest grid point to lon_val/lat_val."""
    if var not in ds:
        return None
    if lon_val is None or lat_val is None:
        lon_val = float(ds.lon.mean())
        lat_val = float(ds.lat.mean())
    # find nearest indices
    lon_idx = int(np.abs(ds.lon.values - lon_val).argmin())
    lat_idx = int(np.abs(ds.lat.values - lat_val).argmin())
    da = ds[var].isel(time=time_index, lat=lat_idx, lon=lon_idx)
    prof = pd.DataFrame({"depth": ds.depth.values, var: da.values})
    fig = px.line(prof, x=var, y="depth", title=f"{var} profile at lon={lon_val:.2f}, lat={lat_val:.2f}")
    fig.update_yaxes(autorange="reversed")  # depth increasing downward
    _style_plotly_light(fig)
    return fig

def plot_variable_timeseries_at_point(ds, var="temperature", lon_val=None, lat_val=None, depth_index=0):
    """Timeseries of a var at a fixed point & depth."""
    if var not in ds:
        return None
    if lon_val is None or lat_val is None:
        lon_val = float(ds.lon.mean())
        lat_val = float(ds.lat.mean())
    lon_idx = int(np.abs(ds.lon.values - lon_val).argmin())
    lat_idx = int(np.abs(ds.lat.values - lat_val).argmin())
    da = ds[var].isel(depth=depth_index, lat=lat_idx, lon=lon_idx)
    ts = pd.DataFrame({"time": ds.time.values, var: da.values})
    fig = px.line(ts, x="time", y=var, title=f"{var} timeseries at lon={lon_val:.2f}, lat={lat_val:.2f}, depth={float(ds.depth.values[depth_index])}m")
    _style_plotly_light(fig)
    return fig

# ---------------- CMEMS: motuclient download helper ----------------
def fetch_cmems_via_motu(username: str, password: str, bbox: dict, start_date: str, end_date: str,
                         variables: list = ("thetao", "so"), product_id: str = "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh",
                         service_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS", motu_base: str = "https://nrt.cmems-du.eu/motu-web/Motu"):
    """
    Download a small CMEMS subset via motuclient CLI and return an xarray.Dataset.
    - username/password: your CMEMS credentials
    - bbox: dict with keys lonmin, lonmax, latmin, latmax
    - start_date, end_date: ISO strings "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
    - variables: list of variable names (e.g. ['thetao','so'])
    Returns xarray.Dataset or raises an exception.
    NOTE: requires motuclient installed (`pip install motuclient`) and a working CMEMS account.
    """
    try:
        import xarray as _xr
    except Exception as e:
        raise RuntimeError("xarray is required to load the downloaded CMEMS NetCDF: " + str(e))

    # defensive check: bbox keys
    try:
        lonmin = float(bbox["lonmin"]); lonmax = float(bbox["lonmax"])
        latmin = float(bbox["latmin"]); latmax = float(bbox["latmax"])
    except Exception:
        raise ValueError("bbox must be a dict containing numeric lonmin, lonmax, latmin, latmax")

    # prepare a temporary output file
    tmpf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmpf_name = tmpf.name
    tmpf.close()

    # Build motuclient command (CLI). Using --user/--pwd for auth.
    cmd = [
        sys.executable, "motuclient",
        "--motu", motu_base,
        "--service-id", service_id,
        "--product-id", product_id,
        "--longitude-min", str(lonmin),
        "--longitude-max", str(lonmax),
        "--latitude-min", str(latmin),
        "--latitude-max", str(latmax),
        "--date-min", str(start_date),
        "--date-max", str(end_date),
        "--out-dir", os.path.dirname(tmpf_name),
        "--out-name", os.path.basename(tmpf_name),
        "--user", username,
        "--pwd", password,
    ]

    # set a small depth window (user can change this later) -- optional, but many products require depth
    # Note: comment/uncomment or change the following two lines depending on the product needs
    # cmd += ["--depth-min", "0.0", "--depth-max", "0.5"]

    # add variable flags
    for v in variables:
        cmd.extend(["--variable", v])

    # run motuclient (this may take a few seconds)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    except FileNotFoundError:
        # motuclient not installed
        raise RuntimeError("motuclient is not installed in this environment. Install it with `pip install motuclient`")
    except subprocess.CalledProcessError as e:
        # capture stderr to help debugging
        raise RuntimeError(f"motuclient call failed: {e.stderr.decode('utf-8', errors='ignore') if getattr(e,'stderr',None) else str(e)}")
    except Exception as e:
        raise RuntimeError(f"motuclient call failed: {e}")

    # Load with xarray
    try:
        ds = _xr.open_dataset(tmpf_name, decode_times=True, use_cftime=False)
    except Exception as e:
        # try forcing engine='netcdf4' as fallback
        try:
            ds = _xr.open_dataset(tmpf_name, engine="netcdf4")
        except Exception as e2:
            raise RuntimeError(f"Failed to open downloaded CMEMS file: {e}; fallback also failed: {e2}")

    # Normalize coords if needed (many CMEMS products use 'longitude'/'latitude' or 'lon'/'lat')
    try:
        if "longitude" in ds.coords and "lon" not in ds.coords:
            ds = ds.rename({"longitude": "lon"})
        if "latitude" in ds.coords and "lat" not in ds.coords:
            ds = ds.rename({"latitude": "lat"})
    except Exception:
        # non-fatal
        pass

    return ds
# ---------------- end CMEMS helper ----------------




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

def _auto_download_pdf_bytes(pdf_bytes: bytes, filename: str):
    """Auto-trigger browser download of PDF bytes using a small JS snippet."""
    try:
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            pdf_bytes = pdf_bytes.getvalue()
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        # small HTML/JS that auto-clicks a hidden link
        html = f"""
        <html>
          <body>
            <a id="dl" href="data:application/pdf;base64,{b64}" download="{filename}"></a>
            <script>
              const a = document.getElementById('dl');
              if (a) {{
                a.click();
              }} else {{
                console.log('download link not found');
              }}
            </script>
          </body>
        </html>
        """
        components.html(html, height=0)
        return True
    except Exception as e:
        print("[WARN] auto-download failed:", e)
        return False


def _style_plotly_light(fig):
    """
    Make a plotly figure clearly visible on a dark page by forcing a light
    plot background and dark text/axes. Call this on every plotly figure
    before showing or exporting it.
    """
    try:
        if fig is None:
            return
        # use the clean white template so colors/lines are visible
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#06202a", size=11),
            legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#d1d5db", borderwidth=0.5)
        )
        # make axis grids subtle and readable
        fig.update_xaxes(showgrid=True, gridcolor="#e6eef6", zerolinecolor="#e6eef6",
                         tickcolor="#06202a", title_font=dict(color="#06202a"))
        fig.update_yaxes(showgrid=True, gridcolor="#e6eef6", zerolinecolor="#e6eef6",
                         tickcolor="#06202a", title_font=dict(color="#06202a"))
    except Exception:
        # be silent on styling errors so plots still render
        pass


def dl_button(container, label, data, file_name, mime, base="dl"):
    """
    Wrapper for container.download_button that guarantees a unique key.
    container can be st (top-level) or a column object (left_col, right_col).
    """
    key = _make_dl_key(base, file_name)
    return container.download_button(label, data=data, file_name=file_name, mime=mime, key=key)


# --- NetCDF dataset generator (paste just before CONFIG) ---
def generate_ocean_netcdf(outfile_path,
                          lon_min=68.0, lon_max=96.0,
                          lat_min=6.0, lat_max=24.0,
                          nx=40, ny=40,
                          nt=12, start_date=None,
                          depths=None, variables=None):
    """
    Create a synthetic but physically-plausible NetCDF with dims:
      time, depth, lat, lon
    Variables: temperature (C), salinity (PSU), optional others.
    This function uses xarray to build and save the dataset.
    """
    # defaults
    if start_date is None:
        start_date = pd.to_datetime(date.today()).normalize()
    if depths is None:
        depths = np.array([0, 10, 20, 50, 100, 200])  # m
    if variables is None:
        variables = ["temperature", "salinity"]

    # coords
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    times = pd.date_range(start=start_date, periods=nt, freq="MS")

    # build toy fields (you can replace with real profiles later)
    # shape: (time, depth, lat, lon)
    shape = (len(times), len(depths), len(lats), len(lons))
    # Base fields with simple depth & seasonal dependence
    base_temp = 15.0  # surface baseline
    temp = np.zeros(shape, dtype=np.float32)
    salt = np.zeros(shape, dtype=np.float32)
    for t_idx, t in enumerate(times):
        # seasonal cycle
        seasonal = 2.0 * np.sin(2 * np.pi * (t_idx / max(1, nt)))
        for d_idx, d in enumerate(depths):
            depth_decay = np.exp(-d / 50.0)  # shallower warmer
            # add spatial gradients
            lon_grad = (lons[np.newaxis, :] - lon_min) / (lon_max - lon_min)  # shape (1, nx)
            lat_grad = (lats[:, np.newaxis] - lat_min) / (lat_max - lat_min)  # shape (ny, 1)
            grid = (lat_grad * lon_grad).astype(np.float32)  # broadcasts to (ny, nx)

            temp[t_idx, d_idx, :, :] = base_temp + seasonal + 8.0 * depth_decay + 0.5 * grid
            salt[t_idx, d_idx, :, :] = 35.0 + 0.01 * d + 0.2 * grid  # simple salinity structure

    data_vars = {}
    if "temperature" in variables:
        data_vars["temperature"] = (("time","depth","lat","lon"), temp)
    if "salinity" in variables:
        data_vars["salinity"] = (("time","depth","lat","lon"), salt)

    coords = {
        "time": times,
        "depth": depths,
        "lat": lats,
        "lon": lons
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    # add basic metadata
    ds.attrs["title"] = "FloatChat generated ocean dataset"
    ds.attrs["created_by"] = "FloatChat"
    # save NetCDF
    ds.to_netcdf(outfile_path)
    return ds



# -----------------------------
# CONFIG (your provided key)
# -----------------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/sonoma-dusk-alpha"

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

    # Bounding box inputs
    bbox_enable = st.checkbox("Filter by bounding box (lon/lat)", value=False)
    lon_min = st.number_input("Lon min", value=68.0, step=0.1, format="%.3f")
    lon_max = st.number_input("Lon max", value=96.0, step=0.1, format="%.3f")
    lat_min = st.number_input("Lat min", value=6.0, step=0.1, format="%.3f")
    lat_max = st.number_input("Lat max", value=24.0, step=0.1, format="%.3f")

    # Only apply if enabled
    if not bbox_enable:
        lon_min = lon_max = lat_min = lat_max = None

    st.markdown("---")
    st.markdown("## Date range (optional)")
    start_date = st.date_input("Start date", value=date(2000, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    if start_date and end_date and start_date > end_date:
        st.warning("Start date is after end date — results will be empty until corrected.")

    # --- NetCDF generation controls (add into the sidebar block) ---
    st.markdown("---")
    st.markdown("## Generate NetCDF dataset (synthetic)")
    gen_enable = st.checkbox("Enable NetCDF generator", value=False)
    if gen_enable:
        nc_nx = st.number_input("Longitude points (nx)", min_value=8, max_value=400, value=40, step=8)
        nc_ny = st.number_input("Latitude points (ny)", min_value=8, max_value=400, value=40, step=8)
        nc_nt = st.number_input("Time steps (nt)", min_value=1, max_value=48, value=12)
        nc_depths_str = st.text_input("Depths (comma-separated, meters)", value="0,10,20,50,100,200")
        nc_vars = st.multiselect("Variables", ["temperature","salinity","oxygen","nitrate"], default=["temperature","salinity"])
    # ---------------- CMEMS UI (paste INSIDE the `with st.sidebar:` block, after the NetCDF generator controls) ---------------
    st.markdown("---")
    st.markdown("## Copernicus (CMEMS) fetch (optional)")
    enable_cmems = st.checkbox("Enable CMEMS fetch", value=False)
    if enable_cmems:
        cmems_user = st.text_input("CMEMS username", value="", help="Your Copernicus Marine Service username")
        cmems_pwd = st.text_input("CMEMS password", value="", type="password")
        cmems_vars = st.multiselect("Variables to fetch", options=["thetao", "so", "uo", "vo", "zos"], default=["thetao", "so"], help="thetao = temperature, so = salinity")
        cmems_date_min = st.date_input("CMEMS start date", value=date.today())
        cmems_date_max = st.date_input("CMEMS end date", value=date.today())
        st.markdown("Tip: choose a small bbox and short time-range for quick responses.")
    # ---------------- end sidebar UI ----------------

# ---------------- CMEMS action block (paste in main app, right after the sidebar block ends) ----------------
if 'enable_cmems' in locals() and enable_cmems:
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("### CMEMS fetch & visualisation")
        st.write("CMEMS variables:", ", ".join(cmems_vars))
        # determine bbox fallback: prefer explicit UI bbox if enabled, otherwise fallback to the app bbox defaults
        if bbox_enable:
            cm_bbox = {"lonmin": float(lon_min), "lonmax": float(lon_max), "latmin": float(lat_min), "latmax": float(lat_max)}
        else:
            # fallback: small Sri Lanka area (example) — user can change via sidebar bbox_enable
            cm_bbox = {"lonmin": 79.0, "lonmax": 82.0, "latmin": 5.0, "latmax": 10.0}

        if st.button("Fetch CMEMS data now"):
            # validate credentials
            if not cmems_user or not cmems_pwd:
                st.error("Please supply your CMEMS username and password in the sidebar.")
            else:
                # assemble date strings for motuclient (use full timestamp format)
                sd = f"{cmems_date_min.isoformat()} 00:00:00"
                ed = f"{cmems_date_max.isoformat()} 23:59:59"
                try:
                    with st.spinner("Requesting CMEMS subset (may take a few seconds)..."):
                        ds = fetch_cmems_via_motu(username=cmems_user, password=cmems_pwd, bbox=cm_bbox,
                                                  start_date=sd, end_date=ed, variables=cmems_vars)
                    st.success("CMEMS subset downloaded and loaded.")
                    # show dataset info
                    st.write(ds)
                    # standardize var names mapping: many CMEMS products use 'thetao' and 'so'
                    for var in cmems_vars:
                        try:
                            fig_map = plot_variable_map_from_ds(ds, var=var, time_index=0, depth_index=0)
                            if fig_map is not None:
                                st.plotly_chart(fig_map, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Failed to plot {var}: {e}")
                    # interactive profile/time series for a chosen point (nearest gridpoint)
                    try:
                        lon_choice = float((cm_bbox["lonmin"] + cm_bbox["lonmax"]) / 2.0)
                        lat_choice = float((cm_bbox["latmin"] + cm_bbox["latmax"]) / 2.0)
                        st.markdown("#### Timeseries & profile at bbox center")
                        for var in cmems_vars:
                            try:
                                fig_ts = plot_variable_timeseries_at_point(ds, var=var, lon_val=lon_choice, lat_val=lat_choice, depth_index=0)
                                if fig_ts is not None:
                                    st.plotly_chart(fig_ts, use_container_width=True)
                                fig_prof = plot_variable_profile_at_point(ds, var=var, lon_val=lon_choice, lat_val=lat_choice, time_index=0)
                                if fig_prof is not None:
                                    st.plotly_chart(fig_prof, use_container_width=True)
                            except Exception as e:
                                st.info(f"Could not generate profile/timeseries for {var}: {e}")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"CMEMS fetch failed: {e}")
# ---------------- end CMEMS action block ----------------


    
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
@cache_data(ttl=60 * 30)
def fetch_obis_records(species_name: str, size: int = 100, bbox: Optional[dict] = None):
    """ Fetch OBIS records and try to capture measurement/eMoF rows when present.
        Returns a DataFrame with occurrence rows; if measurements present, they will be
        flattened into repeated rows (one per measurement) in additional columns.
    """
    # Build bbox tuple (same as before)
    bbox_tuple = None
    if isinstance(bbox, dict):
        try:
            bbox_tuple = (float(bbox.get("lonmin")), float(bbox.get("lonmax")),
                          float(bbox.get("latmin")), float(bbox.get("latmax")))
        except Exception:
            bbox_tuple = None
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_tuple = tuple(bbox)

    params = {"scientificname": species_name, "size": size}
    if bbox_tuple:
        lonmin, lonmax, latmin, latmax = bbox_tuple
        # OBIS accepts WKT polygon for geometry
        poly = f"POLYGON(({lonmin} {latmin}, {lonmax} {latmin}, {lonmax} {latmax}, {lonmin} {latmax}, {lonmin} {latmin}))"
        params["geometry"] = poly

    r = requests.get(OBIS_API_URL, params=params, timeout=40)
    r.raise_for_status()
    js = r.json()
    results = js.get("results", []) or []

    # If results are empty, return empty DataFrame
    if not results:
        return pd.DataFrame()

    # Flatten occurrences to DataFrame first
    occ_df = pd.DataFrame(results)

    # Look for possible measurement/eMoF fields — could be in an 'extendedMeasurements' / 'measurements'
    # or maybe nested under a key like 'extendedMeasurementOrFact' depending on dataset.
    # We'll defensively detect common names and, if present, expand into a measurements DataFrame and join.
    measurement_rows = []
    for rec in results:
        occ_id = rec.get("occurrenceID") or rec.get("id") or rec.get("occurrence_id")
        # try common nested keys
        nested = rec.get("extendedMeasurementOrFact") or rec.get("measurements") or rec.get("measurement")
        if nested and isinstance(nested, list):
            for m in nested:
                m_rec = {
                    "occurrenceID": occ_id,
                    "measurementType": m.get("measurementType") or m.get("measurement_type"),
                    "measurementValue": m.get("measurementValue") or m.get("measurement_value") or m.get("measurement"),
                    "measurementUnit": m.get("measurementUnit") or m.get("measurement_unit"),
                }
                measurement_rows.append(m_rec)

    if measurement_rows:
        meas_df = pd.DataFrame(measurement_rows)
        # Pivot common measurement types into columns (temperature, salinity, etc.)
        # Normalize measurementType text for matching common names
        def _norm(s):
            return (s or "").strip().lower()
        meas_df["mtype_norm"] = meas_df["measurementType"].apply(_norm)

        # pick common measurement types (add more as needed)
        rename_map = {}
        for key in ["temperature", "sea temperature", "sea_surface_temperature", "salinity", "sst", "ssalinity"]:
            # unify a few variants to simple column names
            norm = key.replace(" ", "_")
            rename_map[key] = norm

        # pivot: for each occurrenceID, keep first measurementValue for types we care about
        pivot = {}
        for idx, row in meas_df.iterrows():
            occ = row["occurrenceID"]
            mt = row["mtype_norm"]
            val = row["measurementValue"]
            if occ is None or mt is None or val is None:
                continue
            # naive mapping: if measurement type contains 'temp' -> 'temperature', 'salin' -> 'salinity'
            if "temp" in mt:
                pivot.setdefault(occ, {})["temperature"] = val
            elif "salin" in mt:
                pivot.setdefault(occ, {})["salinity"] = val
            else:
                # store other types with the raw name (optional)
                pivot.setdefault(occ, {})[mt] = val

        # convert pivot to DataFrame and merge back to occ_df on occurrenceID
        pivot_rows = []
        for occ, d in pivot.items():
            r = {"occurrenceID": occ}
            r.update(d)
            pivot_rows.append(r)
        if pivot_rows:
            pivot_df = pd.DataFrame(pivot_rows)
            # ensure occ_df has a matching occurrenceID column; try common names
            if "occurrenceID" not in occ_df.columns:
                # try to find a unique id column
                if "id" in occ_df.columns:
                    occ_df = occ_df.rename(columns={"id": "occurrenceID"})
                elif "occurrence_id" in occ_df.columns:
                    occ_df = occ_df.rename(columns={"occurrence_id": "occurrenceID"})
            # merge (left)
            occ_df = occ_df.merge(pivot_df, on="occurrenceID", how="left")

    return occ_df




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
    """Return a dict of plotly figures (map, timeseries, depth hist, density)
    and additional measurement plots (temperature, salinity) if those fields exist.
    This preserves all pre-existing plots and adds measurement plots (no other code changed).
    """
    figs = {}

    # ----- Map scatter (same logic as before) -----
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        map_df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
        sample_map = map_df if len(map_df) <= 1000 else map_df.sample(1000, random_state=1)
        figs["map"] = px.scatter_geo(
            sample_map,
            lon="decimalLongitude",
            lat="decimalLatitude",
            hover_name="scientificName" if "scientificName" in sample_map.columns else None,
            hover_data=[c for c in ["eventDate", "depth"] if c in sample_map.columns],
            title=f"{species_name} occurrences (sample)",
            projection="natural earth",
            height=550,
        )
        figs["map"].update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"))
        _style_plotly_light(figs["map"])

    # ----- Time series (yearly / monthly counts) -----
    if "eventDate" in df.columns:
        try:
            df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
            times = df.dropna(subset=["eventDate_parsed"]).copy()
            if not times.empty:
                times["year"] = times["eventDate_parsed"].dt.year
                times["month"] = times["eventDate_parsed"].dt.to_period("M").astype(str)
                yearly = times.groupby("year").size().reset_index(name="count")
                figs["yearly"] = px.bar(yearly, x="year", y="count", title="Records per year", height=300)
                _style_plotly_light(figs["yearly"])
                monthly = times.groupby("month").size().reset_index(name="count").sort_values("month")
                figs["monthly"] = px.line(monthly, x="month", y="count", title="Records per month (period)", height=300)
                _style_plotly_light(figs["monthly"])
        except Exception:
            # keep original behavior: don't break on time parsing errors
            pass

    # ----- Depth distribution (same logic) -----
    if "depth" in df.columns:
        try:
            df_depth = df.dropna(subset=["depth"]).copy()
            if not df_depth.empty:
                depth_nums = pd.to_numeric(df_depth["depth"], errors="coerce").dropna()
                if not depth_nums.empty:
                    figs["depth_hist"] = px.histogram(depth_nums, x=depth_nums, nbins=30,
                                                     title="Depth distribution", labels={"x": "Depth (m)", "count": "Frequency"}, height=300)
                    _style_plotly_light(figs["depth_hist"])
        except Exception as e:
            print("[WARN] depth plot failed:", e)

    # ----- Density heatmap (lon/lat) (same logic) -----
    if "decimalLongitude" in df.columns and "decimalLatitude" in df.columns:
        try:
            heat_df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
            if len(heat_df) >= 20:
                figs["density"] = px.density_heatmap(
                    heat_df, x="decimalLongitude", y="decimalLatitude", nbinsx=60, nbinsy=40,
                    title="Density heatmap (lon/lat)", height=400
                )
                figs["density"].update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        except Exception:
            pass

    # ----- NEW: measurement detection helpers -----
    # Common column name variants we try to support
    TEMP_COLS = [
        "temperature", "temp", "sea_temperature", "sea_temp", "water_temperature", "water_temp", "t"
    ]
    SAL_COLS = [
        "salinity", "psal", "psal_ctd", "salt"
    ]
    # Generic function to find first matching column in list
    def _find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # Also try case-insensitive match for robustness
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    temp_col = _find_col(TEMP_COLS)
    sal_col = _find_col(SAL_COLS)

    # ----- NEW: Temperature plots (timeseries + distribution + summary) -----
    if temp_col is not None:
        try:
            series_temp = pd.to_numeric(df[temp_col].dropna(), errors="coerce").dropna()
            if not series_temp.empty:
                # Distribution / histogram
                figs["temperature_hist"] = px.histogram(series_temp, x=series_temp, nbins=40,
                                                        title=f"Temperature distribution ({temp_col})", labels={"x": "Temperature (°C)", "count": "Frequency"}, height=300)
                _style_plotly_light(figs["temperature_hist"])

                # If eventDate exists and is parseable, create a timeseries
                if "eventDate_parsed" not in df.columns and "eventDate" in df.columns:
                    try:
                        df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
                    except Exception:
                        pass
                if "eventDate_parsed" in df.columns and df["eventDate_parsed"].notna().any():
                    t_ts = df.dropna(subset=[temp_col, "eventDate_parsed"]).copy()
                    t_ts[temp_col] = pd.to_numeric(t_ts[temp_col], errors="coerce")
                    t_ts = t_ts.dropna(subset=[temp_col])
                    if not t_ts.empty:
                        figs["temperature_ts"] = px.line(t_ts.sort_values("eventDate_parsed"),
                                                         x="eventDate_parsed", y=temp_col,
                                                         title=f"Temperature time series ({temp_col})", height=300)
                        _style_plotly_light(figs["temperature_ts"])

                # Summary stats (kept in figs as a small dict for optional display)
                temp_stats = {
                    "count": int(series_temp.count()),
                    "mean": float(series_temp.mean()),
                    "median": float(series_temp.median()),
                    "min": float(series_temp.min()),
                    "max": float(series_temp.max()),
                    "col_name": temp_col
                }
                figs["temperature_summary"] = temp_stats
        except Exception as e:
            print("[WARN] temperature plotting failed:", e)

    # ----- NEW: Salinity plots (timeseries + distribution + summary) -----
    if sal_col is not None:
        try:
            series_sal = pd.to_numeric(df[sal_col].dropna(), errors="coerce").dropna()
            if not series_sal.empty:
                figs["salinity_hist"] = px.histogram(series_sal, x=series_sal, nbins=40,
                                                     title=f"Salinity distribution ({sal_col})", labels={"x": "Salinity (PSU)", "count": "Frequency"}, height=300)
                _style_plotly_light(figs["salinity_hist"])

                # timeseries if dates available
                if "eventDate_parsed" not in df.columns and "eventDate" in df.columns:
                    try:
                        df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
                    except Exception:
                        pass
                if "eventDate_parsed" in df.columns and df["eventDate_parsed"].notna().any():
                    s_ts = df.dropna(subset=[sal_col, "eventDate_parsed"]).copy()
                    s_ts[sal_col] = pd.to_numeric(s_ts[sal_col], errors="coerce")
                    s_ts = s_ts.dropna(subset=[sal_col])
                    if not s_ts.empty:
                        figs["salinity_ts"] = px.line(s_ts.sort_values("eventDate_parsed"),
                                                      x="eventDate_parsed", y=sal_col,
                                                      title=f"Salinity time series ({sal_col})", height=300)
                        _style_plotly_light(figs["salinity_ts"])

                sal_stats = {
                    "count": int(series_sal.count()),
                    "mean": float(series_sal.mean()),
                    "median": float(series_sal.median()),
                    "min": float(series_sal.min()),
                    "max": float(series_sal.max()),
                    "col_name": sal_col
                }
                figs["salinity_summary"] = sal_stats
        except Exception as e:
            print("[WARN] salinity plotting failed:", e)

    # ----- Return all figs (existing keys preserved; new keys added) -----
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
                fig = px.scatter_geo(
                map_df,
                lon="decimalLongitude",
                lat="decimalLatitude",
                hover_name="scientificName",
                hover_data=["eventDate","depth"] if "eventDate" in df_prev.columns else None,
                projection="natural earth",
                height=420,
            )
            # keep ocean color but enforce light styling so markers/axes are visible
            fig.update_layout(geo=dict(showcountries=True, oceancolor="rgb(3,29,44)"))
            _style_plotly_light(fig)
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




# --- START: enhanced handler (drop this block in place of your old `if submit and user_input.strip():`) ---
if submit and user_input.strip():
    # local imports to be defensive (harmless if already imported)
    import json
    import re
    import requests
    import pandas as pd
    from datetime import datetime

    # call LLM and normalize response
    with st.spinner("Routing your input via the LLM..."):
        decision = interpret_input_via_ai(user_input)

    # if LLM returned JSON text, parse it
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except Exception:
            # leave it as-is (will be handled below)
            pass

    # ensure decision is a dict before using .get
    if not isinstance(decision, dict):
        st.warning("AI returned an unexpected response (not a JSON object). Try rephrasing or try again.")
    else:
        # --- If AI asks to search OBIS for a species ---
        if decision.get("action") == "search" and decision.get("species"):
            chosen_species = decision["species"]
            left_col.markdown(f"### Searching OBIS for species: **{chosen_species}**")

            loader = left_col.empty()
            loader.markdown("<div class='muted'><span class='pulse'></span> Fetching records…</div>", unsafe_allow_html=True)

            try:
                # prepare bbox if enabled (defensive casting)
                bbox = None
                try:
                    # explicit bbox in user input
                    bbox = extract_bbox_from_text(user_input)
                except Exception:
                    bbox = None

                if bbox is None:
                    # region presets from free text (keyword-based)
                    try:
                        bbox = parse_region_to_bbox_general(user_input)
                    except Exception:
                        bbox = None

                # If still None, try to geocode a 'near <place>' phrase (best-effort; requires network & may be slow)
                if bbox is None:
                    try:
                        m = re.search(r"near\s+([A-Za-z0-9 \-\,']{3,60})", user_input, flags=re.I)
                        if m:
                            place = m.group(1).strip()
                            geoc = geocode_place(place)  # returns bbox or None
                            if geoc:
                                bbox = geoc
                    except Exception:
                        bbox = None

                # final fallback: the UI bbox controls (unchanged behavior)
                if bbox is None and bbox_enable:
                    try:
                        bbox = {"lonmin": float(lon_min), "lonmax": float(lon_max), "latmin": float(lat_min), "latmax": float(lat_max)}
                    except Exception:
                        bbox = None

                # fetch occurrences from OBIS using your helper
                df = fetch_obis_records(chosen_species, size=max_records, bbox=bbox)

                # normalize possible coordinate column names so plotting helper finds them
                if isinstance(df, (list, dict)):
                    try:
                        import pandas as _pd
                        df = _pd.DataFrame(df)
                    except Exception:
                        pass

                if hasattr(df, "rename"):
                    try:
                        df = df.rename(columns={"longitude": "decimalLongitude", "latitude": "decimalLatitude"})
                    except Exception:
                        pass

                # attempt temperature/salinity plotting (non-fatal)
                try:
                    plot_temp_salinity(df, left_col)
                except Exception:
                    left_col.info("Environmental plotting (temperature/salinity) failed or no data available.")
                # ------------------ END C ------------------

                # fetch occurrences via your helper (may return DataFrame or None)
                df = fetch_obis_records(chosen_species, size=max_records, bbox=bbox)

                # coerce to DataFrame if needed
                if df is None:
                    df = pd.DataFrame()
                elif not isinstance(df, pd.DataFrame):
                    try:
                        df = pd.DataFrame(df)
                    except Exception:
                        df = pd.DataFrame()

                # --- Extra: fetch raw OBIS JSON to look for eMoF / extended measurements ---
                response_json = {"results": []}
                try:
                    params = {"scientificname": chosen_species, "size": int(max_records)}
                    if bbox:
                        lonmin, lonmax = float(bbox["lonmin"]), float(bbox["lonmax"])
                        latmin, latmax = float(bbox["latmin"]), float(bbox["latmax"])
                        poly = f"POLYGON(({lonmin} {latmin}, {lonmax} {latmin}, {lonmax} {latmax}, {lonmin} {latmax}, {lonmin} {latmin}))"
                        params["geometry"] = poly
                    r_meas = requests.get(OBIS_API_URL, params=params, timeout=40)
                    r_meas.raise_for_status()
                    response_json = r_meas.json() or {"results": []}
                except Exception as e:
                    # proceed but warn user
                    left_col.warning(f"Could not fetch raw OBIS JSON for measurements: {e}")

                # build measurement rows defensively
                meas_rows = []
                for rec in response_json.get("results", []):
                    occ_id = rec.get("occurrenceID") or rec.get("id") or rec.get("occurrence_id") or None
                    lon = rec.get("decimalLongitude") or rec.get("longitude") or None
                    lat = rec.get("decimalLatitude") or rec.get("latitude") or None
                    event_date = rec.get("eventDate") or rec.get("year") or None

                    # candidate keys where measurements/eMoF can live
                    candidates = []
                    for k in ("extendedMeasurementOrFact", "extendedmeasurementorfact", "measurements", "measurement", "measurementOrFact", "emof", "extensions"):
                        if k in rec and rec[k]:
                            candidates.append((k, rec[k]))

                    # also inspect extensions mapping (some providers store eMoF under extensions)
                    exts = rec.get("extensions") or {}
                    if isinstance(exts, dict):
                        for ext_name, ext_rows in exts.items():
                            if ext_rows:
                                candidates.append((f"extensions.{ext_name}", ext_rows))

                    for keyname, entries in candidates:
                        entries_list = entries if isinstance(entries, list) else [entries]
                        for m in entries_list:
                            if not isinstance(m, dict):
                                continue
                            mtype = (m.get("measurementType") or m.get("measurementTypeID") or m.get("type")
                                     or m.get("name") or m.get("measurement_type") or "")
                            mval = m.get("measurementValue") or m.get("measurement_value") or m.get("value") or None
                            munit = (m.get("measurementUnit") or m.get("measurement_unit") or m.get("unit") or "")
                            # try to coerce string numbers to float
                            if isinstance(mval, str):
                                try:
                                    found = re.findall(r"[-+]?\d*\.\d+|\d+", mval)
                                    if found:
                                        mval = float(found[0])
                                except Exception:
                                    pass
                            meas_rows.append({
                                "occurrenceID": occ_id,
                                "lon": lon,
                                "lat": lat,
                                "time": event_date,
                                "measurementType": str(mtype),
                                "value": mval,
                                "unit": str(munit)
                            })

                df_meas = pd.DataFrame(meas_rows)

                # show measurement-derived plots if any
                if not df_meas.empty:
                    try:
                        import plotly.express as px
                        df_meas["mtype_norm"] = df_meas["measurementType"].fillna("").astype(str).str.lower()
                        df_meas["unit_norm"] = df_meas["unit"].fillna("").astype(str).str.lower()

                        temp_mask = df_meas["mtype_norm"].str.contains("temp", na=False) | df_meas["unit_norm"].str.contains("c|°c", na=False)
                        sal_mask = df_meas["mtype_norm"].str.contains("salin", na=False) | df_meas["unit_norm"].str.contains("psu|ppt", na=False)

                        df_temp = df_meas[temp_mask & df_meas["value"].notna()].copy()
                        df_sal = df_meas[sal_mask & df_meas["value"].notna()].copy()

                        if not df_temp.empty:
                            left_col.markdown("### Temperature measurements (sample from OBIS eMoF)")
                            left_col.dataframe(df_temp.head(200))
                            fig_t = px.scatter_mapbox(
                                df_temp, lon="lon", lat="lat", color="value",
                                hover_data=["occurrenceID", "time", "unit", "measurementType"],
                                title="OBIS: temperature measurements", height=450
                            )
                            fig_t.update_layout(mapbox_style="open-street-map")
                            try:
                                _style_plotly_light(fig_t)
                            except Exception:
                                pass
                            left_col.plotly_chart(fig_t, use_container_width=True)
                            left_col.plotly_chart(px.histogram(df_temp, x="value", nbins=40, title="Temperature distribution (OBIS)"), use_container_width=True)

                        if not df_sal.empty:
                            left_col.markdown("### Salinity measurements (sample from OBIS eMoF)")
                            left_col.dataframe(df_sal.head(200))
                            fig_s = px.scatter_mapbox(
                                df_sal, lon="lon", lat="lat", color="value",
                                hover_data=["occurrenceID", "time", "unit", "measurementType"],
                                title="OBIS: salinity measurements", height=450
                            )
                            fig_s.update_layout(mapbox_style="open-street-map")
                            try:
                                _style_plotly_light(fig_s)
                            except Exception:
                                pass
                            left_col.plotly_chart(fig_s, use_container_width=True)
                            left_col.plotly_chart(px.histogram(df_sal, x="value", nbins=40, title="Salinity distribution (OBIS)"), use_container_width=True)
                    except Exception:
                        # plotting is optional — if plotly not installed or fails, show measurements table
                        left_col.info("Measurement rows were found but plotting failed. Showing table preview.")
                        left_col.dataframe(df_meas.head(200))
                else:
                    left_col.info("No eMoF/measurement rows found in the returned OBIS occurrences for this query.")

                # --- filter by date range if available (defensive) ---
                try:
                    if not df.empty and "eventDate" in df.columns:
                        df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
                        if start_date:
                            df = df[df["eventDate"] >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df["eventDate"] <= pd.to_datetime(end_date)]
                except Exception:
                    # non-fatal; continue with unfiltered data
                    pass

                loader.empty()

                # handle empty dataset
                if df.empty:
                    left_col.warning(f"No records found for species: {chosen_species} (after date/filters).")
                else:
                    # select a small set of useful columns for display and caching
                    keep_cols = [c for c in ["scientificName", "eventDate", "decimalLongitude", "decimalLatitude", "depth", "basisOfRecord", "institutionCode"] if c in df.columns]
                    df_clean = df[keep_cols].copy() if keep_cols else df.copy()
                    # coerce eventDate for summary
                    if "eventDate" in df_clean.columns:
                        try:
                            df_clean["eventDate"] = pd.to_datetime(df_clean["eventDate"], errors="coerce")
                        except Exception:
                            pass

                    # persist for later UI / AI steps
                    try:
                        save_obis_df(df_clean)
                    except Exception:
                        # best-effort; don't crash if save helper missing
                        pass
                    st.session_state["last_species"] = chosen_species

                    left_col.success(f"Found {len(df_clean)} records for '{chosen_species}'")

                    # summary stats
                    c1, c2, c3 = left_col.columns([1, 1, 2])
                    with c1:
                        c1.markdown(f"<div class='stat'><strong>{len(df_clean)}</strong><div class='small muted'>records</div></div>", unsafe_allow_html=True)
                    with c2:
                        unique_locs = df_clean.dropna(subset=["decimalLongitude", "decimalLatitude"]).shape[0] if {"decimalLongitude", "decimalLatitude"}.issubset(df_clean.columns) else 0
                        c2.markdown(f"<div class='stat'><strong>{unique_locs}</strong><div class='small muted'>geo points</div></div>", unsafe_allow_html=True)
                    with c3:
                        range_time = "-"
                        if "eventDate" in df_clean.columns and df_clean["eventDate"].notna().any():
                            mn = df_clean["eventDate"].min(); mx = df_clean["eventDate"].max()
                            range_time = f"{mn.date()} → {mx.date()}"
                        c3.markdown(f"<div class='small muted'>Date range: {range_time}</div>", unsafe_allow_html=True)

                    # plots from helper
                    try:
                        figs = make_plots_from_df(df_clean, chosen_species)
                        if isinstance(figs, dict):
                            if "map" in figs:
                                left_col.plotly_chart(figs["map"], use_container_width=True)
                            # 2-column layout for other figs
                            pair = left_col.columns(2)
                            if "yearly" in figs:
                                pair[0].plotly_chart(figs["yearly"], use_container_width=True)
                            if "monthly" in figs:
                                pair[1].plotly_chart(figs["monthly"], use_container_width=True)
                            pair2 = left_col.columns(2)
                            if "depth_hist" in figs:
                                pair2[0].plotly_chart(figs["depth_hist"], use_container_width=True)
                            if "density" in figs:
                                pair2[1].plotly_chart(figs["density"], use_container_width=True)
                    except Exception:
                        left_col.info("Plot generation failed — showing table preview.")
                        left_col.dataframe(df_clean.head(200))

                    # sample records display
                    left_col.markdown("#### Sample records")
                    left_col.dataframe(df_clean.head(200))

                    # download buttons (CSV + Excel where possible)
                    try:
                        csv_str = prepare_csv_download(df_clean)
                        species_key = chosen_species.replace(" ", "_").replace("/", "_")
                        dl_button(left_col, "Download fetched records (CSV)", data=csv_str, file_name=f"{species_key}_obis.csv", mime="text/csv", base=f"fetched_{species_key}")
                        try:
                            xlsx_bytes = prepare_excel_download(df_clean)
                            dl_button(left_col, "Download fetched records (Excel)", data=xlsx_bytes, file_name=f"{species_key}_obis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", base=f"fetched_{species_key}_xlsx")
                        except Exception:
                            # Excel optional
                            pass
                    except Exception:
                        left_col.info("Could not prepare downloads for this dataset.")

                    # Optional auto summary
                    if auto_summary:
                        right_col.markdown("### AI Summary (auto)")
                        loading_slot = right_col.empty()
                        loading_slot.markdown("<div class='muted'><span class='pulse'></span> AI summarizing the fetched records...</div>", unsafe_allow_html=True)
                        try:
                            summary = ai_summarize_records(df_clean, chosen_species)
                            st.session_state["last_summary"] = summary
                            right_col.markdown(summary)
                            # attempt to auto-generate PDF bytes (best-effort)
                            try:
                                df_for_pdf = load_obis_df()
                                if df_for_pdf is not None and not df_for_pdf.empty:
                                    figs_local = make_plots_from_df(df_for_pdf, st.session_state.get("last_species", chosen_species))
                                    pdf_bytes_auto = generate_pdf_report(df_for_pdf, st.session_state.get("last_species", chosen_species), summary, figs_local)
                                    if isinstance(pdf_bytes_auto, (bytes, bytearray)) and len(pdf_bytes_auto) > 0:
                                        fname_auto = f"{(st.session_state.get('last_species') or chosen_species).replace(' ','_')}_auto_report.pdf"
                                        st.session_state["last_pdf"] = pdf_bytes_auto
                                        st.session_state["last_pdf_name"] = fname_auto
                                        # try JS auto-download, don't break UI on failure
                                        try:
                                            _auto_download_pdf_bytes(pdf_bytes_auto, fname_auto)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        except Exception as e:
                            right_col.error(f"Auto-summary failed: {e}")
                        finally:
                            loading_slot.empty()

                    # DATASET-AWARE AI CHAT UI (keeps small sample + summary)
                    right_col.markdown("### Ask AI about this dataset")
                    ai_data_query = right_col.text_area(
                        "Ask a question about the current dataset (the AI will see a small sample and previous summary):",
                        key="data_ai_input",
                        height=120
                    )
                    if right_col.button("Ask AI about data", key="ask_data_ai"):
                        ai_data_query = st.session_state.get("data_ai_input", "").strip()
                        st.session_state["last_data_ai_query"] = ai_data_query
                        # prepare sample for AI
                        sample_for_ai = None
                        try:
                            df_for_sample = load_obis_df()
                            if df_for_sample is not None and not df_for_sample.empty:
                                sample_for_ai = df_for_sample.head(30).to_dict(orient="records")
                        except Exception:
                            sample_for_ai = None

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

                        loading_slot = right_col.empty()
                        loading_slot.markdown("<div class='muted'><span class='pulse'></span> AI analyzing dataset...</div>", unsafe_allow_html=True)
                        try:
                            ai_reply = ask_openrouter([system_msg, user_msg])
                            entry = {"question": ai_data_query, "answer": ai_reply, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
                            st.session_state.setdefault("data_ai_history", []).append(entry)
                            st.session_state["last_data_ai_reply"] = ai_reply
                            st.session_state["last_summary"] = ai_reply
                            right_col.markdown("**AI answer:**")
                            right_col.markdown(ai_reply)
                        except Exception as e:
                            right_col.error(f"AI request failed: {e}")
                        finally:
                            loading_slot.empty()

                    # recent Q&A
                    if st.session_state.get("data_ai_history"):
                        right_col.markdown("#### Recent dataset queries")
                        for item in reversed(st.session_state["data_ai_history"][-6:]):
                            with right_col.expander(f"Q: {item['question'][:60] or '(no question)'} — {item['time']}", expanded=False):
                                right_col.markdown(f"**Q:** {item['question']}")
                                right_col.markdown(f"**A:** {item['answer']}")

                    # Export / PDF button (single, consolidated flow)
                    right_col.markdown("### Export")
                    if right_col.button("Generate PDF report (maps + summary)"):
                        right_col.info("Generating PDF... this may take a few seconds.")
                        try:
                            df_for_pdf = load_obis_df()
                            if df_for_pdf is None or df_for_pdf.empty:
                                right_col.error("No cached dataset available for PDF generation. Fetch a species first.")
                            else:
                                summary_text = st.session_state.get("last_summary", "")
                                figs_local = make_plots_from_df(df_for_pdf, st.session_state.get("last_species", ""))
                                pdf_loading = right_col.empty()
                                pdf_loading.markdown("<div class='muted'><span class='pulse'></span> Rendering PDF...</div>", unsafe_allow_html=True)
                                try:
                                    pdf_bytes = generate_pdf_report(df_for_pdf, st.session_state.get("last_species", ""), summary_text, figs_local)
                                    if not isinstance(pdf_bytes, (bytes, bytearray)):
                                        try:
                                            pdf_bytes = pdf_bytes.getvalue()
                                        except Exception as e:
                                            raise RuntimeError(f"PDF generator returned non-bytes and could not be converted: {e}")
                                    fname = f"{(st.session_state.get('last_species') or 'obis_report').replace(' ','_')}_report.pdf"
                                    st.session_state["last_pdf"] = pdf_bytes
                                    st.session_state["last_pdf_name"] = fname
                                    st.session_state["last_pdf_species"] = st.session_state.get("last_species")
                                    # try auto-download; fallback to manual
                                    auto_ok = False
                                    try:
                                        auto_ok = bool(_auto_download_pdf_bytes(pdf_bytes, fname))
                                    except Exception:
                                        auto_ok = False
                                    if not auto_ok:
                                        try:
                                            dl_button(right_col, "Download PDF report", data=pdf_bytes, file_name=fname, mime="application/pdf", base=f"pdf_{fname}")
                                        except Exception as e:
                                            right_col.error(f"Could not create download button: {e}")
                                    right_col.success("PDF generated.")
                                    right_col.markdown(f"**PDF size:** {len(pdf_bytes):,} bytes")
                                except Exception as e:
                                    right_col.error(f"PDF creation failed: {e}")
                                    right_col.info("Common cause: missing kaleido or reportlab in the environment. Install them and restart Streamlit.")
                                finally:
                                    pdf_loading.empty()
                        except Exception as e:
                            right_col.error(f"Failed to create PDF: {e}")

            except Exception as e:
                # outer fetch + processing error
                try:
                    loader.empty()
                except Exception:
                    pass
                left_col.error(f"Failed to fetch/process OBIS data: {e}")

        else:
            # LLM decided to answer directly (not a 'search' action)
            ai_query = decision.get("query") or user_input
            right_col.markdown("### AI Response")
            right_col.markdown("<div class='muted'>AI interpreted your input as a question — here's the answer.</div>", unsafe_allow_html=True)
            system_msg = {"role": "system", "content": "You are a marine biology data assistant. Answer clearly and concisely."}
            user_msg = {"role": "user", "content": f"User input: {ai_query}"}
            try:
                with st.spinner("AI thinking..."):
                    reply = ask_openrouter([system_msg, user_msg])
                right_col.markdown(reply)
            except Exception as e:
                right_col.error(f"AI request failed: {e}")
# --- END enhanced handler ---

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

# --- Trigger NetCDF creation and show plots (paste near data display area) ---
if gen_enable:
    depths = [float(x.strip()) for x in nc_depths_str.split(",") if x.strip()]
    if st.button("Generate NetCDF and plots"):
        tmpf = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp_path = tmpf.name
        tmpf.close()
        ds = generate_ocean_netcdf(tmp_path,
                                  lon_min=lon_min, lon_max=lon_max,
                                  lat_min=lat_min, lat_max=lat_max,
                                  nx=nc_nx, ny=nc_ny, nt=nc_nt,
                                  start_date=start_date, depths=np.array(depths),
                                  variables=nc_vars)
        st.success(f"NetCDF created: {tmp_path} (dataset dims: {ds.dims})")
        # Download button using your dl_button wrapper
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        dl_button(st, "Download NetCDF", data, file_name="floatchat_ocean.nc", mime="application/x-netcdf")
        # Show quick plots
        fig_map = plot_variable_map_from_ds(ds, var=nc_vars[0], time_index=0, depth_index=0)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
        fig_profile = plot_variable_profile_at_point(ds, var=nc_vars[0], lon_val=None, lat_val=None, time_index=0)
        if fig_profile:
            st.plotly_chart(fig_profile, use_container_width=True)
        fig_ts = plot_variable_timeseries_at_point(ds, var=nc_vars[0], lon_val=None, lat_val=None, depth_index=0)
        if fig_ts:
            st.plotly_chart(fig_ts, use_container_width=True)



# Footer
st.markdown("---")
st.markdown("<div class='small muted'>Data: OBIS (api.obis.org) • LLM: OpenRouter • Created by Pushpal Sanyal</div>", unsafe_allow_html=True)
st.markdown("<div class='muted small'>Last updated: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "</div>", unsafe_allow_html=True)
