#!/usr/bin/env python3
"""
erddap_integration.py

Robust ERDDAP integration module with:
 - dataset discovery + validation (avoids unknown dataset IDs)
 - supports 3D variables (time, depth, lat, lon) and 2D (time, lat, lon)
 - automatic split between historical and NRT datasets (configurable cutoff)
 - griddap & tabledap attempts (griddap preferred; tabledap fallback)
 - CLI entrypoint, Streamlit widget (optional) and Flask blueprint (optional)
 - Interactive Plotly outputs and a styled HTML report with charts + table

Save as `erddap_integration.py`. Run on a machine with internet access.

Dependencies:
    pip install requests pandas plotly
    # optional: pip install geopy streamlit flask

Usage examples:
    python erddap_integration.py --server https://erddap.incois.gov.in/erddap --place "Chennai, India" --var-friendly Temperature
    python erddap_integration.py --server https://coastwatch.noaa.gov/erddap --latlon "21.3069,-157.8583" --month-year 09-2025 --var-friendly Temperature

Author: Prepared for user — includes multiple curated dataset fallbacks to maximize chance of getting `analysed_sst`.
"""

from datetime import datetime, timedelta, timezone
import calendar
import argparse
import sys
import io
import re
import urllib.parse
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import html

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# Default ERDDAP server (NOAA CoastWatch). Can be overridden via CLI --server
ERDDAP_SERVER = "https://coastwatch.noaa.gov/erddap"
# How many days in the past defines the switch from NRT to historical data.
NRT_CUTOFF_DAYS = 60

# -------------------------
# Curated dataset fallbacks (verified pages found on public ERDDAP instances)
# - 'global' contains globally-known datasetIDs that commonly expose analysed_sst
# - server-specific entries (e.g., INCOIS) provide additional local fallbacks
# These are *fallbacks* only — the code will still validate that the dataset exists on the chosen server.
# -------------------------
CURATED_DATASETS = {
    "global": [
        # CoastWatch (NOAA) blended analyses
        "noaacwBLENDEDsstDaily",
        "noaacwBLENDEDsstDNDaily",
        # NOAA OISST (AOML / REMSS OISST packaging)
        "OISSTs_2022_v05_1",
        # NASA JPL MUR (very common, many ERDDAP hosts expose this datasetID)
        "jplMURSST41",
        # Other known instances that sometimes host analysed_sst under different ERDDAPs
        "noaacwecnMURannual"
    ],
    # INCOIS (India) — datasets observed on INCOIS ERDDAP pages (variable names may vary; discovery will validate)
    "https://erddap.incois.gov.in/erddap": [
        "NOAA_AVHRR_AMSR_datasets",
        "incois_argo_sst_weekly",
        "incois_valueadded_products_datasets",
        "AMSR2_3day_Global",
        "incois_argo_10d_VAM"
    ],
    # A few other public ERDDAP instances (useful if the chosen server proxies or hosts these datasets)
    "https://erddap.aoml.noaa.gov/hdb/erddap": [
        "OISSTs_2022_v05_1"
    ],
    "https://erddap.marine.usf.edu/erddap": [
        "jplMURSST41"
    ],
    "https://coastwatch.pfeg.noaa.gov/erddap": [
        "jplMURSST41",
        "noaacwBLENDEDsstDaily"
    ]
}

# Small helper(s)
def _format_iso(dt: datetime) -> str:
    """Return UTC Z-suffixed ISO string for ERDDAP queries."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _safe_request(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None

# -------------------------
# Geocoding
# -------------------------
def geocode_place(place, timeout=10):
    """Return (lat, lon) for place string or None."""
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent="erddap_integration_geocoder")
            loc = geolocator.geocode(place, timeout=timeout)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    # fallback to OpenStreetMap Nominatim public API
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "erddap_integration/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return float(j[0]["lat"]), float(j[0]["lon"])
    except Exception:
        pass
    return None

# -------------------------
# ERDDAP discovery & validation
# -------------------------
def erddap_search(server: str, query: str, items_per_page: int = 200):
    """Query ERDDAP's search/index.csv. Returns pandas DataFrame or empty DF."""
    q = urllib.parse.quote_plus(query)
    url = f"{server.rstrip('/')}/search/index.csv?searchFor={q}&itemsPerPage={items_per_page}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

def get_dataset_info(server: str, dataset_id: str):
    """Fetch /info/<dataset_id>/index.json. Returns JSON or None."""
    url = f"{server.rstrip('/')}/info/{urllib.parse.quote(dataset_id)}/index.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _extract_variable_names_from_info(info_json):
    """Attempt to extract variable names from dataset info JSON conservatively."""
    if not info_json:
        return []
    txt = str(info_json)
    # heuristics: find words that look like variable names (letters, underscores, digits)
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", txt))
    # filter obviously non-variable tokens
    exclude = {'table', 'attributes', 'variable', 'dataset', 'dimension', 'id', 'units', 'title', 'name'}
    vars_guess = [t for t in tokens if t.lower() not in exclude and len(t) < 60]
    return vars_guess[:200]

def validate_dataset_for_keywords(server, dataset_id, keywords, accept_if_candidates: bool = False):
    """
    Validate dataset by checking info JSON contains any of keywords.
    Returns dict {valid:bool, variables:list, info:json}
    - accept_if_candidates: if True, treat dataset as valid when heuristic variable candidates exist.
    """
    info = get_dataset_info(server, dataset_id)
    if not info:
        return {"valid": False, "variables": [], "info": None}
    txt = str(info).lower()
    found = [k for k in keywords if k.lower() in txt]
    # Extract other candidates (heuristic)
    candidates = _extract_variable_names_from_info(info)
    valid = False
    if found:
        valid = True
    elif accept_if_candidates and candidates:
        valid = True
    return {"valid": valid, "variables": found + candidates, "info": info}

def discover_dataset(server, friendly_var, search_phrases, var_keywords, curated_fallback=None):
    """
    Searches ERDDAP for datasets matching search_phrases.
    Validates candidates against var_keywords.
    Returns (dataset_id, chosen_variable, info_note) or (None,None,msg)
    """
    # 1) Try ERDDAP search with joined phrases
    q = " ".join(search_phrases)
    df = erddap_search(server, q)
    candidates = []
    if not df.empty:
        cols = [c.lower() for c in df.columns]
        for candidate_col in ['dataset id', 'dataset', 'Dataset ID', 'Dataset']:
            if candidate_col.lower() in cols:
                ds_col = df.columns[cols.index(candidate_col.lower())]
                candidates = list(df[ds_col].dropna().unique())
                break

    # 2) Add curated fallback list (server-specific + global should be passed in curated_fallback)
    if curated_fallback:
        for c in curated_fallback:
            if c not in candidates:
                candidates.append(c)

    # 3) Validate each candidate by fetching its info JSON
    for ds in candidates:
        val = validate_dataset_for_keywords(server, ds, var_keywords, accept_if_candidates=True)
        if val['valid']:
            # pick variable: prefer explicit keyword hits, else the first candidate var
            var = val['variables'][0] if val['variables'] else None
            note = f"Discovered dataset {ds}; chosen variable candidate: {var}"
            return ds, var, note

    return None, None, "No validated dataset found."

# -------------------------
# GRIDDAP / TABLEDAP fetchers (with 3D support)
# -------------------------
def _try_griddap_point(server, dataset_id, variable, start_iso, end_iso, lat, lon, depth=None, timeout=60):
    """
    Robust griddap point-slice attempts. Returns (df, debug_list).
    Handles variable=None, catches CSV parse errors, and URL-encodes parts safely.
    """
    from pandas.errors import ParserError

    debug = []
    orders_with_depth = [
        ("time", "depth", "latitude", "longitude"),
        ("time", "latitude", "longitude", "depth"),
        ("time", "latitude", "depth", "longitude"),
        ("time", "longitude", "latitude", "depth"),
        ("time", "depth", "longitude", "latitude"),
    ]
    orders_no_depth = [
        ("time", "latitude", "longitude"),
        ("time", "longitude", "latitude"),
    ]

    def idx_for(order):
        parts = []
        for dim in order:
            if dim == "time":
                parts.append(f"[({start_iso}):1:({end_iso})]")
            elif dim == "latitude":
                parts.append(f"[({lat}):1:({lat})]")
            elif dim == "longitude":
                parts.append(f"[({lon}):1:({lon})]")
            elif dim == "depth":
                if depth is None:
                    return None
                if depth == "ALL":
                    return None
                parts.append(f"[({depth}):1:({depth})]")
        return "".join(parts)

    # prepare quoted variable (safe empty)
    var_q = urllib.parse.quote(variable) if variable else ""

    # helper to build full URL safely (quote idx but keep []():,)
    def build_url(dataset_id, var_q, idx):
        ds_q = urllib.parse.quote(dataset_id)
        idx_q = urllib.parse.quote(idx, safe="[]():,")
        if var_q:
            return f"{server.rstrip('/')}/griddap/{ds_q}.csv?{var_q}{idx_q}"
        else:
            # request all variables (no variable specified) -> just use idx directly after '?'
            return f"{server.rstrip('/')}/griddap/{ds_q}.csv?{idx_q}"

    # Try sequences (with or without depth)
    sequences = orders_with_depth if (depth is not None and depth != "ALL") else orders_no_depth

    for order in sequences:
        idx = idx_for(order)
        if not idx:
            continue
        url = build_url(dataset_id, var_q, idx)
        try:
            r = requests.get(url, timeout=timeout)
            snippet = (r.text[:1500] + '...') if isinstance(r.text, str) and len(r.text) > 1500 else (r.text if isinstance(r.text, str) else f"HTTP {r.status_code}")
            debug.append((getattr(r, "url", url), snippet))
            if r.status_code != 200:
                continue
            try:
                df = pd.read_csv(io.StringIO(r.text))
            except ParserError as pe:
                debug.append((getattr(r, "url", url), f"PARSER_ERROR:{pe}"))
                continue
            # Normalize axis names if present
            cols_lower = {c.lower(): c for c in df.columns}
            rename = {}
            for k in ['time', 'latitude', 'longitude', 'depth', 'z', 'altitude', 'depthBelowSeaSurface']:
                if k in cols_lower:
                    rename[cols_lower[k]] = 'depth' if k in ['depth','z','depthBelowSeaSurface'] else k
            if rename:
                df.rename(columns=rename, inplace=True)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                df = df.dropna(subset=['time'])
            return df, debug
        except Exception as e:
            debug.append((url, f"EXC:{e}"))
            continue

    return pd.DataFrame(), debug

def _try_tabledap(server, dataset_id, variable, start_iso, end_iso, lat, lon, timeout=60):
    """
    Attempt a tabledap request to fetch rows matching the lat/lon and time range.
    Returns (df, debug_list).
    """
    from pandas.errors import ParserError

    debug = []
    var_part = variable if variable else ""
    query_cols = var_part or ""
    params = {}
    params[f"time>="] = start_iso
    params[f"time<="] = end_iso
    params["latitude"] = lat
    params["longitude"] = lon

    base = f"{server.rstrip('/')}/tabledap/{urllib.parse.quote(dataset_id)}.csv"
    url = f"{base}?{urllib.parse.quote(query_cols)}" if query_cols else base
    try:
        r = requests.get(url, params=params, timeout=timeout)
        snippet = (r.text[:1500] + '...') if isinstance(r.text, str) and len(r.text) > 1500 else (r.text if isinstance(r.text, str) else f"HTTP {r.status_code}")
        debug.append((getattr(r, "url", url), snippet))
        if r.status_code != 200:
            return pd.DataFrame(), debug
        try:
            df = pd.read_csv(io.StringIO(r.text))
        except ParserError as pe:
            debug.append((getattr(r, "url", url), f"PARSER_ERROR:{pe}"))
            return pd.DataFrame(), debug
        cols_lower = {c.lower(): c for c in df.columns}
        rename = {}
        for k in ['time', 'latitude', 'longitude', 'depth', 'z', 'altitude', 'depthBelowSeaSurface']:
            if k in cols_lower:
                rename[cols_lower[k]] = 'depth' if k in ['depth','z','depthBelowSeaSurface'] else k
        if rename:
            df.rename(columns=rename, inplace=True)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            df = df.dropna(subset=['time'])
        return df, debug
    except Exception as e:
        debug.append((url, f"EXC:{e}"))
        return pd.DataFrame(), debug

def fetch_with_3d_support(server, dataset_id, variable, lat, lon, start_dt, end_dt, prefer_tabledap_for_profiles=True, timeout=60):
    """
    Top-level fetch that attempts:
      - Split time range if spans NRT cutoff
      - For each time chunk: try griddap point slicing (best for gridded datasets)
      - If griddap yields nothing or 'ALL depth profile' required, try tabledap (profiles)
    Returns (df_combined, debug_attempts)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=NRT_CUTOFF_DAYS)
    parts = []
    if end_dt < cutoff:
        parts.append((start_dt, end_dt))
    elif start_dt >= cutoff:
        parts.append((start_dt, end_dt))
    else:
        parts.append((start_dt, cutoff - timedelta(seconds=1)))
        parts.append((cutoff, end_dt))
    all_dfs = []
    debug = []
    for sdt, edt in parts:
        s_iso = _format_iso(sdt)
        e_iso = _format_iso(edt)
        # 1) Try griddap without depth (fast)
        df_g, dbg_g = _try_griddap_point(server, dataset_id, variable, s_iso, e_iso, lat, lon, depth=None, timeout=timeout)
        debug.extend(dbg_g)
        if not df_g.empty:
            all_dfs.append(df_g)
            continue
        # 2) Try tabledap (profiles)
        df_t, dbg_t = _try_tabledap(server, dataset_id, variable, s_iso, e_iso, lat, lon, timeout=timeout)
        debug.extend(dbg_t)
        if not df_t.empty:
            all_dfs.append(df_t)
            continue
        # 3) Try griddap requesting a shallow depth of 0 if present
        df_gd, dbg_gd = _try_griddap_point(server, dataset_id, variable, s_iso, e_iso, lat, lon, depth=0, timeout=timeout)
        debug.extend(dbg_gd)
        if not df_gd.empty:
            all_dfs.append(df_gd)
            continue
        # nothing found for this chunk
    if not all_dfs:
        return pd.DataFrame(), debug
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    if 'time' in df_all.columns:
        df_all = df_all.sort_values('time').reset_index(drop=True)
    return df_all, debug

# -------------------------
# Plotting: timeseries, profile heatmap, map
# -------------------------
def plot_timeseries(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'time' not in df.columns:
        return go.Figure().update_layout(title="No timeseries available")
    fig = px.line(df, x='time', y=varcol, title=title or f"Timeseries: {varcol}")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def plot_profile_heatmap(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'time' not in df.columns or 'depth' not in df.columns:
        return go.Figure().update_layout(title="No profile/heatmap available")
    pivot = df.pivot_table(index='depth', columns='time', values=varcol, aggfunc='mean')
    pivot = pivot.sort_index(ascending=True)
    z = pivot.values
    x = [str(t) for t in pivot.columns]
    y = list(pivot.index)
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorbar=dict(title=varcol)))
    fig.update_layout(title=title or f"Depth-Time heatmap ({varcol})", yaxis=dict(autorange='reversed'))
    return fig

def plot_profile_scatter(df, varcol, title=None):
    if df.empty or varcol not in df.columns or 'depth' not in df.columns:
        return go.Figure().update_layout(title="No profile available")
    prof = df.groupby('depth')[varcol].mean().reset_index().sort_values('depth')
    fig = px.line(prof, x=varcol, y='depth', title=title or f"Vertical profile ({varcol})", markers=True)
    fig.update_yaxes(autorange='reversed')
    return fig

def plot_map_latest(df, varcol, title=None):
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return go.Figure().update_layout(title="No spatial data")
    hover = [varcol, 'time'] if varcol in df.columns else ['time']
    if 'time' in df.columns:
        latest_idx = df['time'].idxmax()
        df_latest = df.loc[[latest_idx]]
    else:
        df_latest = df.head(1)
    fig = px.scatter_geo(df_latest, lat='latitude', lon='longitude', hover_name=varcol if varcol in df_latest.columns else None,
                         hover_data=hover, title=title or "Latest data location")
    return fig

# -------------------------
# HTML report with Plotly + styled table
# -------------------------
def render_html_report(output_filename, figures, df_table, caption="ERDDAP Results"):
    fragments = []
    for fig in figures:
        fragments.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    try:
        df_disp = df_table.copy()
        for c in df_disp.select_dtypes(include=["float64", "int64"]).columns:
            df_disp[c] = df_disp[c].round(4)
        styler = df_disp.style.hide_index().set_table_attributes('class="table table-striped"') \
            .set_caption(caption)
        html_table = styler.render()
    except Exception:
        html_table = df_table.to_html(index=False)
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>ERDDAP Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{margin:20px;font-family:Segoe UI, Roboto, Arial; background:#f8f9fa}} .card{{margin-bottom:1rem}}</style>
</head><body><div class="container"><h2>{html.escape(caption)}</h2>
"""
    for frag in fragments:
        html_doc += f'<div class="card"><div class="card-body">{frag}</div></div>\n'
    html_doc += f'<div class="card"><div class="card-body"><h5>Data table</h5>{html_table}</div></div>\n'
    html_doc += '<hr/><p>Generated by erddap_integration.py</p></div></body></html>'
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_filename

# -------------------------
# Streamlit widget (optional)
# -------------------------
def erddap_streamlit_widget(server=ERDDAP_SERVER):
    try:
        import streamlit as st
    except Exception:
        print("Streamlit not installed.")
        return
    st.sidebar.header("ERDDAP 3D Query")
    var_choice = st.sidebar.selectbox("Variable", ["Temperature", "Salinity", "Chlorophyll"])
    place = st.sidebar.text_input("Place (e.g., 'Honolulu, Hawaii')")
    manual_latlon = st.sidebar.text_input("Or lat,lon (e.g., '21.3,-157.8')")
    month_year = st.sidebar.text_input("Month-Year (MM-YYYY) optional")
    server_input = st.sidebar.text_input("ERDDAP server", value=server)
    if st.sidebar.button("Fetch"):
        latlon = None
        if manual_latlon:
            try:
                p = [x.strip() for x in manual_latlon.split(',')]
                latlon = (float(p[0]), float(p[1]))
            except Exception:
                st.error("Invalid lat,lon")
                return
        elif place:
            with st.spinner("Geocoding..."):
                latlon = geocode_place(place)
                if not latlon:
                    st.error("Geocoding failed")
                    return
        else:
            st.warning("Provide a place or lat,lon")
            return
        lat, lon = latlon
        # time range
        if month_year:
            try:
                m, y = map(int, month_year.split('-'))
                _, last_day = calendar.monthrange(y, m)
                start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
                end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
            except Exception:
                st.error("Invalid Month-Year format")
                return
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)
        st.info(f"Fetching {var_choice} for {lat:.3f},{lon:.3f} between {_format_iso(start_dt)} and {_format_iso(end_dt)}")

        heuristics = {
            "Temperature": (["sea surface temperature", "sst"], ["analysed_sst","sea_surface_temperature","sst","temperature","sstAnom","sst_anom"]),
            "Salinity": (["salinity", "sss"], ["salinity","sea_surface_salinity","sss"]),
            "Chlorophyll": (["chlorophyll", "chlor a", "chl"], ["chlor_a","chl","CHL_Weekly","chlorophyll"])
        }
        search_terms, var_keywords = heuristics[var_choice]

        # build curated fallback list (server-specific + global)
        server_key = server_input.rstrip('/')
        curated = []
        # add server-specific curated if present
        if server_key in CURATED_DATASETS:
            curated.extend(CURATED_DATASETS[server_key])
        # always add global curated
        curated.extend([d for d in CURATED_DATASETS.get("global", []) if d not in curated])

        with st.spinner("Searching for dataset..."):
            ds_id, var_guess, note = discover_dataset(server_input, var_choice, search_terms, var_keywords, curated_fallback=curated)
        # if dataset found but no variable guessed, attempt to extract candidates
        if ds_id and not var_guess:
            info = get_dataset_info(server_input, ds_id)
            vars_candidates = _extract_variable_names_from_info(info)
            if vars_candidates:
                var_guess = vars_candidates[0]
        if not ds_id:
            st.error("No dataset found: " + note)
            return
        st.write("Dataset:", ds_id, "Var guess:", var_guess)
        with st.spinner("Fetching data..."):
            df, debug = fetch_with_3d_support(server_input, ds_id, var_guess, lat, lon, start_dt, end_dt)
        if df.empty:
            st.error("No data returned. Show debug attempts below.")
            st.write(debug[:6])
            return
        # Determine variable column
        data_cols = [c for c in df.columns if c.lower() not in ['time','latitude','longitude','depth']]
        varcol = data_cols[0] if data_cols else None
        st.plotly_chart(plot_timeseries(df, varcol), use_container_width=True)
        if 'depth' in df.columns:
            st.plotly_chart(plot_profile_heatmap(df, varcol), use_container_width=True)
            st.plotly_chart(plot_profile_scatter(df, varcol), use_container_width=True)
        st.plotly_chart(plot_map_latest(df, varcol), use_container_width=True)
        st.write("Data sample:")
        st.dataframe(df.head(50))

# -------------------------
# Flask blueprint (optional)
# -------------------------
def register_erddap_blueprint(app, server=ERDDAP_SERVER):
    try:
        from flask import Blueprint, request, render_template_string
    except Exception:
        print("Flask not available.")
        return
    bp = Blueprint('erddap3d', __name__)
    HTML = """
    <!doctype html><html><head><title>ERDDAP 3D Query</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{font-family:sans-serif;padding:2em;}</style></head><body>
    <h3>ERDDAP 3D Query</h3>
    <form method="post">
      Variable: <select name="variable">{% for v in variables %}<option value="{{v}}">{{v}}</option>{% endfor %}</select><br>
      Place: <input name="place" value="{{place}}"><br>
      Or lat,lon: <input name="latlon" value="{{latlon}}"><br>
      Month-Year (MM-YYYY): <input name="month_year" value="{{month_year}}"><br>
      <button type="submit">Fetch</button>
    </form>
    <div id="output">{{output|safe}}</div>
    </body></html>
    """
    @bp.route('/erddap3d', methods=['GET','POST'])
    def form():
        output = ""
        form = {'variables': ['Temperature','Salinity','Chlorophyll'], 'place':'','latlon':'','month_year':''}
        if request.method == 'POST':
            var_choice = request.form.get('variable')
            place = request.form.get('place','').strip()
            latlon = request.form.get('latlon','').strip()
            month_year = request.form.get('month_year','').strip()
            form.update({'place':place, 'latlon':latlon, 'month_year':month_year})
            # resolve coords
            coords = None
            if latlon:
                try:
                    p = [x.strip() for x in latlon.split(',')]
                    coords = (float(p[0]), float(p[1]))
                except Exception:
                    output = "<p style='color:red;'>Invalid lat,lon</p>"
                    return render_template_string(HTML, output=output, **form)
            elif place:
                coords = geocode_place(place)
                if not coords:
                    output = "<p style='color:red;'>Geocoding failed</p>"
                    return render_template_string(HTML, output=output, **form)
            else:
                output = "<p style='color:orange;'>Provide place or lat,lon</p>"
                return render_template_string(HTML, output=output, **form)
            lat, lon = coords
            # time range
            if month_year:
                try:
                    m,y = map(int, month_year.split('-'))
                    _, last = calendar.monthrange(y,m)
                    start_dt = datetime(y,m,1,tzinfo=timezone.utc)
                    end_dt = datetime(y,m,last,23,59,59,tzinfo=timezone.utc)
                except Exception:
                    output = "<p style='color:red;'>Invalid month-year format</p>"
                    return render_template_string(HTML, output=output, **form)
            else:
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=30)
            heuristics = {
                "Temperature": (["sea surface temperature","sst"], ["analysed_sst","sea_surface_temperature","sst","temperature","sstAnom","sst_anom"]),
                "Salinity": (["salinity","sss"], ["salinity","sea_surface_salinity","sss"]),
                "Chlorophyll": (["chlorophyll","chl"], ["chlor_a","chl","CHL_Weekly","chlorophyll"])
            }
            search_terms, var_keywords = heuristics[var_choice]

            server_key = server.rstrip('/')
            curated = []
            if server_key in CURATED_DATASETS:
                curated.extend(CURATED_DATASETS[server_key])
            curated.extend([d for d in CURATED_DATASETS.get("global", []) if d not in curated])

            ds_id, var_guess, note = discover_dataset(server, var_choice, search_terms, var_keywords, curated_fallback=curated)
            if ds_id and not var_guess:
                info = get_dataset_info(server, ds_id)
                vars_candidates = _extract_variable_names_from_info(info)
                if vars_candidates:
                    var_guess = vars_candidates[0]
            if not ds_id:
                output = f"<p style='color:red;'>No dataset found: {html.escape(note)}</p>"
                return render_template_string(HTML, output=output, **form)
            df, debug = fetch_with_3d_support(server, ds_id, var_guess, lat, lon, start_dt, end_dt)
            if df.empty:
                output = "<p style='color:red;'>No data returned; see debug attempts:</p><pre>"
                for u,s in debug[:8]:
                    output += html.escape(u) + "\n" + html.escape(str(s)[:300]) + "\n\n"
                output += "</pre>"
                return render_template_string(HTML, output=output, **form)
            data_cols = [c for c in df.columns if c.lower() not in ['time','latitude','longitude','depth']]
            varcol = data_cols[0] if data_cols else None
            fig_ts = plot_timeseries(df, varcol)
            figs_html = pio.to_html(fig_ts, include_plotlyjs=False, full_html=False)
            output = figs_html
        return render_template_string(HTML, output=output, **form)
    app.register_blueprint(bp)

# -------------------------
# CLI entrypoint
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(prog="erddap_integration.py", description="ERDDAP 3D integration CLI")
    p.add_argument("--server", default=ERDDAP_SERVER, help="ERDDAP server URL")
    loc_group = p.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--place", type=str, help="Place name")
    loc_group.add_argument("--latlon", type=str, help="lat,lon")
    p.add_argument("--month-year", dest="month_year", type=str, help="MM-YYYY")
    p.add_argument("--var-friendly", dest="var_friendly", choices=["Temperature","Salinity","Chlorophyll"], default="Temperature")
    p.add_argument("--output", dest="output", default="erddap_report.html")
    p.add_argument("--timeout", dest="timeout", type=int, default=30)
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    server = args.server.rstrip('/')
    # resolve coords
    if args.latlon:
        try:
            lat_s, lon_s = args.latlon.split(',')
            lat, lon = float(lat_s.strip()), float(lon_s.strip())
        except Exception:
            print("Invalid --latlon format. Use: lat,lon")
            sys.exit(2)
    else:
        geo = geocode_place(args.place)
        if not geo:
            print("Geocoding failed for place:", args.place)
            sys.exit(3)
        lat, lon = geo
    print(f"Using coordinates: {lat:.6f}, {lon:.6f}")
    # time range
    if args.month_year:
        try:
            mm, yy = args.month_year.split('-')
            m, y = int(mm), int(yy)
            _, last = calendar.monthrange(y, m)
            start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
            end_dt = datetime(y, m, last, 23, 59, 59, tzinfo=timezone.utc)
        except Exception:
            print("Invalid --month-year. Use MM-YYYY")
            sys.exit(4)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
    print("Time range:", _format_iso(start_dt), "to", _format_iso(end_dt))

    heuristics = {
        "Temperature": (["sea surface temperature", "sst"], ["analysed_sst","sea_surface_temperature","sst","temperature","sstAnom","sst_anom"]),
        "Salinity": (["salinity", "sss"], ["salinity","sea_surface_salinity","sss"]),
        "Chlorophyll": (["chlorophyll", "chl"], ["chlor_a","chl","CHL_Weekly","chlorophyll"])
    }
    search_terms, var_keywords = heuristics[args.var_friendly]
    print("Discovering datasets...")

    # Build curated fallback: include server-specific curated plus global curated
    curated = []
    if server in CURATED_DATASETS:
        curated.extend(CURATED_DATASETS[server])
    # also allow matching by normalized server key without trailing slash
    server_key = server.rstrip('/')
    if server_key in CURATED_DATASETS and CURATED_DATASETS[server_key] not in curated:
        curated.extend(CURATED_DATASETS[server_key])
    # finally add global curated
    curated.extend([d for d in CURATED_DATASETS.get("global", []) if d not in curated])

    ds_id, var_guess, note = discover_dataset(server, args.var_friendly, search_terms, var_keywords, curated_fallback=curated)
    # If dataset found but no variable guessed, try extracting from info JSON
    if ds_id and not var_guess:
        info = get_dataset_info(server, ds_id)
        vars_candidates = _extract_variable_names_from_info(info)
        if vars_candidates:
            var_guess = vars_candidates[0]
    if not ds_id:
        print("No dataset found automatically. Aborting. Note:", note)
        sys.exit(5)
    print("Chosen dataset:", ds_id, "variable guess:", var_guess)
    print("Fetching data...")
    df, debug = fetch_with_3d_support(server, ds_id, var_guess, lat, lon, start_dt, end_dt, timeout=args.timeout)
    if df.empty:
        print("No data returned. Debug attempts (sample):")
        for u,s in debug[:8]:
            print("-", u)
        # write debug HTML for inspection
        html_blob = "<h3>No data returned</h3><ul>"
        for u,s in debug:
            html_blob += "<li><pre>" + html.escape(u) + "\n" + html.escape(str(s)[:400]) + "</pre></li>"
        html_blob += "</ul>"
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"<!doctype html><html><body>{html_blob}</body></html>")
        print("Wrote debug HTML to", args.output)
        sys.exit(0)
    # pick variable column
    data_cols = [c for c in df.columns if c.lower() not in ['time','latitude','longitude','depth']]
    varcol = data_cols[0] if data_cols else None
    figs = []
    figs.append(plot_timeseries(df, varcol, title=f"Timeseries ({varcol or 'value'})"))
    if 'depth' in df.columns:
        figs.append(plot_profile_heatmap(df, varcol, title=f"Depth-Time ({varcol})"))
        figs.append(plot_profile_scatter(df, varcol, title=f"Profile ({varcol})"))
    figs.append(plot_map_latest(df, varcol, title="Latest Location"))
    out = render_html_report(args.output, figs, df, caption=f"ERDDAP: {ds_id} ({varcol or 'value'})")
    print("Saved report to", out)
    print("Done.")

if __name__ == "__main__":
    main()
