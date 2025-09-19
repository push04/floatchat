#!/usr/bin/env python3
"""
erddap_integration_validated.py

- Discovers valid ERDDAP datasets for the requested variable (Temperature, Salinity, Chlorophyll)
  by querying the ERDDAP search API and validating dataset info.
- Fetches point time-series from griddap (or tabledap fallback).
- Generates interactive Plotly charts and a styled HTML table saved to erddap_output.html.

Usage examples (CLI):
  python erddap_integration_validated.py --place "Honolulu, Hawaii"
  python erddap_integration_validated.py --latlon "21.3069,-157.8583" --month-year 08-2025 --var-friendly Temperature

Dependencies:
  pip install requests pandas plotly
  # optional: pip install geopy
"""

from datetime import datetime, timedelta, timezone
import calendar
import argparse
import sys
import io
import urllib.parse
import requests
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import html

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

ERDDAP_SERVER = "https://coastwatch.noaa.gov/erddap"  # default server
NRT_CUTOFF_DAYS = 60  # cutover between NRT and historical


# -- helpers: time formatting -------------------------------------------------
def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- geocode -----------------------------------------------------------------
def geocode_place(place, timeout=10):
    """Return (lat, lon) or None."""
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent="erddap_integration_validated")
            loc = geolocator.geocode(place, timeout=timeout)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    # fallback to public nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "erddap_integration_validated/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return float(j[0]["lat"]), float(j[0]["lon"])
    except Exception:
        pass
    return None


# -- ERDDAP discovery & validation ------------------------------------------
def erddap_search(server: str, query: str, items_per_page: int = 200, timeout: int = 30) -> pd.DataFrame:
    """
    Use ERDDAP `search/index.csv` to find datasets matching query.
    Returns a pandas DataFrame of search results if possible.
    """
    q = urllib.parse.quote_plus(query)
    url = f"{server}/search/index.csv?searchFor={q}&itemsPerPage={items_per_page}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception:
        # return empty DataFrame on failure
        return pd.DataFrame()


def validate_dataset_has_variable(server: str, dataset_id: str, var_keywords: list, timeout: int = 20) -> dict:
    """
    Validate a dataset by requesting /info/<dataset_id>/index.json.
    Returns dict with keys: valid (bool), variables (list of names), info (raw json or None)
    """
    url = f"{server}/info/{dataset_id}/index.json"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        info = r.json()
        # info is a dict with 'table' probably; find variable names in the JSON
        vars_found = []
        # try to walk the JSON to find variable names
        if isinstance(info, dict):
            # often info['table']['rows'] contains rows where col0 is name etc.
            # We'll search the entire JSON for strings matching variable-like patterns.
            txt = str(info).lower()
            for kw in var_keywords:
                if kw.lower() in txt:
                    vars_found.append(kw)
        return {"valid": True, "variables": vars_found, "info": info}
    except Exception:
        return {"valid": False, "variables": [], "info": None}


def discover_and_pick_dataset(server: str, friendly_var: str, search_terms: list, var_keywords: list, prefer_griddap=True):
    """
    Discover dataset candidates using search terms; validate them to pick a dataset id
    and the likely variable name to request. Returns (dataset_id, variable_name, note).
    If nothing found, returns (None, None, message).
    """
    # 1) search with the joined search_terms
    joined_query = " ".join(search_terms)
    search_df = erddap_search(server, joined_query)

    # 2) build candidate list: datasetID from search_df 'Dataset ID' or 'Dataset ID' column
    candidates = []
    if not search_df.empty:
        # column variations: 'Dataset ID' or 'Dataset'
        cols_lower = [c.lower() for c in search_df.columns]
        id_col = None
        for name in ['dataset id', 'dataset', 'datasetid', 'Dataset ID']:
            if name.lower() in cols_lower:
                id_col = search_df.columns[cols_lower.index(name.lower())]
                break
        if id_col:
            candidates = list(search_df[id_col].dropna().unique())
    # fallback curated candidates (safe list)
    curated = {
        "Temperature": ["noaacwLEOACSPOSSTL3SnrtCDaily", "jplMURSST41", "noaacwecnMURannual"],
        "Chlorophyll": ["noaacwNPPVIIRSchlaSectorYWDaily", "noaacwNPPVIIRSchlaGlobal", "USM_VIIRS_DAP"],
        "Salinity": ["NCOM_amseas_latest3d", "ncom_global", "salinity"]  # fallback only, will validate
    }
    for c in curated.get(friendly_var, []):
        if c not in candidates:
            candidates.append(c)

    # 3) validate candidates
    for ds in candidates:
        res = validate_dataset_has_variable(server, ds, var_keywords)
        if res["valid"]:
            # attempt to select a variable name by inspecting the metadata 'info' if possible
            var_name = None
            info = res.get("info")
            if isinstance(info, dict):
                # try to find variable list in JSON structure (conservative)
                txt = str(info)
                for vk in var_keywords:
                    # pick first keyword that appears in the info text
                    if vk.lower() in txt.lower():
                        var_name = vk
                        break
            # if no variable name guessed, leave variable None (fetcher will attempt to use first non-axis column)
            note = f"Selected dataset {ds}; guessed variable '{var_name}' from keywords." if var_name else f"Selected dataset {ds}; variable to be determined from dataset."
            return ds, var_name, note
    return None, None, "No validated datasets found on server for this variable."


# -- Low-level fetch --------------------------------------------------------
def try_griddap_fetch(server, dataset_id, variable, lat, lon, start_iso, end_iso, timeout=60):
    """
    Attempt to fetch using griddap point slice URL.
    Returns DataFrame (possibly empty) and debug info (url,text snippet).
    """
    # Try both lat,lon and lon,lat orders (some datasets differ)
    orders = [("lat", "lon"), ("lon", "lat")]
    raw_debug = []
    for order in orders:
        if order == ("lat", "lon"):
            point_idx = f"[({start_iso}):1:({end_iso})][({lat}):1:({lat})][({lon}):1:({lon})]"
        else:
            point_idx = f"[({start_iso}):1:({end_iso})][({lon}):1:({lon})][({lat}):1:({lat})]"
        if variable:
            query = f"{variable}{point_idx}"
        else:
            # unknown variable: request CSV without variable selector - this is sometimes rejected.
            # fallback: attempt to request time-lon-lat if griddap supports `.csv?time,latitude,longitude`
            query = f"time,latitude,longitude{point_idx}"
        url = f"{server}/griddap/{dataset_id}.csv?{query}"
        try:
            r = requests.get(url, timeout=timeout)
            raw_text = r.text[:2000] if r.text else f"HTTP {r.status_code}"
            raw_debug.append((url, raw_text))
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            # normalize columns
            cols_lower = {c.lower(): c for c in df.columns}
            rename_map = {}
            for key in ['time', 'latitude', 'longitude']:
                if key in cols_lower:
                    rename_map[cols_lower[key]] = key
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                df = df.dropna(subset=['time'])
            # rename first non-axis col to variable for clarity if variable provided
            data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
            if data_cols and variable and data_cols[0] != variable:
                df.rename(columns={data_cols[0]: variable}, inplace=True)
            return df, raw_debug
        except Exception as e:
            raw_debug.append((url, f"EXC:{e}"))
            continue
    return pd.DataFrame(), raw_debug


def try_tabledap_fetch(server, dataset_id, variable, lat, lon, start_iso, end_iso, timeout=60):
    """
    Fallback: try tabledap-style query. Many datasets expose tabledap values.
    Returns (DataFrame, debug_texts)
    """
    # Build a simple time-range & bounding box query for tabledap: datasetID.csv?var&time>=...&time<=...&latitude=...
    raw_debug = []
    # If variable present, use it; else request all columns
    var_part = variable if variable else ""
    # tabledap time constraints use &time>=...&time<=...
    url = f"{server}/tabledap/{dataset_id}.csv?{var_part}&time>={start_iso}&time<={end_iso}&latitude={lat}&longitude={lon}"
    try:
        r = requests.get(url, timeout=timeout)
        raw_debug.append((url, r.text[:2000] if r.text else f"HTTP {r.status_code}"))
        if r.status_code != 200:
            return pd.DataFrame(), raw_debug
        df = pd.read_csv(io.StringIO(r.text))
        # normalize time
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            df = df.dropna(subset=['time'])
        return df, raw_debug
    except Exception as e:
        raw_debug.append((url, f"EXC:{e}"))
        return pd.DataFrame(), raw_debug


# -- High level fetch with strategy (historical vs NRT split) --------------
def fetch_point_timeseries(server, dataset_id, variable, lat, lon, start_dt, end_dt, debug=False):
    """
    Fetch the point timeseries for given dataset. Will attempt griddap first, then tabledap.
    If the requested time range spans the NRT cutoff, this function will split the request
    and concatenate the two pieces.
    Returns (df, debug_info)
    debug_info is a list of attempted URLs/snippets.
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
    all_debug = []
    for sdt, edt in parts:
        s_iso = _format_iso(sdt)
        e_iso = _format_iso(edt)
        # try griddap
        df_g, debug_g = try_griddap_fetch(server, dataset_id, variable, lat, lon, s_iso, e_iso)
        all_debug.extend(debug_g)
        if not df_g.empty:
            all_dfs.append(df_g)
            continue
        # fallback tabledap
        df_t, debug_t = try_tabledap_fetch(server, dataset_id, variable, lat, lon, s_iso, e_iso)
        all_debug.extend(debug_t)
        if not df_t.empty:
            all_dfs.append(df_t)

    if not all_dfs:
        return pd.DataFrame(), all_debug
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    if 'time' in df_all.columns:
        df_all = df_all.sort_values('time').reset_index(drop=True)
    return df_all, all_debug


# -- plotting & HTML output -------------------------------------------------
def make_plots_and_html(df: pd.DataFrame, variable_label: str, output_filename: str = "erddap_output.html"):
    """
    Create interactive Plotly time series + map and a beautiful HTML table, saved to output_filename.
    Returns full path of saved HTML.
    """
    figs = []
    note = ""
    if df.empty:
        # create a single empty figure
        figs.append(go.Figure().update_layout(title="No data available"))
        html_table = "<p>No data to show.</p>"
    else:
        # time series
        data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
        varcol = data_cols[0] if data_cols else None
        if varcol and 'time' in df.columns:
            fig_ts = px.line(df.sort_values('time'), x='time', y=varcol, title=f"Timeseries for {variable_label or varcol}")
            fig_ts.update_xaxes(rangeslider_visible=True)
            figs.append(fig_ts)
        else:
            figs.append(go.Figure().update_layout(title="No timeseries variable found"))
        # map - latest point
        if 'latitude' in df.columns and 'longitude' in df.columns and 'time' in df.columns:
            latest_idx = df['time'].idxmax()
            df_latest = df.loc[[latest_idx]]
            if varcol:
                fig_map = px.scatter_geo(df_latest, lat='latitude', lon='longitude', hover_name=varcol,
                                         hover_data=['time'], title=f"Latest {variable_label or varcol}")
            else:
                fig_map = px.scatter_geo(df_latest, lat='latitude', lon='longitude', hover_data=['time'],
                                         title="Latest location")
            figs.append(fig_map)
        else:
            figs.append(go.Figure().update_layout(title="No spatial data to show"))

        # styled HTML table using pandas Styler
        try:
            # round numeric columns sensibly
            df_disp = df.copy()
            for c in df_disp.select_dtypes(include=["float64", "int64"]).columns:
                df_disp[c] = df_disp[c].round(4)
            styler = df_disp.style.hide_index().set_table_attributes('class="dataframe table table-striped"') \
                .set_caption(f"Data table (rows: {len(df_disp)})")
            html_table = styler.render()
        except Exception:
            # fallback
            html_table = df.head(100).to_html(index=False)
    # Compose a single HTML file with Plotly figures + the table
    # Use plotly.io.to_html to get standalone fragments
    fragments = []
    for fig in figs:
        fragments.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=None))

    # include Plotly.js once via CDN, and add Bootstrap CDN for table styling
    html_head = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>ERDDAP Output</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <style>body{margin:20px;font-family:Segoe UI,Roboto,Arial;background:#f8f9fa;} .card{margin-bottom:1rem;}</style>
    </head>
    <body>
    <div class="container">
      <h2>ERDDAP Query Results</h2>
    """
    html_tail = """
      <hr/>
      <p>Generated by erddap_integration_validated.py</p>
    </div>
    </body></html>
    """
    body = []
    # add plots in cards
    for i, frag in enumerate(fragments):
        body.append(f'<div class="card"><div class="card-body">{frag}</div></div>')
    # add table card
    body.append(f'<div class="card"><div class="card-body"><h5>Data table</h5>{html_table}</div></div>')
    html_all = html_head + "\n".join(body) + html_tail

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_all)
    return output_filename


# -- CLI --------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(prog="erddap_integration_validated.py")
    p.add_argument("--server", default=ERDDAP_SERVER, help="ERDDAP server base URL")
    loc_group = p.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--place", type=str, help="Place name (e.g., 'Honolulu, Hawaii')")
    loc_group.add_argument("--latlon", type=str, help="Manual lat,lon (e.g., '21.3,-157.8')")
    p.add_argument("--month-year", dest="month_year", type=str, help="MM-YYYY for a specific month (optional)")
    p.add_argument("--var-friendly", dest="var_friendly", choices=["Temperature", "Salinity", "Chlorophyll"], default="Temperature")
    p.add_argument("--output", dest="output", default="erddap_output.html", help="Output HTML filename")
    p.add_argument("--timeout", dest="timeout", type=int, default=30, help="HTTP timeout seconds")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Resolve lat/lon
    latlon = None
    if args.latlon:
        try:
            lat_str, lon_str = args.latlon.split(",")
            latlon = (float(lat_str.strip()), float(lon_str.strip()))
        except Exception:
            print("Invalid --latlon format. Use: lat,lon")
            sys.exit(2)
    elif args.place:
        print(f"Geocoding '{args.place}' ...")
        latlon = geocode_place(args.place)
    if not latlon:
        print("Could not resolve location. Exiting.")
        sys.exit(3)
    lat, lon = latlon
    print(f"Using coordinates: {lat:.6f}, {lon:.6f}")

    # Resolve time range
    if args.month_year:
        try:
            m_str, y_str = args.month_year.split("-")
            m, y = int(m_str), int(y_str)
            _, last_day = calendar.monthrange(y, m)
            start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
            end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
            print(f"Querying for {start_dt.strftime('%B %Y')}")
        except Exception:
            print("Invalid --month-year. Use MM-YYYY")
            sys.exit(4)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
        print("Defaulting to last 30 days")

    # Discover dataset for variable
    server = args.server.rstrip("/")
    friendly = args.var_friendly
    # search terms and var keywords heuristics
    heuristics = {
        "Temperature": (["sea surface temperature", "sst", "sea-surface temperature"], ["sea_surface_temperature", "analysed_sst", "sst", "temperature"]),
        "Salinity": (["salinity", "sea surface salinity", "sss"], ["salinity", "sea_surface_salinity", "sss"]),
        "Chlorophyll": (["chlorophyll", "chlor a", "chlor_a", "chlorophyll a", "viirs chl"], ["chlor_a", "chl", "CHL_Weekly", "chlorophyll"])
    }
    search_terms, var_keywords = heuristics[friendly]
    print(f"Discovering datasets for '{friendly}' on {server} ... (query: {' '.join(search_terms)})")
    ds_id, var_guess, note = discover_and_pick_dataset(server, friendly, search_terms, var_keywords)
    if not ds_id:
        print("No valid dataset found automatically. Aborting.")
        print("Note:", note)
        sys.exit(5)
    print("Picked dataset:", ds_id, "| variable guess:", var_guess or "(unspecified)")
    print("Note:", note)

    # Fetch data (variable may be None; fetcher will try to pick the first data column)
    df, debug_info = fetch_point_timeseries(server, ds_id, var_guess, lat, lon, start_dt, end_dt, debug=True)

    # Debug info summary (print a few attempted URLs)
    if debug_info:
        print("Debug attempts (sample):")
        for url, snippet in debug_info[:6]:
            print(" -", url)
    if df.empty:
        print("No data returned for this query. Check debug attempts above.")
        # still create an HTML with debug info
        html_blob = "<h3>No data returned</h3><ul>"
        for u, s in debug_info:
            html_blob += f"<li><pre>{html.escape(u)}\n{html.escape(str(s)[:400])}</pre></li>"
        html_blob += "</ul>"
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"<!doctype html><html><body>{html_blob}</body></html>")
        print(f"Wrote debug HTML to {args.output}")
        sys.exit(0)

    # Make plots & HTML
    out = make_plots_and_html(df, var_guess or friendly, output_filename=args.output)
    print("Saved interactive HTML to:", out)
    print("Open this file in a browser to view charts and table.")


if __name__ == "__main__":
    main()
