#!/usr/bin/env python3
"""
erddap_integration.py

Complete, self-contained ERDDAP helper module that:
 - Discovers & validates datasets on an ERDDAP server for Temperature / Salinity / Chlorophyll.
 - Fetches point time-series (griddap preferred, tabledap fallback).
 - Splits queries across an NRT/historical cutoff if needed.
 - Produces interactive Plotly timeseries + map, and a styled HTML output.
 - Exposes `erddap_streamlit_widget(server=...)` and `register_erddap_blueprint(app, server=...)`
   so other apps (e.g., your `app.py`) can `from erddap_integration import erddap_streamlit_widget`.
 - Also includes a CLI entrypoint.

Usage examples:
  # import into your app:
  from erddap_integration import erddap_streamlit_widget
  erddap_streamlit_widget()  # inside streamlit app

  # CLI:
  python erddap_integration.py --place "Honolulu, Hawaii"
  python erddap_integration.py --latlon "21.3069,-157.8583" --month-year 08-2025 --var-friendly Temperature
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
import logging

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# Defaults & configuration
ERDDAP_SERVER = "https://coastwatch.noaa.gov/erddap"
NRT_CUTOFF_DAYS = 60  # days before 'now' considered historical boundary
HTTP_TIMEOUT = 30

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("erddap_integration")

# -------------------------
# Utility helpers
# -------------------------
def _format_iso(dt: datetime) -> str:
    """Format timezone-aware datetime to ERDDAP-friendly ISO Z string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------
# Geocoding
# -------------------------
def geocode_place(place: str, timeout: int = 10):
    """Resolve place name to (lat, lon). Returns None if cannot resolve."""
    if GEOPY_AVAILABLE:
        try:
            geo = Nominatim(user_agent="erddap_integration")
            loc = geo.geocode(place, timeout=timeout)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            logger.debug("geopy geocode failed; falling back to Nominatim HTTP")
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "erddap_integration/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return float(j[0]["lat"]), float(j[0]["lon"])
    except Exception as e:
        logger.debug("Nominatim HTTP geocode failed: %s", e)
    return None


# -------------------------
# ERDDAP discovery & validation
# -------------------------
def erddap_search(server: str, query: str, items_per_page: int = 200) -> pd.DataFrame:
    """
    Query ERDDAP's search/index.csv endpoint to find datasets matching `query`.
    Returns DataFrame or empty DataFrame on error.
    """
    try:
        q = urllib.parse.quote_plus(query)
        url = f"{server.rstrip('/')}/search/index.csv?searchFor={q}&itemsPerPage={items_per_page}"
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        logger.debug("erddap_search failed (%s). Returning empty DataFrame", e)
        return pd.DataFrame()


def validate_dataset_has_keywords(server: str, dataset_id: str, keywords: list) -> dict:
    """
    Request /info/<dataset_id>/index.json and check for presence of keywords.
    Returns {'valid': bool, 'variables': list, 'info': obj or None}
    """
    try:
        url = f"{server.rstrip('/')}/info/{dataset_id}/index.json"
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        info = r.json()
        txt = str(info).lower()
        found = [k for k in keywords if k.lower() in txt]
        return {"valid": True, "variables": found, "info": info}
    except Exception as e:
        logger.debug("validate_dataset_has_keywords failed for %s: %s", dataset_id, e)
        return {"valid": False, "variables": [], "info": None}


def discover_dataset(server: str, friendly_var: str) -> tuple:
    """
    Discover a dataset id and likely variable name for a friendly variable
    ('Temperature', 'Salinity', 'Chlorophyll').
    Returns (dataset_id or None, var_guess or None, note string).
    """
    heuristics = {
        "Temperature": {
            "search": "sea surface temperature sst sea-surface temperature",
            "keywords": ["sea_surface_temperature", "analysed_sst", "sst", "temperature", "analysed sst"]
        },
        "Salinity": {
            "search": "salinity sea surface salinity sss",
            "keywords": ["salinity", "sea_surface_salinity", "sss"]
        },
        "Chlorophyll": {
            "search": "chlorophyll chlor a chlor_a viirs chl",
            "keywords": ["chlor_a", "chl", "CHL_Weekly", "chlorophyll"]
        }
    }
    if friendly_var not in heuristics:
        return None, None, f"No heuristics for '{friendly_var}'"

    search_terms = heuristics[friendly_var]["search"]
    keywords = heuristics[friendly_var]["keywords"]

    logger.info("Searching ERDDAP server for '%s' datasets...", friendly_var)
    df = erddap_search(server, search_terms)

    candidates = []
    if not df.empty:
        # typical columns include 'Dataset ID' or 'Dataset'
        cols = {c.lower(): c for c in df.columns}
        for possible in ("dataset id", "dataset", "datasetid", "Dataset ID"):
            if possible.lower() in cols:
                idcol = cols[possible.lower()]
                candidates = list(df[idcol].dropna().astype(str).unique())
                break

    # Add curated safe candidates (fallback)
    curated = {
        "Temperature": ["noaacwLEOACSPOSSTL3SnrtCDaily", "jplMURSST41"],
        "Chlorophyll": ["USM_VIIRS_DAP", "noaacwNPPVIIRSchlaGlobal"],
        "Salinity": ["NCOM_amseas_latest3d", "ncom_global"]
    }
    for c in curated.get(friendly_var, []):
        if c not in candidates:
            candidates.append(c)

    # Validate candidates
    for ds in candidates:
        res = validate_dataset_has_keywords(server, ds, keywords)
        if res["valid"]:
            # pick a variable guess if any keyword matched
            var_guess = res["variables"][0] if res["variables"] else None
            note = f"Dataset '{ds}' validated; variable guess: {var_guess}"
            logger.info(note)
            return ds, var_guess, note

    return None, None, "No validated dataset found"


# -------------------------
# Fetchers: griddap and tabledap
# -------------------------
def try_griddap_fetch(server: str, dataset_id: str, variable: str, lat: float, lon: float, start_iso: str, end_iso: str):
    """
    Attempt to fetch griddap CSV point timeseries. Tries lat,lon and lon,lat orders.
    Returns (df, debug_list)
    """
    debug = []
    orders = [("lat", "lon"), ("lon", "lat")]
    for order in orders:
        if order == ("lat", "lon"):
            point_idx = f"[({start_iso}):1:({end_iso})][({lat}):1:({lat})][({lon}):1:({lon})]"
        else:
            point_idx = f"[({start_iso}):1:({end_iso})][({lon}):1:({lon})][({lat}):1:({lat})]"

        query = variable + point_idx if variable else "time,latitude,longitude" + point_idx
        url = f"{server.rstrip('/')}/griddap/{dataset_id}.csv?{query}"
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            snippet = (r.text[:2000] if r.text else f"HTTP {r.status_code}")
            debug.append((url, snippet))
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            # normalize column names
            cols_lower = {c.lower(): c for c in df.columns}
            rename = {}
            for axis in ("time", "latitude", "longitude"):
                if axis in cols_lower:
                    rename[cols_lower[axis]] = axis
            if rename:
                df.rename(columns=rename, inplace=True)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
                df = df.dropna(subset=["time"])
            # if variable present as different name, rename first data column to variable for clarity
            data_cols = [c for c in df.columns if c.lower() not in ("time", "latitude", "longitude")]
            if data_cols and variable and data_cols[0] != variable:
                try:
                    df.rename(columns={data_cols[0]: variable}, inplace=True)
                except Exception:
                    pass
            return df, debug
        except Exception as e:
            debug.append((url, f"EXC:{e}"))
            continue
    return pd.DataFrame(), debug


def try_tabledap_fetch(server: str, dataset_id: str, variable: str, lat: float, lon: float, start_iso: str, end_iso: str):
    """
    Attempt to fetch using tabledap. Returns (df, debug_list).
    """
    debug = []
    # build a tabledap query requesting time,latitude,longitude and variable if given
    var_part = variable + "," if variable else ""
    # Many tabledap endpoints accept "time>=...&time<=..." style filters
    url = f"{server.rstrip('/')}/tabledap/{dataset_id}.csv?{var_part}time&time>={start_iso}&time<={end_iso}&latitude={lat}&longitude={lon}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        snippet = (r.text[:2000] if r.text else f"HTTP {r.status_code}")
        debug.append((url, snippet))
        if r.status_code != 200:
            return pd.DataFrame(), debug
        df = pd.read_csv(io.StringIO(r.text))
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df = df.dropna(subset=["time"])
        return df, debug
    except Exception as e:
        debug.append((url, f"EXC:{e}"))
        return pd.DataFrame(), debug


# -------------------------
# High-level point timeseries fetch (handles cutoff splitting)
# -------------------------
def fetch_point_timeseries(server: str, dataset_id: str, variable: str, lat: float, lon: float, start_dt: datetime, end_dt: datetime):
    """
    Fetch point timeseries. Splits request if spans NRT_CUTOFF_DAYS.
    Returns (df, debug_list)
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
        # try griddap first
        df_g, dbg_g = try_griddap_fetch(server, dataset_id, variable, lat, lon, s_iso, e_iso)
        debug.extend(dbg_g)
        if not df_g.empty:
            all_dfs.append(df_g)
            continue
        # fallback to tabledap
        df_t, dbg_t = try_tabledap_fetch(server, dataset_id, variable, lat, lon, s_iso, e_iso)
        debug.extend(dbg_t)
        if not df_t.empty:
            all_dfs.append(df_t)

    if not all_dfs:
        return pd.DataFrame(), debug
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    if "time" in df_all.columns:
        df_all = df_all.sort_values("time").reset_index(drop=True)
    return df_all, debug


# -------------------------
# Plot generation + HTML output
# -------------------------
def timeseries_plot(df: pd.DataFrame):
    """Return a Plotly Figure for the main timeseries (if available)."""
    if df.empty:
        return go.Figure().update_layout(title="No data available")
    data_cols = [c for c in df.columns if c.lower() not in ("time", "latitude", "longitude")]
    varcol = data_cols[0] if data_cols else None
    if not varcol or "time" not in df.columns:
        return go.Figure().update_layout(title="No timeseries variable found")
    fig = px.line(df, x="time", y=varcol, title=f"Timeseries: {varcol}")
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def map_scatter_plot(df: pd.DataFrame):
    """Return a Plotly geo scatter for the latest point if lat/lon available."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns or "time" not in df.columns:
        return go.Figure().update_layout(title="No spatial data")
    latest_idx = df["time"].idxmax()
    df_latest = df.loc[[latest_idx]]
    data_cols = [c for c in df.columns if c.lower() not in ("time", "latitude", "longitude")]
    varcol = data_cols[0] if data_cols else None
    if varcol:
        fig = px.scatter_geo(df_latest, lat="latitude", lon="longitude", hover_name=varcol, hover_data=["time"], title=f"Latest: {varcol}")
    else:
        fig = px.scatter_geo(df_latest, lat="latitude", lon="longitude", hover_data=["time"], title="Latest location")
    return fig


def make_html_output(df: pd.DataFrame, variable_label: str, output_filename: str = "erddap_output.html"):
    """
    Create an HTML file with Plotly charts and a styled data table (Bootstrap).
    Returns output_filename.
    """
    figs = []
    if df.empty:
        figs.append(go.Figure().update_layout(title="No data available"))
        html_table = "<p>No data returned for this query.</p>"
    else:
        fig_ts = timeseries_plot(df)
        figs.append(fig_ts)
        fig_map = map_scatter_plot(df)
        figs.append(fig_map)
        # create a styled table using pandas Styler with Bootstrap classes
        try:
            df_display = df.copy()
            # round numeric columns for display
            for c in df_display.select_dtypes(include=["float", "int"]).columns:
                df_display[c] = df_display[c].round(6)
            styler = df_display.style.hide_index().set_table_attributes('class="table table-striped"') \
                .set_caption(f"Data table (rows: {len(df_display)})")
            html_table = styler.render()
        except Exception:
            html_table = df.head(200).to_html(index=False)

    # Build HTML fragments for each figure
    fragments = [pio.to_html(fig, include_plotlyjs=False, full_html=False) for fig in figs]

    html_head = """<!doctype html><html><head><meta charset="utf-8"/>
    <title>ERDDAP Output</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{margin:20px;font-family:Segoe UI,Roboto,Arial;background:#f7f7f7;} .card{margin-bottom:1rem;}</style>
    </head><body><div class="container"><h2>ERDDAP Query Results</h2>"""
    html_tail = """<hr/><p>Generated by erddap_integration.py</p></div></body></html>"""

    body = []
    for frag in fragments:
        body.append(f'<div class="card"><div class="card-body">{frag}</div></div>')
    body.append(f'<div class="card"><div class="card-body"><h5>Data table</h5>{html_table}</div></div>')
    html_all = html_head + "\n".join(body) + html_tail

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_all)
    logger.info("Wrote output HTML to %s", output_filename)
    return output_filename


# -------------------------
# Streamlit & Flask helpers (lazy imports)
# -------------------------
def erddap_streamlit_widget(server: str = ERDDAP_SERVER):
    """
    Streamlit widget to embed in a Streamlit app.
    Call this function from inside a streamlit app (after `import streamlit as st`).
    """
    try:
        import streamlit as st
    except Exception:
        logger.error("Streamlit not installed; erddap_streamlit_widget requires streamlit.")
        return

    st.sidebar.header("ERDDAP Ocean Data")
    var_choice = st.sidebar.selectbox("Variable", ["Temperature", "Salinity", "Chlorophyll"])
    place = st.sidebar.text_input("Place (e.g., 'Honolulu, Hawaii')")
    manual_latlon = st.sidebar.text_input("Or lat,lon (e.g., '21.3,-157.8')")
    month_year = st.sidebar.text_input("Month-Year (MM-YYYY) optional", placeholder="08-2025")
    server_input = st.sidebar.text_input("ERDDAP server", value=server)

    if st.sidebar.button("Fetch Data"):
        # Resolve lat/lon
        latlon = None
        if manual_latlon:
            try:
                lat, lon = [float(p.strip()) for p in manual_latlon.split(",")]
                latlon = (lat, lon)
            except Exception:
                st.error("Invalid lat,lon format. Use 'lat,lon'")
                return
        elif place:
            with st.spinner("Geocoding..."):
                latlon = geocode_place(place)
                if not latlon:
                    st.error("Geocoding failed.")
                    return
        else:
            st.warning("Provide a place or lat,lon.")
            return

        lat, lon = latlon

        # Resolve time
        if month_year:
            try:
                m, y = [int(x) for x in month_year.split("-")]
                _, last_day = calendar.monthrange(y, m)
                start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
                end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
            except Exception:
                st.error("Invalid Month-Year format. Use MM-YYYY.")
                return
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)

        st.info(f"Discovering dataset for {var_choice} on {server_input} ...")
        ds_id, var_guess, note = discover_dataset(server_input, var_choice)
        if not ds_id:
            st.error("No valid dataset found: " + note)
            return

        st.write(f"Using dataset: `{ds_id}` (variable guess: `{var_guess}`)")
        with st.spinner("Fetching data..."):
            df, debug = fetch_point_timeseries(server_input, ds_id, var_guess, lat, lon, start_dt, end_dt)

        if df.empty:
            st.error("No data returned. See debug info below.")
            st.subheader("Debug attempts")
            for url, snippet in debug[:10]:
                st.markdown(f"**URL:** `{url}`")
                st.code(snippet[:1000])
            return

        st.subheader("Timeseries")
        st.plotly_chart(timeseries_plot(df), use_container_width=True)

        st.subheader("Latest point on map")
        st.plotly_chart(map_scatter_plot(df), use_container_width=True)

        st.subheader("Data table")
        # show pandas dataframe in streamlit
        st.dataframe(df.head(200))

        # allow saving to HTML
        if st.button("Save interactive HTML"):
            out = make_html_output(df, var_guess or var_choice, output_filename="erddap_output.html")
            st.success(f"Saved to {out}")


def register_erddap_blueprint(app, server: str = ERDDAP_SERVER):
    """
    Register a Flask blueprint at /erddap that provides a simple form and renders
    Plotly figures inline.
    """
    try:
        from flask import Blueprint, request, render_template_string
    except Exception:
        logger.error("Flask not installed; cannot register blueprint.")
        return

    bp = Blueprint("erddap_integration", __name__)
    FORM_HTML = """
    <!doctype html><html><head><meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{font-family:sans-serif;padding:1rem;} .card{margin-bottom:1rem;}</style>
    </head><body>
    <h3>ERDDAP Query</h3>
    <form method="post">
      Variable:
      <select name="variable">
        {% for v in variables %}
          <option value="{{v}}" {% if v==selected_var %}selected{% endif %}>{{v}}</option>
        {% endfor %}
      </select><br>
      Place name: <input name="place" value="{{place}}"><br>
      Or lat,lon: <input name="latlon" value="{{latlon}}"><br>
      Month-Year (MM-YYYY): <input name="month_year" value="{{month_year}}"><br>
      <button type="submit">Fetch</button>
    </form>
    <div id="info">{{info|safe}}</div>
    <div id="charts">{{charts|safe}}</div>
    </body></html>
    """

    @bp.route("/erddap", methods=["GET", "POST"])
    def erddap_form():
        info_html = ""
        charts_html = ""
        form_data = {"variables": ["Temperature", "Salinity", "Chlorophyll"],
                     "selected_var": "Temperature", "place": "", "latlon": "", "month_year": ""}
        if request.method == "POST":
            var_choice = request.form.get("variable")
            place = request.form.get("place", "").strip()
            manual_latlon = request.form.get("latlon", "").strip()
            month_year = request.form.get("month_year", "").strip()
            form_data.update({"selected_var": var_choice, "place": place, "latlon": manual_latlon, "month_year": month_year})

            # Resolve location
            latlon = None
            if manual_latlon:
                try:
                    lat, lon = [float(x.strip()) for x in manual_latlon.split(",")]
                    latlon = (lat, lon)
                except Exception:
                    info_html = "<p style='color:red;'>Invalid lat,lon format.</p>"
                    return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)
            elif place:
                latlon = geocode_place(place)
                if not latlon:
                    info_html = "<p style='color:red;'>Geocoding failed.</p>"
                    return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)
            else:
                info_html = "<p style='color:orange;'>Provide a place or lat,lon.</p>"
                return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)

            lat, lon = latlon

            # Resolve time
            if month_year:
                try:
                    m, y = [int(x) for x in month_year.split("-")]
                    _, last_day = calendar.monthrange(y, m)
                    start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
                    end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
                except Exception:
                    info_html = "<p style='color:red;'>Invalid Month-Year format.</p>"
                    return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)
            else:
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=30)

            # Discover dataset and fetch
            ds_id, var_guess, note = discover_dataset(server, var_choice)
            if not ds_id:
                info_html = f"<p style='color:red;'>No dataset found: {note}</p>"
                return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)

            df, debug = fetch_point_timeseries(server, ds_id, var_guess, lat, lon, start_dt, end_dt)
            if df.empty:
                info_html = "<p style='color:red;'>No data returned. See debug above.</p>"
                dbg_html = "<pre>" + html.escape("\n".join(u for u, _ in debug[:10])) + "</pre>"
                charts_html = dbg_html
            else:
                fig_ts = timeseries_plot(df)
                fig_map = map_scatter_plot(df)
                # embed two plots; use Plotly JSON for safety
                plots_js = f"<div id='plot1'></div><div id='plot2'></div><script>Plotly.newPlot('plot1', {fig_ts.to_json()});Plotly.newPlot('plot2', {fig_map.to_json()});</script>"
                # minimal table
                table_html = df.head(200).to_html(index=False)
                charts_html = plots_js + "<h4>Data sample</h4>" + table_html

        return render_template_string(FORM_HTML, **form_data, info=info_html, charts=charts_html)

    app.register_blueprint(bp)


# -------------------------
# CLI helpers & entrypoint
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(prog="erddap_integration.py")
    p.add_argument("--server", default=ERDDAP_SERVER, help="ERDDAP server URL")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--place", type=str, help="Place name")
    group.add_argument("--latlon", type=str, help="Manual lat,lon (e.g., '21.3,-157.8')")
    p.add_argument("--month-year", dest="month_year", type=str, help="MM-YYYY optional")
    p.add_argument("--var-friendly", dest="var_friendly", choices=["Temperature", "Salinity", "Chlorophyll"], default="Temperature")
    p.add_argument("--output", dest="output", default="erddap_output.html", help="Output HTML filename")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Resolve lat/lon
    latlon = None
    if args.latlon:
        try:
            lat, lon = [float(x.strip()) for x in args.latlon.split(",")]
            latlon = (lat, lon)
        except Exception:
            logger.error("Invalid --latlon format. Use lat,lon")
            sys.exit(2)
    elif args.place:
        logger.info("Geocoding '%s'...", args.place)
        latlon = geocode_place(args.place)
    if not latlon:
        logger.error("Could not resolve location.")
        sys.exit(3)
    lat, lon = latlon
    logger.info("Using coordinates: %s, %s", lat, lon)

    # Resolve time range
    if args.month_year:
        try:
            m, y = [int(x) for x in args.month_year.split("-")]
            _, last_day = calendar.monthrange(y, m)
            start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
            end_dt = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
        except Exception:
            logger.error("Invalid --month-year. Use MM-YYYY")
            sys.exit(4)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)

    server = args.server.rstrip("/")
    friendly = args.var_friendly
    logger.info("Discovering dataset for %s on %s", friendly, server)
    ds_id, var_guess, note = discover_dataset(server, friendly)
    if not ds_id:
        logger.error("No validated dataset found: %s", note)
        sys.exit(5)
    logger.info("Using dataset %s (var guess: %s)", ds_id, var_guess)

    df, debug = fetch_point_timeseries(server, ds_id, var_guess, lat, lon, start_dt, end_dt)
    if df.empty:
        logger.warning("No data returned. Writing debug HTML.")
        debug_html = "<h3>No data returned</h3><ul>"
        for u, s in debug:
            debug_html += f"<li><pre>{html.escape(u)}\n{html.escape(str(s)[:400])}</pre></li>"
        debug_html += "</ul>"
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"<!doctype html><html><body>{debug_html}</body></html>")
        logger.info("Wrote debug output to %s", args.output)
        sys.exit(0)

    out = make_html_output(df, var_guess or friendly, output_filename=args.output)
    logger.info("Saved interactive HTML to: %s", out)
    logger.info("Open the file in a browser to view charts and table.")


if __name__ == "__main__":
    main()
```0
