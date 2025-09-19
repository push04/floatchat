"""
ERDDAP integration helper for floatchat/app.py

USAGE (paste these lines into your repo):
1) Create a new file at the root of the project: `erddap_integration.py` and paste this whole file there.
2) Install required packages if not present:
   pip install pandas requests plotly geopy

3a) If your app is a Streamlit app: open app.py and *only* add the following two lines near the top (after other imports):
   from erddap_integration import erddap_streamlit_widget

   And then inside your main UI rendering function (where the Streamlit widgets live), add a single call:
   erddap_streamlit_widget()

3b) If your app is a Flask (or other WSGI) app: add these two lines near app initialization:
   from erddap_integration import register_erddap_blueprint
   register_erddap_blueprint(app)

IMPORTANT: This module only *adds* new functions/routes/UI. It does not modify any of your existing functions.

This file is a corrected and more robust version of the helper you provided. It contains:
 - fixes for indentation and URL/time formatting bugs
 - safer dataset/variable probing using the dataset index CSV
 - attempts with both lat/lon ordering when fetching griddap data
 - normalization of common lat/lon/time column names
 - clearer debug output when fetch fails

If you want further customization (e.g. support for tabledap-only datasets, or more permissive variable name matching), tell me which datasets you use and I'll tune it.
"""

from datetime import datetime, timedelta
import io
import re
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional geocoding (install geopy). If not installed, we fall back to Nominatim HTTP.
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

ERDDAP_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
# small bbox half-width (deg) to capture nearest grid cell
BBOX_HALF_DEG = 0.125


def get_dataset_variables(server, dataset_id, timeout=20):
    """
    Query ERDDAP info page CSV for a dataset and return a list of variable names (destination names).
    Returns list of strings (may be empty) or raises on network errors.
    """
    try:
        url = f"{server}/info/{dataset_id}/index.csv"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Look for a column that contains variable/Variable name
        cols = [c for c in df.columns if 'variable' in c.lower() and 'name' in c.lower()]
        if not cols:
            # often the index.csv has a column 'Variable Name' or 'destinationName'
            cols = [c for c in df.columns if 'destination' in c.lower()]
        if not cols:
            return []
        var_col = cols[0]
        vars_list = df[var_col].dropna().astype(str).unique().tolist()
        vars_list = [v.strip() for v in vars_list if v.strip() and len(v.strip()) < 200]
        return vars_list
    except Exception as e:
        print(f"[get_dataset_variables] failed: {e}")
        return []


def get_preferred_for(friendly_variable):
    """Return preferred dataset list (dataset_id, varname) for a friendly variable."""
    return PREFERRED_DATASETS.get(friendly_variable, [])


# -------------------------
# Utils
# -------------------------
def _normalize_colname(s: str) -> str:
    """Normalize column names to compare easily (remove non-alphanum, lower)."""
    return re.sub(r'\W+', '', str(s)).lower()


# -------------------------
# Helper: discover datasets
# -------------------------
def discover_erddap_datasets(server=ERDDAP_SERVER, keyword="sst", max_results=20):
    """
    Search ERDDAP server for datasets matching keyword.
    Returns a pandas.DataFrame standardized to have columns: dataset_id, title (if available).
    Falls back to returning the raw DataFrame if parsing fails.
    """
    try:
        q = requests.utils.requote_uri(f"{server}/search/index.csv?searchFor={keyword}&itemsPerPage={max_results}")
        r = requests.get(q, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))

        # Normalize column names and look for dataset-id like columns
        col_map = { _normalize_colname(c): c for c in df.columns }
        candidates = ['datasetid', 'dataset_id', 'dataset', 'datasetname']
        dataset_col = None
        for cand in candidates:
            if cand in col_map:
                dataset_col = col_map[cand]
                break
        title_col = None
        for cand in ['title', 'datasettitle', 'titletext']:
            if cand in col_map:
                title_col = col_map[cand]
                break

        if dataset_col:
            df = df.copy()
            df.rename(columns={dataset_col: 'dataset_id'}, inplace=True)
            if title_col:
                df.rename(columns={title_col: 'title'}, inplace=True)
                return df[['dataset_id', 'title']].drop_duplicates().reset_index(drop=True)
            return df[['dataset_id']].drop_duplicates().reset_index(drop=True)

        # If we couldn't find a dataset id column, return raw df (caller will handle empty case)
        return df

    except Exception as e:
        print("ERDDAP discover failed:", e)
        return pd.DataFrame()


# -------------------------
# Helper: basic geocode (geopy optional, otherwise HTTP Nominatim)
# -------------------------
def geocode_place(place):
    """
    Return (lat, lon) from place name. Returns None on failure.
    """
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent="floatchat_erddap_integration")
            loc = geolocator.geocode(place, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception as e:
            print("Geopy geocode failed:", e)

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1, "addressdetails": 0}
        headers = {"User-Agent": "floatchat_erddap_integration/1.0 (your_email@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and len(j) > 0:
            lat = float(j[0].get("lat"))
            lon = float(j[0].get("lon"))
            return lat, lon
    except Exception as e:
        print("Geocode failed (nominatim):", e)

    return None


# -------------------------
# Helper: fetch griddap point timeseries
# -------------------------
def fetch_griddap_point_timeseries(server, dataset_id, variable, lat, lon, end_date=None, days=7, start_year=None, end_year=None, debug=False):
    """
    Robust fetch supporting either day-window (legacy) or year-range (preferred).
    - If start_year and end_year are provided, uses Jan 1 start_year -> Dec 31 end_year.
    - Otherwise uses 'days' back from end_date (legacy behaviour).

    The function will:
      * try to probe dataset / variable existence using /info/{dataset}/index.csv
      * try a small bbox around the requested lat/lon with progressively larger bboxes
      * attempt both lat/lon ordering variants (some datasets use lon,lat order)

    Returns a DataFrame or (DataFrame, raw_texts) when debug=True.
    """
    raw_texts = []

    # Determine date window
    if start_year is not None and end_year is not None:
        start_date = datetime(int(start_year), 1, 1).date()
        end_date_obj = datetime(int(end_year), 12, 31).date()
    else:
        if end_date is None:
            end_date_obj = datetime.utcnow().date()
        elif isinstance(end_date, datetime):
            end_date_obj = end_date.date()
        else:
            end_date_obj = end_date
        start_date = end_date_obj - timedelta(days=days - 1)

    # Convert to ISO strings with time to be safer for griddap
    start = start_date.isoformat() + "T00:00:00Z"
    end = end_date_obj.isoformat() + "T00:00:00Z"

    # Get variable list from dataset info to know dimension ordering (best-effort)
    try:
        dataset_vars = get_dataset_variables(server, dataset_id)
    except Exception:
        dataset_vars = []

    # Try multiple bbox sizes if single bbox returns empty (progressively larger)
    attempt_bboxes = [BBOX_HALF_DEG, 0.25, 0.5, 1.0, 2.0]

    # For some griddap datasets, dimension ordering is (time, lat, lon) or (time, lon, lat).
    # We'll try both ordering variants.
    orders = [ ("lat", "lon"), ("lon", "lat") ]

    for hb in attempt_bboxes:
        lat_min = lat - hb
        lat_max = lat + hb
        lon_min = lon - hb
        lon_max = lon + hb

        for order in orders:
            if order == ("lat", "lon"):
                bounds = f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
            else:
                # lon, lat ordering
                bounds = f"[({lon_min}):1:({lon_max})][({lat_min}):1:({lat_max})]"

            url = (
                f"{server}/griddap/{dataset_id}.csv?{variable}"
                f"[({start}):1:({end})]"
                f"{bounds}"
            )

            try:
                r = requests.get(url, timeout=90)
                raw_texts.append((url, r.text[:50000] if isinstance(r.text, str) else str(r.status_code)))
                if r.status_code != 200:
                    continue
                df = pd.read_csv(io.StringIO(r.text))

                # Normalize common column names
                col_lower = {c.lower(): c for c in df.columns}
                if 'time' in col_lower:
                    df.rename(columns={col_lower['time']: 'time'}, inplace=True)
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                # lat/lon normalization
                if 'latitude' in col_lower:
                    df.rename(columns={col_lower['latitude']: 'latitude'}, inplace=True)
                elif 'lat' in col_lower:
                    df.rename(columns={col_lower['lat']: 'latitude'}, inplace=True)
                elif 'y' in col_lower:
                    df.rename(columns={col_lower['y']: 'latitude'}, inplace=True)

                if 'longitude' in col_lower:
                    df.rename(columns={col_lower['longitude']: 'longitude'}, inplace=True)
                elif 'lon' in col_lower:
                    df.rename(columns={col_lower['lon']: 'longitude'}, inplace=True)
                elif 'x' in col_lower:
                    df.rename(columns={col_lower['x']: 'longitude'}, inplace=True)

                # Some griddap outputs name value column as '<varname>' or '<varname>_analysis'; handle generically.
                # If variable column is not present, pick the first non-lat/lon/time column.
                non_geo_cols = [c for c in df.columns if c not in ['time', 'latitude', 'longitude']]
                if not non_geo_cols:
                    # nothing useful here
                    continue

                # Drop rows with NaN time
                if 'time' in df.columns:
                    df = df.dropna(subset=['time'])

                if not df.empty:
                    if debug:
                        return df, raw_texts
                    return df

            except Exception as e:
                raw_texts.append((url, f"EXCEPTION: {e}"))
                continue

    # As a last attempt, try a single-point query (no bbox ranges) using the exact lat/lon
    for order in orders:
        lat_q = lat
        lon_q = lon
        if order == ("lat", "lon"):
            point_idx = f"[({lat_q}):1:({lat_q})][({lon_q}):1:({lon_q})]"
        else:
            point_idx = f"[({lon_q}):1:({lon_q})][({lat_q}):1:({lat_q})]"
        url = f"{server}/griddap/{dataset_id}.csv?{variable}[({start}):1:({end})]{point_idx}"
        try:
            r = requests.get(url, timeout=90)
            raw_texts.append((url, r.text[:50000] if isinstance(r.text, str) else str(r.status_code)))
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            # normalize lat/lon as above
            col_lower = {c.lower(): c for c in df.columns}
            if 'latitude' in col_lower:
                df.rename(columns={col_lower['latitude']: 'latitude'}, inplace=True)
            elif 'lat' in col_lower:
                df.rename(columns={col_lower['lat']: 'latitude'}, inplace=True)
            if 'longitude' in col_lower:
                df.rename(columns={col_lower['longitude']: 'longitude'}, inplace=True)
            elif 'lon' in col_lower:
                df.rename(columns={col_lower['lon']: 'longitude'}, inplace=True)

            if not df.empty:
                if debug:
                    return df, raw_texts
                return df
        except Exception as e:
            raw_texts.append((url, f"EXCEPTION: {e}"))
            continue

    # If we reached here, no data found in any bbox attempt
    if debug:
        return pd.DataFrame(), raw_texts
    return pd.DataFrame()


# -------------------------
# Helper: choose dataset & variable mapping
# -------------------------
# Friendly variable -> search keywords and common variable names to try
DEFAULT_VARIABLES = {
    'Temperature': {'search_kw': 'sst OR sea surface temperature OR sea_surface_temperature', 'var_names': ['sst', 'analysed_sst', 'sea_surface_temperature', 'temperature']},
    'Salinity': {'search_kw': 'salinity OR sss OR sea surface salinity', 'var_names': ['sss', 'salinity', 'sea_surface_salinity']},
    'Chlorophyll': {'search_kw': 'chlorophyll OR chlor_a', 'var_names': ['chlor_a', 'chlorophyll', 'chl']},
}

# ---------- Preferred datasets for common project needs ----------
# These are examples that tend to exist on many ERDDAP servers. If a server uses different IDs,
# you can override using the Streamlit override input box.
PREFERRED_DATASETS = {
    'Temperature': [
        # NOAA OISST / AVHRR style datasets (dataset ids vary across ERDDAP servers)
        ('ncdcOisst', 'sst'),
        ('ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon', 'sst'),
    ],
    'Salinity': [
        ('NCOM_amseas_latest3d', 'salinity'),
    ],
    'Chlorophyll': [
        ('USM_VIIRS_DAP', 'chlor_a'),
    ],
}


def pick_dataset_and_var(server, friendly_variable):
    """
    Try to discover a dataset and a plausible variable name for the friendly_variable.
    Returns (dataset_id, variable_name) or (None, None).

    Strategy:
      1. Try the PREFERRED_DATASETS list for the friendly variable (quick "probe" by reading index.csv)
      2. If not found, run discover_erddap_datasets() and probe the first few returned datasets
      3. Use get_dataset_variables() to check for available variable names and choose a best match
    """
    # guard
    if friendly_variable not in DEFAULT_VARIABLES:
        return None, None

    info = DEFAULT_VARIABLES[friendly_variable]

    # 1) Try preferred datasets first
    prefs = PREFERRED_DATASETS.get(friendly_variable, [])
    for ds_id, varname in prefs:
        try:
            vars_list = get_dataset_variables(server, ds_id)
            if not vars_list:
                continue
            # Normalize names and check if requested variable exists
            vars_lower = [v.lower() for v in vars_list]
            if varname.lower() in vars_lower:
                return ds_id, varname
            # also accept any of our candidate alternative var names
            for candidate in info['var_names']:
                if candidate.lower() in vars_lower:
                    return ds_id, candidate
        except Exception:
            continue

    # 2) Discover datasets via search
    df = discover_erddap_datasets(server, info['search_kw'])
    if df.empty:
        # try simpler keyword
        df = discover_erddap_datasets(server, friendly_variable.lower())
    if df.empty:
        return None, None

    # Probe top N discovered datasets to find a matching variable
    probe_n = min(6, len(df))
    for i in range(probe_n):
        dataset_id = df.iloc[i]['dataset_id'] if 'dataset_id' in df.columns else df.iloc[i].iloc[0]
        try:
            vars_list = get_dataset_variables(server, dataset_id)
            if not vars_list:
                continue
            vars_lower = [v.lower() for v in vars_list]
            # look for best candidate among our preferred names
            for candidate in info['var_names']:
                if candidate.lower() in vars_lower:
                    # return the actual variable name as appears in the dataset (case preserved)
                    matched = [v for v in vars_list if v.lower() == candidate.lower()][0]
                    return dataset_id, matched
            # fallback: return first non-dimension variable
            # pick the longest variable name that looks like a data var (not 'latitude'/'time')
            non_geo = [v for v in vars_list if v.lower() not in ('latitude', 'longitude', 'time', 'depth')]
            if non_geo:
                return dataset_id, non_geo[0]
        except Exception:
            continue

    return None, None


# -------------------------
# Plot helpers
# -------------------------
def timeseries_plot(df, variable):
    """Create a Plotly figure for a small timeseries DataFrame."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='No data')
        return fig
    # Find the variable column (exclude latitude longitude time)
    candidates = [c for c in df.columns if c not in ['time', 'latitude', 'longitude']]
    varcol = variable if variable in df.columns else (candidates[0] if candidates else None)
    if varcol is None:
        fig = go.Figure()
        fig.update_layout(title='Variable not found in fetched data')
        return fig
    # ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    fig = px.line(df.sort_values('time'), x='time', y=varcol, title=f"{varcol} @ {df['latitude'].median():.3f},{df['longitude'].median():.3f}")
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def map_scatter_plot(df, variable):
    """Create a small scatter map (lat/lon colored by variable)."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='No data')
        return fig
    candidates = [c for c in df.columns if c not in ['time', 'latitude', 'longitude']]
    varcol = variable if variable in df.columns else (candidates[0] if candidates else None)
    if varcol is None:
        fig = go.Figure()
        fig.update_layout(title='Variable not found')
        return fig
    # Use latest time slice
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        latest_time = df['time'].max()
        df_latest = df[df['time'] == latest_time]
    else:
        df_latest = df
        latest_time = None

    # make sure lat/lon exist
    if 'latitude' not in df_latest.columns or 'longitude' not in df_latest.columns:
        fig = go.Figure()
        fig.update_layout(title='No spatial coordinates in data')
        return fig

    fig = px.scatter_geo(df_latest, lat='latitude', lon='longitude', size_max=8,
                         hover_name=varcol, hover_data=['time', 'latitude', 'longitude'],
                         projection='natural earth')
    title = f"Map of {varcol}"
    if latest_time is not None:
        title += f" at {latest_time}"
    fig.update_layout(title=title)
    return fig


# -------------------------
# Streamlit widget (drop-in)
# -------------------------
def erddap_streamlit_widget(server=ERDDAP_SERVER):
    """If your app is Streamlit, call this function in your main UI to mount the ERDDAP widget.

    It creates an options select (Temperature/Salinity/Chlorophyll), a place text input (or manual lat,lon),
    timeframe selector (years), and renders Plotly charts inline.
    """
    try:
        import streamlit as st
    except Exception:
        print('Streamlit not available. This widget is for Streamlit apps.')
        return

    st.sidebar.header('Ocean data (ERDDAP)')
    var_choice = st.sidebar.selectbox('Variable', list(DEFAULT_VARIABLES.keys()))
    # Show recommended datasets for this variable (helps user choose overrides quickly)
    prefs = get_preferred_for(var_choice)
    if prefs:
        st.sidebar.markdown("**Recommended datasets:**")
        for ds, vn in prefs:
            st.sidebar.caption(f"{ds}  —  try var `{vn}`")

    place = st.sidebar.text_input('Place name (or leave blank to enter lat,lon manually)')
    manual_latlon = st.sidebar.text_input('Manual lat,lon (e.g. "20.5,70.2")')
    current_year = datetime.utcnow().year
    year_range = st.sidebar.slider('Year range', min_value=1980, max_value=current_year, value=(current_year-1, current_year), step=1)
    start_year, end_year = year_range
    server_input = st.sidebar.text_input('ERDDAP server', value=server)

    if st.sidebar.button('Fetch'):
        # resolve lat lon
        latlon = None
        if manual_latlon and manual_latlon.strip():
            try:
                parts = [p.strip() for p in manual_latlon.split(',')]
                latlon = (float(parts[0]), float(parts[1]))
            except Exception:
                st.error('Manual lat,lon parse failed. Use format: lat,lon')
                return
        elif place and place.strip():
            with st.spinner('Geocoding place...'):
                g = geocode_place(place)
                if g is None:
                    st.warning('Geocoding failed or not available. Please enter manual lat,lon.')
                else:
                    latlon = g
        else:
            st.info('Enter a place name OR manual lat,lon to fetch data.')
            return

        if latlon is None:
            st.stop()

        lat, lon = latlon

        # Optional manual overrides for dataset & var (very useful when auto-pick fails)
        st.markdown("**Optional overrides (use if automatic discovery returns no data):**")
        manual_dataset = st.text_input('Override dataset_id (leave blank to use auto-discovery)')
        manual_var = st.text_input('Override variable name (leave blank to use auto-detected var)')

        with st.spinner('Discovering dataset...'):
            if manual_dataset and manual_dataset.strip():
                dataset_id = manual_dataset.strip()
                if manual_var and manual_var.strip():
                    varname = manual_var.strip()
                else:
                    # try to auto-detect a variable in the user-supplied dataset
                    vars_list = get_dataset_variables(server_input, dataset_id)
                    if vars_list:
                        # pick first candidate matching our desired list
                        candidate = None
                        for c in DEFAULT_VARIABLES[var_choice]['var_names']:
                            if c.lower() in [v.lower() for v in vars_list]:
                                candidate = [v for v in vars_list if v.lower() == c.lower()][0]
                                break
                        varname = candidate or vars_list[0]
                    else:
                        varname = DEFAULT_VARIABLES[var_choice]['var_names'][0]
            else:
                dataset_id, varname = pick_dataset_and_var(server_input, var_choice)

        if dataset_id is None:
            st.error('Could not find a suitable dataset on ERDDAP. Please enter dataset_id and variable manually.')
            return

        st.write(f"Using dataset: **{dataset_id}** and variable **{varname}**")

        with st.spinner('Fetching timeseries for years %d - %d (may try larger bbox)...' % (start_year, end_year)):
            result = fetch_griddap_point_timeseries(server_input, dataset_id, varname, lat, lon, start_year=start_year, end_year=end_year, debug=True)
            if isinstance(result, tuple):
                df, raw_texts = result
            else:
                df = result
                raw_texts = []

        if df.empty:
            st.error('No data returned for that dataset/variable/point/time. Showing debug info below.')
            for i, (u, t) in enumerate(raw_texts[:8]):
                st.markdown(f"**Attempt {i+1}**: `{u}`")
                st.code(t[:4000])
            st.info('Try entering a different dataset_id, a different variable name (e.g. sst, sss), a nearby lat/lon, or expand the year range.')
            return

        st.subheader('Timeseries')
        fig_ts = timeseries_plot(df, varname)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader('Map (latest slice)')
        fig_map = map_scatter_plot(df, varname)
        st.plotly_chart(fig_map, use_container_width=True)

        # Quick stats
        try:
            latest = df.loc[df['time'].idxmax()]
            val = None
            for c in df.columns:
                if c not in ['time', 'latitude', 'longitude']:
                    val = latest[c]
                    break
            if val is not None and (isinstance(val, (int, float)) or hasattr(val, '__float__')):
                st.metric(label=f"Latest {varname}", value=f"{float(val):.3f}")
        except Exception:
            pass


# -------------------------
# Flask blueprint (optional)
# -------------------------
def register_erddap_blueprint(app, server=ERDDAP_SERVER):
    """Register a small Flask blueprint at /erddap for interactive queries.

    This adds routes GET/POST /erddap with a simple form and embedded Plotly charts.
    """
    try:
        from flask import Blueprint, request, render_template_string
    except Exception:
        print('Flask not available or not requested; skipping blueprint registration')
        return

    bp = Blueprint('erddap_integration', __name__, template_folder='templates')

    FORM_HTML = """
    <!doctype html>
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <title>ERDDAP Query</title>
    </head>
    <body>
      <h3>ERDDAP Query</h3>
      <form method="post">
        Variable: <select name="variable"><option>Temperature</option><option>Salinity</option><option>Chlorophyll</option></select><br>
        Place name: <input name="place" placeholder="e.g. Mumbai, India"><br>
        Or lat,lon: <input name="latlon" placeholder="20.5,70.2"><br>
        Days: <input name="days" value="7"><br>
        <button type="submit">Fetch</button>
      </form>
      <div id="charts">{{charts|safe}}</div>
    </body>
    </html>
    """

    @bp.route('/erddap', methods=['GET', 'POST'])
    def erddap_form():
        charts_html = ''
        if request.method == 'POST':
            var_choice = request.form.get('variable')
            place = request.form.get('place','').strip()
            manual = request.form.get('latlon','').strip()
            days = int(request.form.get('days') or 7)

            latlon = None
            if manual:
                try:
                    p = [x.strip() for x in manual.split(',')]
                    latlon = (float(p[0]), float(p[1]))
                except Exception:
                    latlon = None
            elif place:
                g = geocode_place(place)
                if g:
                    latlon = g

            if not latlon:
                charts_html = '<p>Could not geocode place. Provide manual lat,lon in the form.</p>'
                return render_template_string(FORM_HTML, charts=charts_html)

            lat, lon = latlon
            dataset_id, varname = pick_dataset_and_var(server, var_choice)
            if not dataset_id:
                charts_html = '<p>Could not find dataset automatically; try another variable.</p>'
                return render_template_string(FORM_HTML, charts=charts_html)

            df = fetch_griddap_point_timeseries(server, dataset_id, varname, lat, lon, days=days)
            if df.empty:
                charts_html = '<p>No data returned for that query.</p>'
                return render_template_string(FORM_HTML, charts=charts_html)

            fig_ts = timeseries_plot(df, varname)
            fig_map = map_scatter_plot(df, varname)
            charts_html = '<div id="ts"></div><div id="map"></div>'
            charts_html += f"<script>var fig_ts = {fig_ts.to_json()}; Plotly.newPlot('ts', fig_ts.data, fig_ts.layout);</script>"
            charts_html += f"<script>var fig_map = {fig_map.to_json()}; Plotly.newPlot('map', fig_map.data, fig_map.layout);</script>"

        return render_template_string(FORM_HTML, charts=charts_html)

    app.register_blueprint(bp)
    print('ERDDAP blueprint registered at /erddap')


# -------------------------
# Small test runner (local quick test)
# -------------------------
if __name__ == '__main__':
    print('Discovering SST datasets (quick test)...')
    try:
        print(discover_erddap_datasets())
    except Exception as e:
        print('Smoke test failed:', e)
