# erddap_integration.py
# ERDDAP helper that fetches whatever time-range the dataset actually has (no user time input required).
from datetime import datetime, timedelta
import io
import re
import argparse
import sys
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

# Default ERDDAP server (NOAA CoastWatch). Change if you want a different host.
ERDDAP_SERVER = "https://coastwatch.noaa.gov/erddap"
BBOX_HALF_DEG = 0.125


def get_dataset_variables(server, dataset_id, timeout=8):
    try:
        url = f"{server}/info/{dataset_id}/index.csv"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        cols = [c for c in df.columns if 'variable' in c.lower() and 'name' in c.lower()]
        if not cols:
            cols = [c for c in df.columns if 'destination' in c.lower() or 'destinationname' in c.lower()]
        if not cols:
            if 'Row Type' in df.columns:
                vars_df = df[df['Row Type'].astype(str).str.lower() == 'variable']
                if 'Variable Name' in vars_df.columns:
                    return vars_df['Variable Name'].dropna().astype(str).unique().tolist()
            return []
        var_col = cols[0]
        vars_list = df[var_col].dropna().astype(str).unique().tolist()
        return [v.strip() for v in vars_list if v.strip() and len(v.strip()) < 200]
    except Exception:
        return []


def get_preferred_for(friendly_variable):
    return PREFERRED_DATASETS.get(friendly_variable, [])


def _normalize_colname(s: str) -> str:
    return re.sub(r'\W+', '', str(s)).lower()


def discover_erddap_datasets(server=ERDDAP_SERVER, keyword="sst", max_results=20):
    try:
        q = requests.utils.requote_uri(f"{server}/search/index.csv?searchFor={keyword}&itemsPerPage={max_results}")
        r = requests.get(q, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
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
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def geocode_place(place):
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent="floatchat_erddap_integration")
            loc = geolocator.geocode(place, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1, "addressdetails": 0}
        headers = {"User-Agent": "floatchat_erddap_integration/1.0 (contact@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and len(j) > 0:
            return float(j[0].get("lat")), float(j[0].get("lon"))
    except Exception:
        pass
    return None


# -------------------------
# Fetch that uses dataset's time coverage automatically
# -------------------------
def fetch_griddap_point_timeseries(server, dataset_id, variable, lat, lon,
                                   debug=False):
    """
    Fetches a point timeseries for (lat,lon) using the dataset's available time axis only.
    Works by:
      - first trying a very wide time range (1900 -> now) to elicit axis-min/max hints
      - if ERDDAP responds with a 'axis minimum' or 'axis maximum' message, clamp to those bounds
      - retry once with clamped bounds and return the data available
    Returns DataFrame or (DataFrame, raw_texts) if debug=True.
    """
    raw_texts = []

    def _attempt_request(url):
        try:
            r = requests.get(url, timeout=90)
            text = r.text if isinstance(r.text, str) else ''
            raw_texts.append((url, text[:60000] if text else f"HTTP {r.status_code}"))
            if r.status_code != 200:
                return None, r
            df = pd.read_csv(io.StringIO(r.text))
            # Normalize and coerce
            col_lower = {c.lower(): c for c in df.columns}
            if 'time' in col_lower:
                df.rename(columns={col_lower['time']: 'time'}, inplace=True)
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if 'latitude' in col_lower:
                df.rename(columns={col_lower['latitude']: 'latitude'}, inplace=True)
            elif 'lat' in col_lower:
                df.rename(columns={col_lower['lat']: 'latitude'}, inplace=True)
            if 'longitude' in col_lower:
                df.rename(columns={col_lower['longitude']: 'longitude'}, inplace=True)
            elif 'lon' in col_lower:
                df.rename(columns={col_lower['lon']: 'longitude'}, inplace=True)
            if 'latitude' in df.columns:
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            if 'longitude' in df.columns:
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            value_cols = [c for c in df.columns if c not in ('time', 'latitude', 'longitude')]
            for c in value_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            if 'time' in df.columns:
                df = df.dropna(subset=['time'])
            if df.empty:
                return None, r
            return df, r
        except Exception as e:
            raw_texts.append((url, f"EXCEPTION: {e}"))
            return None, None

    # regex helpers to find axis min/max in ERDDAP error messages
    def _extract_axis_minimum(text):
        m = re.search(r"axis ?minimum=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)", text)
        if m:
            try:
                return datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            except Exception:
                pass
        m2 = re.search(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z)", text)
        if m2:
            try:
                return datetime.fromisoformat(m2.group(1).replace("Z", "+00:00"))
            except Exception:
                pass
        return None

    def _extract_axis_maximum(text):
        m = re.search(r"axis ?maximum=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)", text)
        if m:
            try:
                return datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            except Exception:
                pass
        m2 = re.search(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z)", text)
        if m2:
            try:
                return datetime.fromisoformat(m2.group(1).replace("Z", "+00:00"))
            except Exception:
                pass
        return None

    # Start with a very wide range to prompt ERDDAP to tell us the dataset axis if it differs
    start_iso = "1900-01-01T00:00:00Z"
    end_iso = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    attempt_bboxes = [BBOX_HALF_DEG, 0.25, 0.5, 1.0, 2.0]
    orders = [("lat", "lon"), ("lon", "lat")]
    clamped = False

    # Two passes: 1) initial (wide dates), 2) optional clamped (if we discover axis min/max)
    for pass_num in (1, 2):
        for hb in attempt_bboxes:
            lat_min = lat - hb
            lat_max = lat + hb
            lon_min = lon - hb
            lon_max = lon + hb
            for order in orders:
                if order == ("lat", "lon"):
                    bounds = f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
                else:
                    bounds = f"[({lon_min}):1:({lon_max})][({lat_min}):1:({lat_max})]"
                url = f"{server}/griddap/{dataset_id}.csv?{variable}[({start_iso}):1:({end_iso})]{bounds}"
                df, resp = _attempt_request(url)
                if df is not None:
                    if debug:
                        return df, raw_texts
                    return df
                # If ERDDAP returned a 404 with axis hints, extract and clamp
                if resp is not None and resp.status_code == 404 and not clamped:
                    text = resp.text if isinstance(resp.text, str) else ''
                    axis_min_dt = _extract_axis_minimum(text)
                    axis_max_dt = _extract_axis_maximum(text)
                    if axis_min_dt or axis_max_dt:
                        # clamp start/end if available
                        if axis_min_dt:
                            start_iso = axis_min_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        if axis_max_dt:
                            end_iso = axis_max_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        clamped = True
                        # break to restart attempts with clamped times immediately
                        break
            else:
                continue
            break
        # continue to second pass if we clamped; otherwise pass_num loop will end naturally

    # Final exact-point attempt (no bbox)
    for pass_num in (1, 2):
        for order in orders:
            if order == ("lat", "lon"):
                point_idx = f"[({lat}):1:({lat})][({lon}):1:({lon})]"
            else:
                point_idx = f"[({lon}):1:({lon})][({lat}):1:({lat})]"
            url = f"{server}/griddap/{dataset_id}.csv?{variable}[({start_iso}):1:({end_iso})]{point_idx}"
            df, resp = _attempt_request(url)
            if df is not None:
                if debug:
                    return df, raw_texts
                return df
            if resp is not None and resp.status_code == 404 and not clamped:
                text = resp.text if isinstance(resp.text, str) else ''
                axis_min_dt = _extract_axis_minimum(text)
                axis_max_dt = _extract_axis_maximum(text)
                if axis_min_dt or axis_max_dt:
                    if axis_min_dt:
                        start_iso = axis_min_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if axis_max_dt:
                        end_iso = axis_max_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    clamped = True
                    break
        if clamped:
            continue
        else:
            break

    # Nothing found
    if debug:
        return pd.DataFrame(), raw_texts
    return pd.DataFrame()


# -------------------------
# Variable mappings & preferences
# -------------------------
DEFAULT_VARIABLES = {
    'Temperature': {'search_kw': 'sst OR sea surface temperature OR sea_surface_temperature', 'var_names': ['sea_surface_temperature', 'sst', 'analysed_sst', 'temperature']},
    'Salinity': {'search_kw': 'salinity OR sss OR sea surface salinity', 'var_names': ['sss', 'salinity', 'sea_surface_salinity']},
    'Chlorophyll': {'search_kw': 'chlorophyll OR chlor_a', 'var_names': ['chlor_a', 'chlorophyll', 'chl']},
}

PREFERRED_DATASETS = {
    'Temperature': [
        ('noaacwLEOACSPOSSTL3SnrtCDaily', 'sea_surface_temperature'),
        ('jplMURSST41', 'analysed_sst'),
    ],
    'Salinity': [
        ('NCOM_amseas_latest3d', 'salinity'),
    ],
    'Chlorophyll': [
        ('USM_VIIRS_DAP', 'chlor_a'),
    ],
}


def pick_dataset_and_var(server, friendly_variable):
    if friendly_variable not in DEFAULT_VARIABLES:
        return None, None
    info = DEFAULT_VARIABLES[friendly_variable]
    prefs = PREFERRED_DATASETS.get(friendly_variable, [])
    for ds_id, varname in prefs:
        vars_list = get_dataset_variables(server, ds_id)
        if not vars_list:
            continue
        vs = [v.lower() for v in vars_list]
        if varname.lower() in vs:
            matched = [v for v in vars_list if v.lower() == varname.lower()][0]
            return ds_id, matched
        for candidate in info['var_names']:
            for v in vars_list:
                if v.lower() == candidate.lower():
                    return ds_id, v
    # discover on server
    df = discover_erddap_datasets(server, info['search_kw'])
    if df.empty:
        df = discover_erddap_datasets(server, friendly_variable.lower())
    if df.empty:
        return None, None
    probe_n = min(6, len(df))
    for i in range(probe_n):
        dataset_id = df.iloc[i]['dataset_id']
        vars_list = get_dataset_variables(server, dataset_id)
        if not vars_list:
            continue
        vs = [v.lower() for v in vars_list]
        for candidate in info['var_names']:
            if candidate.lower() in vs:
                matched = [v for v in vars_list if v.lower() == candidate.lower()][0]
                return dataset_id, matched
        non_geo = [v for v in vars_list if v.lower() not in ('latitude','longitude','time','depth')]
        if non_geo:
            return dataset_id, non_geo[0]
    return None, None


# -------------------------
# Plot helpers
# -------------------------
def timeseries_plot(df, variable):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='No data')
        return fig
    candidates = [c for c in df.columns if c not in ['time', 'latitude', 'longitude']]
    varcol = variable if variable in df.columns else (candidates[0] if candidates else None)
    if varcol is None:
        fig = go.Figure()
        fig.update_layout(title='Variable not found in fetched data')
        return fig
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    median_lat = None
    median_lon = None
    if 'latitude' in df.columns:
        median_lat = pd.to_numeric(df['latitude'], errors='coerce').median()
    if 'longitude' in df.columns:
        median_lon = pd.to_numeric(df['longitude'], errors='coerce').median()
    coord_str = ""
    if pd.notna(median_lat) and pd.notna(median_lon):
        coord_str = f" @ {median_lat:.3f},{median_lon:.3f}"
    else:
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coord_str = f" @ {str(df['latitude'].iloc[0])},{str(df['longitude'].iloc[0])}"
    fig = px.line(df.sort_values('time'), x='time', y=varcol, title=f"{varcol}{coord_str}")
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def map_scatter_plot(df, variable):
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
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        latest_time = df['time'].max()
        df_latest = df[df['time'] == latest_time]
    else:
        df_latest = df
        latest_time = None
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
# CLI helpers
# -------------------------
def summarize_df(df, variable_col=None, nrows=10):
    if df.empty:
        return "No data available for the requested query."
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values('time')
    txt = []
    txt.append(f"Rows returned: {len(df)}")
    display_cols = ['time']
    if 'latitude' in df.columns and 'longitude' in df.columns:
        display_cols += ['latitude', 'longitude']
    data_cols = [c for c in df.columns if c not in display_cols]
    if data_cols:
        data_col = variable_col if variable_col in df.columns else data_cols[0]
        display_cols += [data_col]
    else:
        data_col = None
    txt.append("\nSample (first %d rows):" % min(nrows, len(df)))
    sample = df[display_cols].head(nrows).copy()
    txt += sample.to_string(index=False).splitlines()
    if data_col:
        ser = pd.to_numeric(df[data_col], errors='coerce')
        valid = ser.dropna()
        if not valid.empty:
            txt.append("\nSummary statistics for %s:" % data_col)
            txt.append(f"  Count: {int(valid.count())}")
            txt.append(f"  Mean: {valid.mean():.3f}")
            txt.append(f"  Std: {valid.std():.3f}")
            txt.append(f"  Min: {valid.min():.3f}")
            txt.append(f"  Max: {valid.max():.3f}")
        else:
            txt.append(f"\nNo numeric values found in column '{data_col}'.")
    return "\n".join(txt)


def cli_fetch_and_print(args):
    # Resolve lat/lon
    latlon = None
    if args.lat is not None and args.lon is not None:
        latlon = (args.lat, args.lon)
    elif args.place:
        latlon = geocode_place(args.place)
    if latlon is None:
        print("Please provide --lat and --lon or --place (place name).")
        return 2
    lat, lon = latlon

    # Decide dataset/variable
    if args.dataset and args.var:
        dataset_id = args.dataset
        varname = args.var
    else:
        dataset_id, varname = pick_dataset_and_var(args.server, args.var_friendly or 'Temperature')
        if dataset_id is None:
            print("Could not auto-detect dataset/variable. Try using --dataset and --var to override.")
            return 3

    # IMPORTANT: ignore any user-supplied time; fetch whatever the dataset has
    df = fetch_griddap_point_timeseries(args.server, dataset_id, varname, lat, lon, debug=False)
    out = summarize_df(df, variable_col=varname, nrows=10)
    print(f"Using dataset: {dataset_id}  variable: {varname}")
    print(out)
    return 0


# -------------------------
# Streamlit and Flask helpers (unchanged API)
# -------------------------
def erddap_streamlit_widget(server=ERDDAP_SERVER):
    try:
        import streamlit as st
    except Exception:
        return
    st.sidebar.header('Ocean data (ERDDAP)')
    var_choice = st.sidebar.selectbox('Variable', list(DEFAULT_VARIABLES.keys()))
    prefs = get_preferred_for(var_choice)
    if prefs:
        st.sidebar.markdown("**Recommended datasets:**")
        for ds, vn in prefs:
            st.sidebar.caption(f"{ds}  —  try var `{vn}`")
    place = st.sidebar.text_input('Place name (or leave blank to enter lat,lon manually)')
    manual_latlon = st.sidebar.text_input('Manual lat,lon (e.g. \"20.5,70.2\")')
    current_year = datetime.utcnow().year
    year_range = st.sidebar.slider('Year range', min_value=1980, max_value=current_year,
                                   value=(current_year-1, current_year), step=1)
    start_year, end_year = year_range
    server_input = st.sidebar.text_input('ERDDAP server', value=server)
    if st.sidebar.button('Fetch'):
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
                    st.warning('Geocoding failed. Please enter manual lat,lon.')
                else:
                    latlon = g
        else:
            st.info('Enter a place name OR manual lat,lon to fetch data.')
            return
        if latlon is None:
            st.stop()
        lat, lon = latlon
        st.markdown("**Optional overrides (use if automatic discovery returns no data):**")
        manual_dataset = st.text_input('Override dataset_id (leave blank to use auto-discovery)')
        manual_var = st.text_input('Override variable name (leave blank to use auto-detected var)')
        with st.spinner('Discovering dataset...'):
            if manual_dataset and manual_dataset.strip():
                dataset_id = manual_dataset.strip()
                if manual_var and manual_var.strip():
                    varname = manual_var.strip()
                else:
                    vars_list = get_dataset_variables(server_input, dataset_id)
                    if vars_list:
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
        with st.spinner('Fetching timeseries (using dataset time coverage)...'):
            result = fetch_griddap_point_timeseries(server_input, dataset_id, varname, lat, lon, debug=True)
            if isinstance(result, tuple):
                df, raw_texts = result
            else:
                df = result
                raw_texts = []
        if df.empty:
            st.error('No data returned for that dataset/variable/point/time.')
            for i, (u, t) in enumerate(raw_texts[:6]):
                st.markdown(f"**Attempt {i+1}**: `{u}`")
                st.code(t[:2000])
            st.info('Try entering a different dataset_id/variable or a nearby lat/lon.')
            return
        st.subheader('Timeseries')
        fig_ts = timeseries_plot(df, varname)
        st.plotly_chart(fig_ts, use_container_width=True)
        st.subheader('Map (latest slice)')
        fig_map = map_scatter_plot(df, varname)
        st.plotly_chart(fig_map, use_container_width=True)
        try:
            latest = df.loc[df['time'].idxmax()]
            val = None
            for c in df.columns:
                if c not in ['time', 'latitude', 'longitude']:
                    val = latest[c]
                    break
            if val is not None:
                st.metric(label=f"Latest {varname}", value=f"{float(val):.3f}")
        except Exception:
            pass


def register_erddap_blueprint(app, server=ERDDAP_SERVER):
    try:
        from flask import Blueprint, request, render_template_string
    except Exception:
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
            df = fetch_griddap_point_timeseries(server, dataset_id, varname, lat, lon, debug=False)
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


# -------------------------
# CLI entrypoint
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(prog="erddap_integration.py", description="Fetch point timeseries from ERDDAP and print friendly output (uses dataset's available time automatically)")
    p.add_argument("--server", default=ERDDAP_SERVER, help="ERDDAP server base URL")
    p.add_argument("--lat", type=float, help="Latitude of point (decimal degrees)")
    p.add_argument("--lon", type=float, help="Longitude of point (decimal degrees)")
    p.add_argument("--place", type=str, help="Place name (geocoded via Nominatim if geopy not available)")
    p.add_argument("--var", help="Explicit variable name on dataset (use with --dataset)")
    p.add_argument("--dataset", help="Explicit dataset id (if you know it)")
    p.add_argument("--var-friendly", help="Friendly variable name (Temperature/Salinity/Chlorophyll)", default="Temperature")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    try:
        rc = cli_fetch_and_print(args)
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
