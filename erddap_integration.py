"""
ERDDAP integration helper for floatchat/app.py

INSTRUCTIONS (paste these lines into your repo):
1) Create a new file at the root of the project: `erddap_integration.py` and paste this whole file there.
2) Install required packages if not present:
   pip install pandas requests plotly geopy

3a) If your app is a Streamlit app: open app.py and *only* add the following two lines near the top (after other imports):
   from erddap_integration import erddap_streamlit_widget

   And then inside your main UI rendering function (where the Streamlit widgets live), add a single call:
   erddap_streamlit_widget()

   That is the only modification required in app.py for Streamlit integration.

3b) If your app is a Flask (or other WSGI) app: add these two lines near app initialization:
   from erddap_integration import register_erddap_blueprint
   register_erddap_blueprint(app)

   This will mount a small blueprint at /erddap which provides a form, results and Plotly charts.

IMPORTANT: This module only *adds* new functions/routes/UI. It does not modify any of your existing functions.

USAGE:
- Streamlit: the widget will show an options select for variables (temperature, salinity, chlorophyll), a place text box (or manual lat/lon), timeframe, and render interactive Plotly charts.
- Flask: visit /erddap, submit the form and see charts.

Notes on behavior:
- The code auto-discovers ERDDAP datasets by keyword using the ERDDAP search endpoint. It will try to choose a dataset containing the requested variable. If discovery fails, you'll be prompted to specify a dataset_id.
- For geocoding a place name we use Nominatim (geopy). If not available or rate-limited, use the "Manual lat,lon" text input.

"""

from datetime import datetime, timedelta
import math
import io
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional geocoding (install geopy). If not installed, the code will prompt manual lat/lon input.
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

ERDDAP_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
# small bbox half-width (deg) to capture nearest grid cell
BBOX_HALF_DEG = 0.125


# -------------------------
# Helper: discover datasets
# -------------------------

def discover_erddap_datasets(server=ERDDAP_SERVER, keyword="sst", max_results=8):
    """Search ERDDAP server for datasets matching keyword.

    Returns a pandas.DataFrame with columns: datasetID, title, variables (if available)
    """
    try:
        q = requests.utils.requote_uri(f"{server}/search/index.csv?searchFor={keyword}&itemsPerPage={max_results}")
        r = requests.get(q, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # The ERDDAP search returns columns: datasetID, title, id, ... keep datasetID + title
        if 'datasetID' in df.columns:
            return df[['datasetID', 'title']].drop_duplicates().reset_index(drop=True)
        # fallback
        return df
    except Exception as e:
        print("ERDDAP discover failed:", e)
        return pd.DataFrame()


# -------------------------
# Helper: basic geocode
# -------------------------

def geocode_place(place):
    """Return (lat, lon) from place name using Nominatim (geopy). Returns None on failure."""
    if not GEOPY_AVAILABLE:
        return None
    try:
        geolocator = Nominatim(user_agent="floatchat_erddap_integration")
        loc = geolocator.geocode(place, timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception as e:
        print("Geocode failed:", e)
    return None


# -------------------------
# Helper: fetch gridded point timeseries
# -------------------------

def fetch_griddap_point_timeseries(server, dataset_id, variable, lat, lon, end_date=None, days=7):
    """Fetch small time-series for the nearest grid cell around (lat,lon) for variable.

    Returns pandas.DataFrame with columns time, latitude, longitude, <variable>
    """
    if end_date is None:
        end_date = datetime.utcnow().date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    start_date = end_date - timedelta(days=days-1)

    lat_min = lat - BBOX_HALF_DEG
    lat_max = lat + BBOX_HALF_DEG
    lon_min = lon - BBOX_HALF_DEG
    lon_max = lon + BBOX_HALF_DEG

    # ERDDAP griddap CSV syntax
    # e.g. /griddap/<datasetID>.csv?<var>[(start):1:(end)][(latmin):1:(latmax)][(lonmin):1:(lonmax)]
    start = start_date.isoformat()
    end = end_date.isoformat()
    var_part = variable
    url = (
        f"{server}/griddap/{dataset_id}.csv?{var_part}"
        f"[({start}):1:({end})]"
        f"[({lat_min}):1:({lat_max})]"
        f"[({lon_min}):1:({lon_max})]"
    )
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # ERDDAP CSV usually repeats header rows; keep first header row parse
        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print('fetch_griddap_point_timeseries error', e)
        return pd.DataFrame()


# -------------------------
# Helper: choose dataset & variable mapping
# -------------------------

# Friendly variable -> search keywords and common variable names to try
DEFAULT_VARIABLES = {
    'Temperature': {'search_kw': 'sst OR sea surface temperature OR sea_surface_temperature', 'var_names': ['sst', 'analysed_sst', 'sea_surface_temperature', 'temperature']},
    'Salinity': {'search_kw': 'salinity OR sss OR sea surface salinity', 'var_names': ['sss', 'salinity', 'sea_surface_salinity']},
    'Chlorophyll': {'search_kw': 'chlorophyll OR chlor_a', 'var_names': ['chlorophyll', 'chlor_a', 'chl']},
}


def pick_dataset_and_var(server, friendly_variable):
    """Try to discover a dataset and a plausible variable name for the friendly_variable.

    Returns (dataset_id, variable_name) or (None, None).
    """
    if friendly_variable not in DEFAULT_VARIABLES:
        return None, None
    info = DEFAULT_VARIABLES[friendly_variable]
    # search by keywords; if search fails try shorter keyword
    df = discover_erddap_datasets(server, info['search_kw'])
    if df.empty:
        # try single word
        df = discover_erddap_datasets(server, friendly_variable.lower())
    if df.empty:
        return None, None
    # pick first dataset
    dataset_id = df.iloc[0]['datasetID']
    # choose a variable name from candidates
    # Note: we can't inspect dataset variables easily without opening its dataset page. We'll try candidates.
    for v in info['var_names']:
        # Quick test: try fetching just 1 day at lat=0 lon=0 with a tiny bbox - if no error, likely works.
        test_url = f"{server}/griddap/{dataset_id}.csv?{v}[(1970-01-01T00:00:00Z):1:(1970-01-01T00:00:00Z)][(0):1:(0)][(0):1:(0)]"
        try:
            r = requests.get(test_url, timeout=10)
            if r.status_code == 200 and len(r.text) > 0:
                return dataset_id, v
        except Exception:
            continue
    # fallback: return dataset and first candidate (user may need to edit)
    return dataset_id, info['var_names'][0]


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
    fig = px.line(df, x='time', y=varcol, title=f"{varcol} @ {df['latitude'].median():.3f},{df['longitude'].median():.3f}")
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
    latest_time = pd.to_datetime(df['time']).max()
    df_latest = df[df['time'] == latest_time]
    fig = px.scatter_geo(df_latest, lat='latitude', lon='longitude', size_max=8,
                         hover_name=varcol, hover_data=['time', 'latitude', 'longitude'],
                         projection='natural earth')
    fig.update_layout(title=f"Map of {varcol} at {latest_time}")
    return fig


# -------------------------
# Streamlit widget (drop-in)
# -------------------------

def erddap_streamlit_widget(server=ERDDAP_SERVER):
    """If your app is Streamlit, call this function in your main UI to mount the ERDDAP widget.

    It creates an options select (Temperature/Salinity/Chlorophyll), a place text input (or manual lat,lon),
    timeframe selector (days), and renders Plotly charts inline.
    """
    try:
        import streamlit as st
    except Exception:
        print('Streamlit not available. This widget is for Streamlit apps.')
        return

    st.sidebar.header('Ocean data (ERDDAP)')
    var_choice = st.sidebar.selectbox('Variable', list(DEFAULT_VARIABLES.keys()))
    place = st.sidebar.text_input('Place name (or leave blank to enter lat,lon manually)')
    manual_latlon = st.sidebar.text_input('Manual lat,lon (e.g. "20.5,70.2")')
    days = st.sidebar.slider('Days for timeseries', min_value=1, max_value=30, value=7)
    server_input = st.sidebar.text_input('ERDDAP server', value=server)

    if st.sidebar.button('Fetch'):
        # resolve lat lon
        latlon = None
        if manual_latlon.strip():
            try:
                parts = [p.strip() for p in manual_latlon.split(',')]
                latlon = (float(parts[0]), float(parts[1]))
            except Exception:
                st.error('Manual lat,lon parse failed. Use format: lat,lon')
                return
        elif place.strip():
            with st.spinner('Geocoding place...'):
                g = geocode_place(place)
                if g is None:
                    st.warning('Geocoding failed or geopy not installed. Please enter manual lat,lon.')
                else:
                    latlon = g
        else:
            st.info('Enter a place name OR manual lat,lon to fetch data.')
            return

        if latlon is None:
            st.stop()

        lat, lon = latlon
        with st.spinner('Discovering dataset...'):
            dataset_id, varname = pick_dataset_and_var(server_input, var_choice)
        if dataset_id is None:
            st.error('Could not find a suitable dataset on ERDDAP. Try another variable or specify dataset manually.')
            dataset_id = st.text_input('dataset_id (enter a griddap dataset ID if you know one)')
            varname = st.text_input('variable name in dataset (e.g. sst, sss)')
            if not dataset_id or not varname:
                return

        st.write(f"Using dataset: **{dataset_id}** and variable **{varname}**")

        with st.spinner('Fetching timeseries...'):
            df = fetch_griddap_point_timeseries(server_input, dataset_id, varname, lat, lon, days=days)
        if df.empty:
            st.error('No data returned for that dataset/variable/point/time. Try a different dataset or date range.')
            return

        st.subheader('Timeseries')
        fig_ts = timeseries_plot(df, varname)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader('Map (latest slice)')
        fig_map = map_scatter_plot(df, varname)
        st.plotly_chart(fig_map, use_container_width=True)

        # Quick stats
        latest = df.loc[df['time'].idxmax()]
        val = None
        for c in df.columns:
            if c not in ['time', 'latitude', 'longitude']:
                val = latest[c]
                break
        if val is not None:
            st.metric(label=f"Latest {varname}", value=f"{val:.3f}")


# -------------------------
# Flask blueprint (optional)
# -------------------------

def register_erddap_blueprint(app, server=ERDDAP_SERVER):
    """Register a small Flask blueprint at /erddap for interactive queries.

    Usage in your app.py:
        from erddap_integration import register_erddap_blueprint
        register_erddap_blueprint(app)

    This function will add routes:
      GET /erddap -> simple HTML form
      POST /erddap -> returns HTML with Plotly charts
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
    # quick smoke test: discover SST datasets
    print('Discovering SST datasets...')
    print(discover_erddap_datasets())

