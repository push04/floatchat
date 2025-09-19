# erddap_integration.py
# ERDDAP helper that first checks dataset time coverage and then fetches whatever data the dataset has (no user time input required).
from datetime import datetime, timedelta, timezone
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


# -------------------------
# Small helpers
# -------------------------
def _normalize_colname(s: str) -> str:
    return re.sub(r'\W+', '', str(s)).lower()


def _parse_iso_z(s):
    """Parse ISO timestamp possibly ending with Z into a timezone-aware datetime in UTC."""
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        except Exception:
            return None


# -------------------------
# Dataset variables + discovery
# -------------------------
def get_dataset_variables(server, dataset_id, timeout=8):
    """
    Read dataset index.csv to get variable names. Returns list or empty list.
    Silent on failures.
    """
    try:
        url = f"{server}/info/{dataset_id}/index.csv"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        # find a column that looks like 'Variable Name' or 'destinationName'
        cols = [c for c in df.columns if 'variable' in c.lower() and 'name' in c.lower()]
        if not cols:
            cols = [c for c in df.columns if 'destination' in c.lower() or 'destinationname' in c.lower()]
        if not cols:
            # fallback: sometimes index.csv lists a 'Row Type' column with variable rows
            if 'Row Type' in df.columns:
                vars_df = df[df['Row Type'].astype(str).str.lower() == 'variable']
                if 'Variable Name' in vars_df.columns:
                    return vars_df['Variable Name'].dropna().astype(str).unique().tolist()
            return []
        var_col = cols[0]
        vars_list = df[var_col].dropna().astype(str).unique().tolist()
        return [v.strip() for v in vars_list if v.strip()]
    except Exception:
        return []


def discover_erddap_datasets(server=ERDDAP_SERVER, keyword="sst", max_results=20):
    """
    Search ERDDAP server for datasets matching keyword.
    Returns DataFrame with dataset_id (and title if available) or empty DataFrame.
    """
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


# -------------------------
# Geocode helper
# -------------------------
def geocode_place(place):
    """Return (lat, lon) from place name. Returns None on failure."""
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
# Dataset time-range extraction
# -------------------------
def get_dataset_time_range(server, dataset_id, timeout=8):
    """
    Try to determine a dataset's time axis minimum and maximum.
    Returns (min_dt, max_dt) as timezone-aware datetimes in UTC, or (None, None) if unknown.
    Strategy:
      - parse /info/{dataset}/index.csv for time attributes (actual_range, time_coverage_start/end)
      - if not found, probe griddap with wide time window to elicit axis hints
    """
    # 1) parse index.csv
    try:
        url = f"{server}/info/{dataset_id}/index.csv"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        txt = r.text
        # Parse structured table if possible
        try:
            df = pd.read_csv(io.StringIO(txt), dtype=str)
            cols_norm = {c.lower(): c for c in df.columns}
            var_col = None
            attr_col = None
            val_col = None
            for k, v in cols_norm.items():
                if 'variable' in k and 'name' in k:
                    var_col = v
                if 'attribute' in k and 'name' in k:
                    attr_col = v
                if 'value' in k:
                    val_col = v
            start = None
            end = None
            if var_col and attr_col and val_col:
                rows = df[df[var_col].astype(str).str.lower().str.contains(r'\btime\b', na=False)]
                global_rows = df[df[var_col].isna() | (df[var_col].astype(str).str.strip() == '')]
                for _, row in rows.iterrows():
                    attr = str(row[attr_col]).strip().lower()
                    val = str(row[val_col]).strip()
                    if 'actual_range' in attr or 'valid_range' in attr or 'actualrange' in attr:
                        found = re.findall(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z", val)
                        if len(found) >= 1 and start is None:
                            start = _parse_iso_z(found[0])
                        if len(found) >= 2 and end is None:
                            end = _parse_iso_z(found[1])
                        found2 = re.findall(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]", val)
                        if found2:
                            if start is None:
                                start = _parse_iso_z(found2[0] + "T00:00:00Z")
                            if len(found2) >= 2 and end is None:
                                end = _parse_iso_z(found2[1] + "T00:00:00Z")
                    if 'time_coverage_start' in attr or 'timecoverage_start' in attr:
                        if start is None:
                            start = _parse_iso_z(val if 'T' in val else (val + "T00:00:00Z"))
                    if 'time_coverage_end' in attr or 'timecoverage_end' in attr:
                        if end is None:
                            end = _parse_iso_z(val if 'T' in val else (val + "T00:00:00Z"))
                # check global rows
                for _, row in global_rows.iterrows():
                    try:
                        attr = str(row[attr_col]).strip().lower()
                        val = str(row[val_col]).strip()
                    except Exception:
                        continue
                    if 'time_coverage_start' in attr or 'timecoverage_start' in attr:
                        if start is None:
                            start = _parse_iso_z(val if 'T' in val else (val + "T00:00:00Z"))
                    if 'time_coverage_end' in attr or 'timecoverage_end' in attr:
                        if end is None:
                            end = _parse_iso_z(val if 'T' in val else (val + "T00:00:00Z"))
                if start or end:
                    return start, end
            # fallback: search text for ISO datetimes
            found = re.findall(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z", txt)
            if found:
                dts = sorted({_parse_iso_z(x) for x in found if _parse_iso_z(x) is not None})
                if dts:
                    return dts[0], dts[-1]
        except Exception:
            pass
    except Exception:
        pass

    # 2) Probe griddap with a very wide time window to elicit axis hints (catch 404 text)
    try:
        probe_url = f"{server}/griddap/{dataset_id}.csv?time[(1900-01-01T00:00:00Z):1:({(datetime.utcnow()+timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')})][(0):1:(0)][(0):1:(0)]"
        r = requests.get(probe_url, timeout=20)
        text = r.text if isinstance(r.text, str) else ''
        if r.status_code == 200:
            # attempt to parse returned CSV for min/max times
            try:
                df = pd.read_csv(io.StringIO(r.text))
                if 'time' in [c.lower() for c in df.columns]:
                    df_cols = {c.lower(): c for c in df.columns}
                    df.rename(columns={df_cols.get('time'): 'time'}, inplace=True)
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    df = df.dropna(subset=['time'])
                    if not df.empty:
                        mn = df['time'].min().to_pydatetime().astimezone(timezone.utc)
                        mx = df['time'].max().to_pydatetime().astimezone(timezone.utc)
                        return mn, mx
            except Exception:
                pass
        # extract axis min/max from error message
        axis_min = None
        axis_max = None
        mmin = re.search(r"axis ?minimum=([0-9T:\-]+Z)", text)
        mmax = re.search(r"axis ?maximum=([0-9T:\-]+Z)", text)
        if mmin:
            axis_min = _parse_iso_z(mmin.group(1))
        if mmax:
            axis_max = _parse_iso_z(mmax.group(1))
        if not axis_min or not axis_max:
            found = re.findall(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z)", text)
            if found:
                dts = sorted({_parse_iso_z(x) for x in found if _parse_iso_z(x) is not None})
                if dts:
                    if not axis_min:
                        axis_min = dts[0]
                    if not axis_max and len(dts) > 1:
                        axis_max = dts[-1]
        if axis_min or axis_max:
            return axis_min, axis_max
    except Exception:
        pass

    return None, None


# -------------------------
# Core fetch using dataset time coverage
# -------------------------
def fetch_griddap_point_timeseries_using_dataset_time(server, dataset_id, variable, lat, lon, debug=False):
    """
    1) Determine dataset time range via get_dataset_time_range()
    2) Use that exact range to request a point timeseries (attempt lat-lon and lon-lat order)
    3) Return DataFrame (or (df, raw_texts) if debug=True)
    """
    raw_texts = []

    # 1) get dataset time range
    start_dt, end_dt = get_dataset_time_range(server, dataset_id)
    # If not found, fallback to a conservative short window (last 30 days)
    if start_dt is None and end_dt is None:
        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=30)
    elif start_dt is None and end_dt is not None:
        end_dt = end_dt.astimezone(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
    elif start_dt is not None and end_dt is None:
        start_dt = start_dt.astimezone(timezone.utc)
        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)

    def _to_iso(dt):
        if dt is None:
            return None
        if isinstance(dt, datetime):
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        return str(dt)

    start_iso = _to_iso(start_dt)
    end_iso = _to_iso(end_dt)

    orders = [("lat", "lon"), ("lon", "lat")]
    for order in orders:
        if order == ("lat", "lon"):
            point_idx = f"[({start_iso}):1:({end_iso})][({lat}):1:({lat})][({lon}):1:({lon})]"
            url = f"{server}/griddap/{dataset_id}.csv?{variable}{point_idx}"
        else:
            point_idx = f"[({start_iso}):1:({end_iso})][({lon}):1:({lon})][({lat}):1:({lat})]"
            url = f"{server}/griddap/{dataset_id}.csv?{variable}{point_idx}"

        try:
            r = requests.get(url, timeout=60)
            text = r.text if isinstance(r.text, str) else ''
            raw_texts.append((url, text[:60000] if text else f"HTTP {r.status_code}"))
            if r.status_code != 200:
                # if 404, attempt to extract axis hints and continue (defensive)
                continue
            df = pd.read_csv(io.StringIO(r.text))
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
                continue
            if debug:
                return df, raw_texts
            return df
        except Exception:
            continue

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


def get_preferred_for(friendly_variable):
    """Return preferred dataset list (dataset_id, varname) for a friendly variable."""
    return PREFERRED_DATASETS.get(friendly_variable, [])


def pick_dataset_and_var(server, friendly_variable):
    """
    Return (dataset_id, variable_name) or (None, None).
    Uses PREFERRED_DATASETS first then discovery on server.
    """
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
    # discover
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
# Plot helpers (unchanged)
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
# CLI helpers (friendly output)
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

    # Use dataset time coverage automatically
    df = fetch_griddap_point_timeseries_using_dataset_time(args.server, dataset_id, varname, lat, lon, debug=False)
    out = summarize_df(df, variable_col=varname, nrows=10)
    print(f"Using dataset: {dataset_id}  variable: {varname}")
    # show dataset time coverage (best-effort)
    start_dt, end_dt = get_dataset_time_range(args.server, dataset_id)
    if start_dt or end_dt:
        s_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if start_dt else "unknown"
        e_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if end_dt else "unknown"
        print(f"Dataset time coverage (best-effort): {s_str}  →  {e_str}")
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
    manual_latlon = st.sidebar.text_input('Manual lat,lon (e.g. "20.5,70.2")')
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
            result = fetch_griddap_point_timeseries_using_dataset_time(server_input, dataset_id, varname, lat, lon, debug=True)
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
            df = fetch_griddap_point_timeseries_using_dataset_time(server, dataset_id, varname, lat, lon, debug=False)
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
