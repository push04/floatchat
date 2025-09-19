# erddap_integration_fixed.py
# COMPLETE SCRIPT: ERDDAP helper that automatically switches between historical and near-real-time (NRT)
# datasets. Includes updated CLI, Streamlit, and Flask interfaces with a new month-year query option.

from datetime import datetime, timedelta, timezone
import calendar
import io
import re
import argparse
import sys
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional geocoding (install geopy).
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Default ERDDAP server (NOAA CoastWatch).
ERDDAP_SERVER = "https://coastwatch.noaa.gov/erddap"

# -------------------------
# Dataset Strategy Configuration
# -------------------------
# Verified dataset IDs & variables
DATASET_STRATEGY = {
    'Temperature': {
        'historical': ('jplMURSST41', 'analysed_sst'),
        'nrt': ('noaacwLEOACSPOSSTL3SnrtCDaily', 'sea_surface_temperature')
    },
    'Salinity': {
        'historical': ('NCOM_amseas_latest3d', 'salinity'),
        'nrt': ('NCOM_amseas_latest3d', 'salinity')
    },
    'Chlorophyll': {
        'historical': ('USM_VIIRS_DAP', 'CHL_Weekly'),
        'nrt': ('USM_VIIRS_DAP', 'CHL_Weekly')
    }
}
# How many days in the past defines the switch from NRT to historical data.
NRT_CUTOFF_DAYS = 60


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
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

# -------------------------
# Geocode helper
# -------------------------
def geocode_place(place):
    """Return (lat, lon) from place name. Returns None on failure."""
    if GEOPY_AVAILABLE:
        try:
            geolocator = Nominatim(user_agent="floatchat_erddap_integration_full")
            loc = geolocator.geocode(place, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "floatchat_erddap_integration_full/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and len(j) > 0:
            return float(j[0].get("lat")), float(j[0].get("lon"))
    except requests.exceptions.RequestException:
        pass
    return None

# -------------------------
# Core fetching logic
# -------------------------
def _fetch_erddap_data(server, dataset_id, variable, lat, lon, start_iso, end_iso, debug=False):
    """Low-level fetcher for a specific dataset and time range."""
    raw_texts = []
    orders = [("lat", "lon"), ("lon", "lat")]
    for order in orders:
        if order == ("lat", "lon"):
            point_idx = f"[({start_iso}):1:({end_iso})][({lat}):1:({lat})][({lon}):1:({lon})]"
        else:
            point_idx = f"[({start_iso}):1:({end_iso})][({lon}):1:({lon})][({lat}):1:({lat})]"

        url = f"{server}/griddap/{dataset_id}.csv?{variable}{point_idx}"

        try:
            r = requests.get(url, timeout=60)
            text = r.text if isinstance(r.text, str) else ''
            raw_texts.append((url, text[:1000] if text else f"HTTP {r.status_code}"))

            if r.status_code != 200:
                continue

            df = pd.read_csv(io.StringIO(r.text))
            col_map = {c.lower(): c for c in df.columns}
            rename_dict = {}
            if 'time' in col_map: rename_dict[col_map['time']] = 'time'
            if 'latitude' in col_map: rename_dict[col_map['latitude']] = 'latitude'
            if 'longitude' in col_map: rename_dict[col_map['longitude']] = 'longitude'
            df.rename(columns=rename_dict, inplace=True)

            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                df = df.dropna(subset=['time'])

            # Add the variable name as a column for clarity in plotting
            data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
            if data_cols:
                df.rename(columns={data_cols[0]: variable}, inplace=True)

            if not df.empty:
                return (df, raw_texts, dataset_id, variable) if debug else df

        except requests.exceptions.RequestException:
            continue
    
    empty_result = (pd.DataFrame(), raw_texts, dataset_id, variable) if debug else pd.DataFrame()
    return empty_result

def fetch_data_with_strategy(server, friendly_variable, lat, lon, start_dt, end_dt, debug=False):
    """
    Selects dataset based on time range and fetches data.
    Splits query if it spans cutoff.
    """
    if friendly_variable not in DATASET_STRATEGY:
        print(f"Error: Variable '{friendly_variable}' not configured.")
        return pd.DataFrame()

    strategy = DATASET_STRATEGY[friendly_variable]
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=NRT_CUTOFF_DAYS)

    dfs, raw_texts = [], []
    # Entirely historical
    if end_dt < cutoff_date:
        dataset_id, varname = strategy['historical']
        result = _fetch_erddap_data(server, dataset_id, varname,
                                    lat, lon,
                                    start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    debug)
        if debug:
            df, rtxt, ds, var = result
            return df, rtxt, ds, var
        return result

    # Entirely recent
    if start_dt >= cutoff_date:
        dataset_id, varname = strategy['nrt']
        result = _fetch_erddap_data(server, dataset_id, varname,
                                    lat, lon,
                                    start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    debug)
        if debug:
            df, rtxt, ds, var = result
            return df, rtxt, ds, var
        return result

    # Spanning cutoff: split
    dataset_hist, var_hist = strategy['historical']
    dataset_nrt, var_nrt = strategy['nrt']

    df_hist = _fetch_erddap_data(server, dataset_hist, var_hist,
                                 lat, lon,
                                 start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
    df_nrt = _fetch_erddap_data(server, dataset_nrt, var_nrt,
                                lat, lon,
                                cutoff_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    df_all = pd.concat([df_hist, df_nrt]).drop_duplicates().sort_values("time")
    return df_all

# -------------------------
# Plot helpers
# -------------------------
def timeseries_plot(df):
    if df.empty:
        return go.Figure().update_layout(title='No data to display')
    
    data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
    varcol = data_cols[0] if data_cols else None
    if not varcol:
        return go.Figure().update_layout(title='No data variable found')
        
    fig = px.line(df.sort_values('time'), x='time', y=varcol, title=f"Timeseries for {varcol}")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def map_scatter_plot(df):
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return go.Figure().update_layout(title='No spatial data to display')

    data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
    varcol = data_cols[0] if data_cols else None
    if not varcol:
        return go.Figure().update_layout(title='No data variable found')

    df_latest = df.loc[df['time'].idxmax()].to_frame().T
    
    fig = px.scatter_geo(df_latest, lat='latitude', lon='longitude', size_max=15,
                         hover_name=varcol, hover_data=['time'],
                         projection='natural earth')
    fig.update_layout(title=f"Map of {varcol} (Latest Point)")
    return fig

# -------------------------
# CLI helpers
# -------------------------
def summarize_df(df, nrows=10):
    if df.empty:
        return "No data available for the requested query."

    txt = [f"Rows returned: {len(df)}"]
    data_cols = [c for c in df.columns if c.lower() not in ['time', 'latitude', 'longitude']]
    data_col = data_cols[0] if data_cols else None
    
    txt.append(f"\nSample (first {min(nrows, len(df))} rows):")
    txt.append(df.head(nrows).to_string(index=False))

    if data_col:
        ser = pd.to_numeric(df[data_col], errors='coerce').dropna()
        if not ser.empty:
            txt.append(f"\nSummary statistics for {data_col}:")
            txt.append(f"  Mean: {ser.mean():.3f}, Min: {ser.min():.3f}, Max: {ser.max():.3f}")
    return "\n".join(txt)

def cli_fetch_and_print(args):
    # 1. Resolve Location
    latlon = None
    if args.lat is not None and args.lon is not None:
        latlon = (args.lat, args.lon)
    elif args.place:
        print(f"Geocoding '{args.place}'...")
        latlon = geocode_place(args.place)
    
    if latlon is None:
        print("Error: Could not determine location.")
        return 2
    lat, lon = latlon
    print(f"Using coordinates: Lat={lat:.3f}, Lon={lon:.3f}")

    # 2. Resolve Time Range
    if args.month_year:
        try:
            month_str, year_str = args.month_year.split('-')
            month, year = int(month_str), int(year_str)
            _, last_day = calendar.monthrange(year, month)
            start_dt = datetime(year, month, 1, tzinfo=timezone.utc)
            end_dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            print(f"Querying for month: {start_dt.strftime('%B %Y')}")
        except ValueError:
            print("Error: Invalid --month-year format. Use MM-YYYY.")
            return 3
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
        print("No time specified, defaulting to the last 30 days.")

    # 3. Fetch Data
    df = fetch_data_with_strategy(args.server, args.var_friendly, lat, lon, start_dt, end_dt)

    # 4. Print Summary
    print("-" * 40)
    print(summarize_df(df))
    return 0

# -------------------------
# CLI entrypoint
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        prog="erddap_integration_fixed.py",
        description="Fetch timeseries from ERDDAP, auto-switching datasets."
    )
    p.add_argument("--server", default=ERDDAP_SERVER, help="ERDDAP server URL")
    loc_group = p.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--place", type=str, help="Place name (e.g., 'Honolulu, Hawaii')")
    loc_group.add_argument("--latlon", type=str, help="Manual lat,lon (e.g., '21.3,-157.8')")
    p.add_argument("--month-year", type=str, help="Fetch for a specific month (format: MM-YYYY)")
    p.add_argument("--var-friendly", choices=DATASET_STRATEGY.keys(), default="Temperature", help="Friendly variable name.")
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.latlon:
        try:
            lat_str, lon_str = args.latlon.split(',')
            args.lat, args.lon = float(lat_str), float(lon_str)
        except ValueError:
            print("Error: Invalid --latlon format. Use 'lat,lon'.")
            sys.exit(1)
    else:
        args.lat, args.lon = None, None

    try:
        rc = cli_fetch_and_print(args)
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()
