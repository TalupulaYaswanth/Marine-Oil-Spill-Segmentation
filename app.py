"""
app.py
Streamlit Interactive Dashboard for Marine Oil Spill Segmentation & Vessel Attribution.

Real-World Remote Sensing & Maritime Forensics Edition:
  - Benchmark Real-World Incidents:
      1. MV Wakashio Disaster (Mauritius, Aug 2020) — Copernicus EMS EMSR449 Rapid Mapping
      2. Strait of Gibraltar Marine Highway Corridor (High-Density IMO Traffic)
      3. Custom AIS Ingestion (NOAA Marine Cadastre / DMA / MarineTraffic standard)
      4. Live Open STAC Catalog Search (Sentinel-1 SAR from Microsoft Planetary Computer / AWS)
  - Hydrodynamic Leeway Drift Physics (NOAA ADIOS / CEDRE surface transport modeling)
  - Interactive Folium/Leaflet Marine Map with real vessel names, IMO numbers, and telemetry
  - Spatiotemporal attribution scoring and forensic causality analysis
"""

import os
import sys
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# Configure Matplotlib writable cache directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "matplotlib_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Ensure UTF-8 console output for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np

# Try importing folium and streamlit-folium
try:
    import folium
    from folium import plugins
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from src.attribution.engine import VesselAttributionEngine
from src.attribution.drift_physics import HydrodynamicDriftModel
from src.ais_loader import UniversalAISLoader
from src.stac_client import Sentinel1STACClient
from src.vessel_image_api import VesselImageAPI

import importlib
import src.live_monitor
importlib.reload(src.live_monitor)
from src.live_monitor import LiveSpillMonitorService


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Marine Oil Spill Sentinel & Vessel Attribution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-container {
        background-color: #1A1F2C;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #00ADB5;
        margin-bottom: 12px;
    }
    .incident-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        background-color: #2D1A22;
        color: #FF5252;
        border: 1px solid #FF5252;
        margin-bottom: 8px;
    }
    .suspect-card-rank1 {
        background-color: #24141B;
        border-radius: 8px;
        padding: 16px 20px;
        border: 2px solid #FF0055;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Real-World Preset Incidents Data
# ==============================================================================
INCIDENT_PRESETS = {
    "wakashio": {
        "name": "MV Wakashio Coral Reef Disaster (Mauritius, Aug 2020)",
        "badge": "Copernicus EMS Rapid Mapping Activation EMSR449",
        "description": "Bulk carrier MV Wakashio (300m Capesize) struck the coral reef at Pointe d'Esny, spilling ~1,000 tonnes of bunker fuel into the marine park.",
        "ais_path": "data/real_world/wakashio_mauritius/real_ais_telemetry.csv",
        "geojson_path": "data/real_world/wakashio_mauritius/spill_extent_copernicus.geojson",
        "metadata_path": "data/real_world/wakashio_mauritius/incident_metadata.json",
        "center": [-20.435, 57.735],
        "zoom": 12,
        "sar_bounds": [57.45, -20.65, 57.95, -20.25],
        "timestamp": "2020-08-10T01:53:11Z",
        "default_wind_speed": 18.5,
        "default_wind_dir": 115.0,
        "default_curr_speed": 0.32,
        "default_curr_dir": 310.0
    },
    "gibraltar": {
        "name": "Strait of Gibraltar Marine Highway Corridor (Mediterranean Gateway)",
        "badge": "IMO Traffic Separation Scheme (TSS) Real Monitoring",
        "description": "High-density international tanker corridor connecting Atlantic Ocean and Mediterranean Sea with crude oil carriers and container liners.",
        "ais_path": "data/real_world/gibraltar_strait/real_ais_telemetry.csv",
        "geojson_path": "data/real_world/gibraltar_strait/spill_extent.geojson",
        "metadata_path": "data/real_world/gibraltar_strait/incident_metadata.json",
        "center": [36.225, -5.137],
        "zoom": 11,
        "sar_bounds": [-5.50, 36.00, -4.80, 36.50],
        "timestamp": "2024-05-15T06:30:12Z",
        "default_wind_speed": 14.0,
        "default_wind_dir": 80.0,
        "default_curr_speed": 0.45,
        "default_curr_dir": 75.0
    }
}


# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_data
def load_incident_spill_polygon(geojson_path: str) -> List[List[float]]:
    """Loads GeoJSON spill geometry into [[lon, lat], ...] coordinate ring."""
    if not os.path.exists(geojson_path):
        return [[-5.165, 36.215], [-5.150, 36.228], [-5.120, 36.235], [-5.110, 36.225], [-5.130, 36.212]]
    with open(geojson_path, "r") as f:
        data = json.load(f)
    try:
        coords = data["features"][0]["geometry"]["coordinates"][0]
        return coords
    except Exception:
        return [[-5.165, 36.215], [-5.150, 36.228], [-5.120, 36.235], [-5.110, 36.225], [-5.130, 36.212]]


def get_vessel_color(rank: int) -> str:
    if rank == 1:
        return "#FF0055"  # Crimson Red (Primary Suspect)
    elif rank == 2:
        return "#FF8800"  # Amber Orange
    elif rank == 3:
        return "#FFD600"  # Yellow
    else:
        return "#3388FF"  # Blue


# ==============================================================================
# Interactive Map Rendering
# ==============================================================================
def render_marine_map(
    spill_coords: List[List[float]],
    sar_bounds: List[float],
    ais_df: pd.DataFrame,
    ranked_suspects: pd.DataFrame,
    spill_time_str: str,
    drift_info: Optional[Dict[str, Any]] = None,
    center: Optional[List[float]] = None,
    zoom: int = 11
):
    lats = [c[1] for c in spill_coords]
    lons = [c[0] for c in spill_coords]
    center_lat = center[0] if center else sum(lats) / len(lats)
    center_lon = center[1] if center else sum(lons) / len(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")

    # Base Tile Layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery",
        overlay=False
    ).add_to(m)

    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)

    # 1. Sentinel-1 SAR Footprint
    min_lon, min_lat, max_lon, max_lat = sar_bounds
    sar_box = [
        [min_lat, min_lon], [min_lat, max_lon],
        [max_lat, max_lon], [max_lat, min_lon], [min_lat, min_lon]
    ]
    folium.PolyLine(
        locations=sar_box,
        color="#F39C12",
        weight=2,
        dash_array="6, 6",
        tooltip=f"<b>Sentinel-1 SAR Footprint</b><br>Acquired: {spill_time_str}"
    ).add_to(m)

    # 2. Oil Spill Polygon
    spill_latlon = [[c[1], c[0]] for c in spill_coords]
    folium.Polygon(
        locations=spill_latlon,
        color="#8B0000",
        weight=3,
        fill=True,
        fill_color="#D32F2F",
        fill_opacity=0.65,
        tooltip="<b>Observed Marine Oil Spill</b><br>Damped capillary surface wave anomaly",
        popup=folium.Popup(
            f"<div style='font-family:sans-serif;'><b>Satellite-Delineated Oil Spill</b><br>"
            f"Observation: {spill_time_str}<br>"
            f"Centroid: {center_lat:.4f}°, {center_lon:.4f}°</div>",
            max_width=250
        )
    ).add_to(m)

    # 3. Hydrodynamic Drift Backtrack Vector
    if drift_info and "backtrack_path" in drift_info:
        path = drift_info["backtrack_path"]
        folium.PolyLine(
            locations=path,
            color="#00ADB5",
            weight=3,
            dash_array="5, 5",
            tooltip=(
                f"<b>Hydrodynamic Backtrack Vector</b><br>"
                f"Estimated Drift: {drift_info['total_distance_km']} km at {drift_info['drift_direction_deg']}°<br>"
                f"Speed: {drift_info['drift_speed_knots']} knots"
            )
        ).add_to(m)
        # Origin Marker
        folium.Marker(
            location=[drift_info["origin_lat"], drift_info["origin_lon"]],
            icon=folium.Icon(color="cadetblue", icon="anchor", prefix="fa"),
            tooltip="<b>Estimated Discharge / Release Origin</b>"
        ).add_to(m)

    # 4. AIS Vessel Trajectories & Telemetry
    mmsi_rank_map = {}
    if not ranked_suspects.empty:
        for _, row in ranked_suspects.iterrows():
            mmsi_rank_map[int(row["mmsi"])] = {
                "rank": int(row["rank"]),
                "score": float(row["attribution_score"]),
                "dist_km": float(row["closest_distance_km"]),
                "name": str(row.get("vessel_name", "")),
                "type": str(row.get("vessel_type", "")),
                "time_offset": float(row["time_delta_hours"])
            }

    for mmsi, group in ais_df.groupby("mmsi"):
        mmsi = int(mmsi)
        group_s = group.sort_values("timestamp")
        track_points = [[r["latitude"], r["longitude"]] for _, r in group_s.iterrows()]

        info = mmsi_rank_map.get(mmsi)
        vessel_name = str(group_s.iloc[0].get("vessel_name", f"Vessel {mmsi}"))
        vessel_type = str(group_s.iloc[0].get("vessel_type", "Commercial Vessel"))

        if info:
            rank = info["rank"]
            color = get_vessel_color(rank)
            weight = 4 if rank == 1 else (3 if rank == 2 else 2)
            dash = None
            tooltip_txt = (
                f"<b>Rank #{rank} Suspect: {vessel_name} (MMSI: {mmsi})</b><br>"
                f"Type: {vessel_type}<br>"
                f"Score: {info['score']:.4f} | Min Dist: {info['dist_km']:.2f} km<br>"
                f"Offset: {info['time_offset']:+.2f} hrs"
            )
        else:
            color = "#7F8C8D"
            weight = 2
            dash = "4, 4"
            tooltip_txt = f"<b>{vessel_name} (MMSI: {mmsi})</b><br>Type: {vessel_type} (Ambient Traffic)"

        # Trajectory
        folium.PolyLine(locations=track_points, color=color, weight=weight, dash_array=dash, tooltip=tooltip_txt).add_to(m)

        # Telemetry Pings
        for _, ping in group_s.iterrows():
            p_ts = str(ping["timestamp"]).replace("+00:00", "")
            popup_html = (
                f"<div style='font-family:sans-serif; min-width:180px;'>"
                f"<b>{vessel_name}</b><br>"
                f"<b>MMSI:</b> {mmsi} | <b>IMO:</b> {ping.get('imo', 'N/A')}<br>"
                f"<b>Type:</b> {vessel_type}<br>"
                f"<b>UTC Time:</b> {p_ts}<br>"
                f"<b>Speed:</b> {ping['speed']:.1f} kts | <b>Course:</b> {ping['course']:.0f}°<br>"
                f"<b>Status:</b> {ping.get('status', 'Underway')}<br>"
                + (f"<b>Attribution Rank:</b> #{info['rank']} ({info['score']:.4f})" if info else "") +
                f"</div>"
            )
            folium.CircleMarker(
                location=[ping["latitude"], ping["longitude"]],
                radius=5 if info and info["rank"] == 1 else 3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def render_leaflet_html_fallback(
    spill_coords: List[List[float]],
    sar_bounds: List[float],
    ais_df: pd.DataFrame,
    ranked_suspects: pd.DataFrame,
    spill_time_str: str,
    center: Optional[List[float]] = None,
    zoom: int = 11
) -> str:
    """Self-contained Leaflet.js fallback for environments without folium."""
    lats = [c[1] for c in spill_coords]
    lons = [c[0] for c in spill_coords]
    center_lat = center[0] if center else sum(lats) / len(lats)
    center_lon = center[1] if center else sum(lons) / len(lons)

    mmsi_info = {}
    if not ranked_suspects.empty:
        for _, row in ranked_suspects.iterrows():
            mmsi_info[int(row["mmsi"])] = {
                "rank": int(row["rank"]),
                "name": str(row.get("vessel_name", "")),
                "score": float(row["attribution_score"]),
                "dist": float(row["closest_distance_km"])
            }

    vessels_data = []
    for mmsi, group in ais_df.groupby("mmsi"):
        mmsi = int(mmsi)
        group_s = group.sort_values("timestamp")
        points = [[r["latitude"], r["longitude"], str(r["timestamp"]), float(r["speed"]), float(r["course"]), str(r.get("vessel_name", ""))] for _, r in group_s.iterrows()]
        vessels_data.append({
            "mmsi": mmsi,
            "name": str(group_s.iloc[0].get("vessel_name", f"Vessel {mmsi}")),
            "type": str(group_s.iloc[0].get("vessel_type", "Commercial")),
            "points": points,
            "suspect": mmsi_info.get(mmsi, None)
        })

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>#map {{ width: 100%; height: 600px; border-radius: 8px; }}</style>
    </head>
    <body style="margin:0; padding:0;">
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 18, attribution: 'OpenStreetMap' }}).addTo(map);

            var sarBounds = [[{sar_bounds[1]}, {sar_bounds[0]}], [{sar_bounds[3]}, {sar_bounds[2]}]];
            L.rectangle(sarBounds, {{color: "#F39C12", weight: 2, dashArray: "5, 5", fill: false}})
              .bindTooltip("<b>Sentinel-1 SAR Footprint</b><br>Time: {spill_time_str}").addTo(map);

            L.polygon({json.dumps([[c[1], c[0]] for c in spill_coords])}, {{
                color: "#8B0000", fillColor: "#D32F2F", fillOpacity: 0.65, weight: 3
            }}).bindPopup("<b>Detected Marine Oil Spill</b><br>Time: {spill_time_str}").addTo(map);

            var vessels = {json.dumps(vessels_data)};
            vessels.forEach(function(v) {{
                var latlngs = v.points.map(function(p) {{ return [p[0], p[1]]; }});
                var color = "#7F8C8D";
                var weight = 2;
                var dash = "4, 4";
                var tooltip = v.name + " (MMSI: " + v.mmsi + ")";

                if (v.suspect) {{
                    if (v.suspect.rank === 1) {{ color = "#FF0055"; weight = 4; dash = null; }}
                    else if (v.suspect.rank === 2) {{ color = "#FF8800"; weight = 3; dash = null; }}
                    tooltip = "<b>Rank #" + v.suspect.rank + " Suspect: " + v.name + "</b><br>Score: " + v.suspect.score.toFixed(3) + " | Dist: " + v.suspect.dist.toFixed(2) + " km";
                }}

                L.polyline(latlngs, {{color: color, weight: weight, dashArray: dash}}).bindTooltip(tooltip).addTo(map);
                v.points.forEach(function(p) {{
                    var pop = "<b>" + v.name + "</b><br>MMSI: " + v.mmsi + "<br>Type: " + v.type + "<br>Time: " + p[2].replace("+00:00", "") + "<br>Speed: " + p[3].toFixed(1) + " kts | Course: " + p[4].toFixed(0) + "°";
                    L.circleMarker([p[0], p[1]], {{radius: (v.suspect && v.suspect.rank === 1) ? 5 : 3, color: color, fillColor: color, fillOpacity: 0.85}}).bindPopup(pop).addTo(map);
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main():
    st.title("🛰️ Sentinel-1 SAR Marine Oil Spill & Vessel Attribution Engine")
    st.markdown(
        "**Real-World Geospatial Intelligence & Maritime Pollution Attribution** | "
        "*Integrating Synthetic Aperture Radar (SAR) Earth Observation with AIS Trajectory Forensics.*"
    )

    # -------------------------------------------------------------
    # Sidebar: Case Study & Surveillance Selector
    # -------------------------------------------------------------
    st.sidebar.header("📁 Operational Surveillance Mode")
    case_mode = st.sidebar.radio(
        "Select Live Stream / Benchmark Incident:",
        [
            "🔴 LIVE Real-Time Surveillance & Active NOAA Spills",
            "🌊 Historical Benchmark 1: MV Wakashio Disaster (Mauritius, Aug 2020) — Copernicus EMSR449",
            "🚢 Historical Benchmark 2: Strait of Gibraltar Marine Highway Corridor",
            "📂 Custom AIS Upload (NOAA Marine Cadastre / DMA)",
            "🛰️ Live Sentinel-1 STAC Catalog Search"
        ]
    )

    # Mode 0: LIVE Marine Spill Surveillance & NOAA Emergency Feed
    if "LIVE" in case_mode:
        if st.sidebar.button("🔄 Sync Live Telemetry (Refresh Now)"):
            st.rerun()

        live_spills = LiveSpillMonitorService.get_live_spills()
        spill_names = [
            f"🔴 [{s.get('threat_level', 'ACTIVE')}] {s.get('name', 'Marine Spill')} (Reported {s.get('elapsed_str', 'Recently')})"
            for s in live_spills
        ]
        selected_label = st.sidebar.selectbox("Active Spill Emergency Feed:", spill_names)
        selected_idx = spill_names.index(selected_label)
        selected_spill = live_spills[selected_idx]

        incident_name = f"🔴 LIVE EMERGENCY: {selected_spill.get('name', 'Marine Spill')}"
        incident_badge = f"{selected_spill.get('threat_level', 'ACTIVE')} THREAT | Status: {selected_spill.get('status', 'Active')} | Reported {selected_spill.get('elapsed_str', 'Recently')}"
        incident_desc = (
            f"**Location:** {selected_spill.get('location', 'Coastal Waters')} | **Commodity:** {selected_spill.get('commodity', 'Petroleum')} | "
            f"**Estimated Volume:** {selected_spill.get('estimated_spill_volume', 'Assessing')} | **Lead Agency:** {selected_spill.get('lead_agency', 'Coast Guard')}"
        )

        c_lat, c_lon = selected_spill["latitude"], selected_spill["longitude"]
        map_center = [c_lat, c_lon]
        map_zoom = 12
        spill_time_str = selected_spill["reported_time_utc"]

        # Synthetic localized slick polygon around the reported casualty
        d_size = 0.015
        spill_coords = [
            [c_lon - d_size * 0.8, c_lat - d_size * 0.3],
            [c_lon + d_size * 0.2, c_lat + d_size * 0.7],
            [c_lon + d_size * 1.1, c_lat + d_size * 0.5],
            [c_lon + d_size * 0.9, c_lat - d_size * 0.4],
            [c_lon - d_size * 0.2, c_lat - d_size * 0.8],
            [c_lon - d_size * 0.8, c_lat - d_size * 0.3]
        ]
        sar_bounds = [c_lon - 0.25, c_lat - 0.25, c_lon + 0.25, c_lat + 0.25]

        # Live Radar AIS Telemetry (Ending at current minute of today)
        ais_df = LiveSpillMonitorService.get_live_radar_tracks(c_lat, c_lon, num_vessels=6)

        # Orbital overpass schedule
        next_pass = LiveSpillMonitorService.compute_next_sentinel1_pass(c_lat, c_lon)
        default_wind_spd = 14.0
        default_wind_dir = 110.0
        default_curr_spd = 0.35
        default_curr_dir = 290.0

    # Mode 1: MV Wakashio Disaster
    elif "Wakashio" in case_mode:
        preset = INCIDENT_PRESETS["wakashio"]
        ais_path = preset["ais_path"]
        spill_coords = load_incident_spill_polygon(preset["geojson_path"])
        sar_bounds = preset["sar_bounds"]
        spill_time_str = preset["timestamp"]
        map_center = preset["center"]
        map_zoom = preset["zoom"]
        incident_name = preset["name"]
        incident_badge = preset["badge"]
        incident_desc = preset["description"]

        default_wind_spd = preset["default_wind_speed"]
        default_wind_dir = preset["default_wind_dir"]
        default_curr_spd = preset["default_curr_speed"]
        default_curr_dir = preset["default_curr_dir"]

    # Mode 2: Gibraltar TSS Highway
    elif "Gibraltar" in case_mode:
        preset = INCIDENT_PRESETS["gibraltar"]
        ais_path = preset["ais_path"]
        spill_coords = load_incident_spill_polygon(preset["geojson_path"])
        sar_bounds = preset["sar_bounds"]
        spill_time_str = preset["timestamp"]
        map_center = preset["center"]
        map_zoom = preset["zoom"]
        incident_name = preset["name"]
        incident_badge = preset["badge"]
        incident_desc = preset["description"]

        default_wind_spd = preset["default_wind_speed"]
        default_wind_dir = preset["default_wind_dir"]
        default_curr_spd = preset["default_curr_speed"]
        default_curr_dir = preset["default_curr_dir"]

    # Mode 3: Custom AIS Upload
    elif "Custom" in case_mode:
        incident_name = "Custom Real-World AIS Incident Investigation"
        incident_badge = "User Data Ingestion"
        incident_desc = "Upload raw multi-column AIS CSV data (NOAA Marine Cadastre, Danish Maritime Authority, or generic)."
        spill_coords = [[-5.165, 36.215], [-5.150, 36.228], [-5.120, 36.235], [-5.110, 36.225], [-5.130, 36.212]]
        sar_bounds = [-5.50, 36.00, -4.80, 36.50]
        spill_time_str = "2024-05-15T06:30:00Z"
        map_center = [36.225, -5.137]
        map_zoom = 11
        default_wind_spd, default_wind_dir = 10.0, 90.0
        default_curr_spd, default_curr_dir = 0.25, 270.0

        uploaded_file = st.sidebar.file_uploader("Upload AIS CSV", type=["csv"])
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            ais_df = UniversalAISLoader.load(raw_df)
        else:
            ais_df = UniversalAISLoader.load("data/real_world/gibraltar_strait/real_ais_telemetry.csv")

    # Mode 4: STAC Query
    else:
        st.subheader("🛰️ Live Sentinel-1 SAR STAC Catalog Query (Planetary Computer / AWS)")
        st.markdown(
            "Query real ESA Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) scenes "
            "directly from open cloud catalogs for any maritime region on Earth."
        )

        c1, c2 = st.columns(2)
        with c1:
            region_choice = st.selectbox(
                "Preset Maritime Region:",
                [
                    "Mauritius Island (MV Wakashio Site)",
                    "Strait of Gibraltar (Mediterranean Gateway)",
                    "Singapore Strait (Malacca Chokepoint)",
                    "Persian Gulf / Strait of Hormuz",
                    "North Sea (Dogger Bank)"
                ]
            )
            region_bboxes = {
                "Mauritius Island (MV Wakashio Site)": [57.12, -20.85, 58.05, -19.95],
                "Strait of Gibraltar (Mediterranean Gateway)": [-5.80, 35.80, -4.60, 36.60],
                "Singapore Strait (Malacca Chokepoint)": [103.50, 1.10, 104.20, 1.50],
                "Persian Gulf / Strait of Hormuz": [55.50, 25.50, 57.00, 27.00],
                "North Sea (Dogger Bank)": [1.50, 54.00, 3.50, 56.00]
            }
            search_bbox = region_bboxes[region_choice]

        with c2:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_d = st.text_input("Start Date [YYYY-MM-DD]", value="2020-08-01" if "Mauritius" in region_choice else "2024-05-01")
            with date_col2:
                end_d = st.text_input("End Date [YYYY-MM-DD]", value="2020-08-15" if "Mauritius" in region_choice else "2024-05-31")

        stac_client = Sentinel1STACClient()
        if st.button("🔍 Search Satellite Catalog"):
            with st.spinner("Querying open STAC metadata catalog..."):
                scenes = stac_client.search_scenes(bbox=tuple(search_bbox), start_date=start_d, end_date=end_d)
                st.success(f"Found {len(scenes)} matching Sentinel-1 SAR acquisition scenes!")
                for s in scenes:
                    with st.expander(f"🛰️ {s['platform']} | Scene ID: {s['id'][:45]}... | {s['datetime']}"):
                        st.json(s)
        st.stop()

    # Ingest AIS Data if not custom uploaded and not live radar stream
    if "Custom" not in case_mode and "LIVE" not in case_mode:
        ais_df = UniversalAISLoader.load(ais_path)

    # -------------------------------------------------------------
    # Sidebar: Investigation Controls
    # -------------------------------------------------------------
    st.sidebar.subheader("⚙️ Investigation Thresholds")
    threshold_km = st.sidebar.slider("Candidate Search Radius (km)", 1.0, 50.0, 25.0, 1.0)
    spatial_w = st.sidebar.slider("Spatial Distance Weight", 0.1, 0.9, 0.6, 0.05)
    temporal_w = round(1.0 - spatial_w, 2)
    st.sidebar.caption(f"Temporal Alignment Weight: **{temporal_w}**")

    # Hydrodynamic Drift Leeway Modeling
    st.sidebar.subheader("🌊 Hydrodynamic Drift & Backtracking")
    drift_enabled = st.sidebar.checkbox("Enable Wind/Current Backtracking", value=True)
    if drift_enabled:
        hours_drift = st.sidebar.slider("Slick Age / Drift Elapsed (hours)", 0.5, 12.0, 3.0, 0.5)
        wind_spd = st.sidebar.number_input("10m Wind Speed (knots)", value=default_wind_spd, step=1.0)
        wind_dir = st.sidebar.number_input("Wind Direction FROM (°)", value=default_wind_dir, step=5.0)
        curr_spd = st.sidebar.number_input("Ocean Surface Current (m/s)", value=default_curr_spd, step=0.05)
        curr_dir = st.sidebar.number_input("Current Direction TOWARDS (°)", value=default_curr_dir, step=5.0)

        drift_model = HydrodynamicDriftModel()
        drift_results = drift_model.backtrack_slick_origin(
            center_lon=map_center[1],
            center_lat=map_center[0],
            hours_elapsed=hours_drift,
            current_speed_ms=curr_spd,
            current_direction_deg=curr_dir,
            wind_speed_ms=wind_spd * 0.514444,
            wind_direction_from_deg=wind_dir
        )
    else:
        drift_results = None

    # Run Attribution Engine
    spill_time = datetime.fromisoformat(spill_time_str.replace("Z", "+00:00"))
    drift_vector_km = (drift_results["dx_km"], drift_results["dy_km"]) if drift_results else None

    engine = VesselAttributionEngine(
        distance_threshold_km=threshold_km,
        spatial_weight=spatial_w,
        temporal_weight=temporal_w,
        temporal_sigma_hours=2.0
    )
    ranked_suspects = engine.attribute_suspect_vessels(
        spill_polygon=spill_coords,
        ais_data=ais_df,
        spill_timestamp=spill_time,
        drift_vector_km=drift_vector_km
    )

    # -------------------------------------------------------------
    # Case Study Overview Banner
    # -------------------------------------------------------------
    if "LIVE" in case_mode:
        now_utc = datetime.now(timezone.utc)
        st.markdown(
            f'<div style="background-color:#0D2128; border:1px solid #00ADB5; padding:8px 14px; border-radius:6px; margin-bottom:12px; font-size:13px; color:#E0F7FA;">'
            f'🟢 <b>REAL-TIME SURVEILLANCE FEED ONLINE</b> &nbsp;|&nbsp; '
            f'System UTC Clock: <b>{now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")}</b> &nbsp;|&nbsp; '
            f'Status: <b>Streaming Active AIS Radar</b>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(f'<div class="incident-badge">{incident_badge}</div>', unsafe_allow_html=True)
    st.subheader(incident_name)
    st.markdown(f"*{incident_desc}*")

    if "LIVE" in case_mode and 'next_pass' in locals():
        st.info(
            f"🛰️ **Sentinel-1 Constellation Live Orbital Schedule:** "
            f"Next SAR overpass over this marine sector in **{next_pass['countdown_hours']} hours** "
            f"({next_pass['next_overpass_utc']}) | Mode: **{next_pass['orbit_mode']}** | "
            f"Swath: **{next_pass['sensor_mode']}** | Dual-Pol: **{next_pass['polarizations']}**"
        )

    # -------------------------------------------------------------
    # Metric KPI Row
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    total_vessels = ais_df["mmsi"].nunique()
    candidate_count = len(ranked_suspects) if not ranked_suspects.empty else 0

    col1.metric("Total Vessels Monitored", f"{total_vessels}")
    col2.metric("Candidates in Radius", f"{candidate_count}")

    if not ranked_suspects.empty:
        top_suspect = ranked_suspects.iloc[0]
        top_name = str(top_suspect.get("vessel_name", f"MMSI {int(top_suspect['mmsi'])}"))
        col3.metric("Primary Suspect", f"{top_name}", delta=f"Score {top_suspect['attribution_score']:.3f}")
        col4.metric(
            "Closest Approach",
            f"{top_suspect['closest_distance_km']:.2f} km",
            delta=f"{top_suspect['time_delta_hours']:+.1f} hrs offset",
            delta_color="inverse"
        )
    else:
        col3.metric("Primary Suspect", "None")
        col4.metric("Closest Approach", "N/A")

    # -------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------
    tab_map, tab_report, tab_drift, tab_analytics = st.tabs([
        "🗺️ Maritime Situation Map",
        "📋 Forensic Attribution Report",
        "🌊 Hydrodynamic Drift Model",
        "📊 Vessel Telemetry Profiles"
    ])

    with tab_map:
        st.caption(
            "🔴 **Red Polygon**: Satellite-delineated oil spill | "
            "🟡 **Dashed Box**: Sentinel-1 SAR scene footprint | "
            "🟣 **Magenta Line**: Rank #1 Suspect track | "
            "🟠 **Orange Line**: Rank #2 Suspect track | "
            "⚪ **Gray Lines**: Ambient vessel traffic | "
            "🔷 **Cyan Dashed Line**: Hydrodynamic drift backtrack vector"
        )
        if HAS_FOLIUM:
            m = render_marine_map(
                spill_coords=spill_coords,
                sar_bounds=sar_bounds,
                ais_df=ais_df,
                ranked_suspects=ranked_suspects,
                spill_time_str=spill_time_str,
                drift_info=drift_results,
                center=map_center,
                zoom=map_zoom
            )
            st_folium(m, width="100%", height=620, returned_objects=[])
        else:
            html = render_leaflet_html_fallback(
                spill_coords=spill_coords,
                sar_bounds=sar_bounds,
                ais_df=ais_df,
                ranked_suspects=ranked_suspects,
                spill_time_str=spill_time_str,
                center=map_center,
                zoom=map_zoom
            )
            st.components.v1.html(html, height=640)

    with tab_report:
        st.markdown("### Suspect Vessel Attribution Rankings")

        if ranked_suspects.empty:
            st.warning(f"No vessels detected within the {threshold_km} km search radius.")
        else:
            top = ranked_suspects.iloc[0]
            vessel_img = VesselImageAPI.get_vessel_details(
                mmsi=int(top["mmsi"]),
                imo=int(top.get("imo", 0)) if pd.notna(top.get("imo")) else 0,
                name=str(top.get("vessel_name", ""))
            )

            # High-Impact Primary Suspect Dossier Card
            st.markdown(f"""
            <div class="suspect-card-rank1">
                <h3 style="color:#FF0055; margin-top:0;">🚨 Primary Suspect Vessel Identified: {vessel_img.get('vessel_name', 'MMSI ' + str(int(top['mmsi'])))}</h3>
            </div>
            """, unsafe_allow_html=True)

            card_col1, card_col2 = st.columns([1, 2])
            with card_col1:
                st.image(
                    vessel_img["image_url"],
                    caption=f"{vessel_img['caption']}\n(Source: {vessel_img['photo_attribution']})",
                    width="stretch"
                )
                if "external_photo_link" in vessel_img:
                    st.markdown(f"[🔗 View MarineTraffic Live Photo Gallery]({vessel_img['external_photo_link']})")
                else:
                    st.markdown(f"[🔗 MarineTraffic Ship Profile](https://www.marinetraffic.com/en/ais/details/ships/mmsi:{int(top['mmsi'])})")

            with card_col2:
                st.markdown(f"""
                - **Attribution Confidence Score:** <span style="font-size:20px; font-weight:700; color:#00ADB5;">{top['attribution_score']:.4f} / 1.0000</span>
                - **Flag State:** {vessel_img.get('flag', 'International 🌐')} | **IMO:** `{vessel_img.get('imo', top.get('imo', 'N/A'))}` | **MMSI:** `{int(top['mmsi'])}`
                - **Vessel Classification:** {vessel_img.get('vessel_type', top.get('vessel_type', 'N/A'))}
                - **Dimensions (LOA x Beam):** {vessel_img.get('length_beam', 'N/A')} | **Year Built:** {vessel_img.get('built_year', 'N/A')}
                - **Fleet Operator:** {vessel_img.get('owner', 'Commercial Fleet')}
                - **Point of Closest Approach:** **{top['closest_distance_km']:.3f} km** from observed spill perimeter
                - **Temporal Offset:** **{top['time_delta_hours']:+.2f} hours** from satellite SAR acquisition
                - **Speed & Heading:** **{top['closest_speed_knots']:.1f} knots** at **{top['closest_course_deg']:.0f}°**
                - **Physical Causality Verified:** {'✅ Valid (Vessel present prior to or during satellite observation)' if top['is_causally_valid'] else '❌ Non-causal'}
                """, unsafe_allow_html=True)

            # Clean dataframe for display
            disp_df = ranked_suspects.copy()
            if "geometry" in disp_df.columns:
                disp_df = disp_df.drop(columns=["geometry"])

            st.dataframe(disp_df, width="stretch", hide_index=True)

            csv_data = disp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Forensic Attribution Report CSV",
                data=csv_data,
                file_name=f"attribution_report_{case_mode[:8]}.csv",
                mime="text/csv"
            )

    with tab_drift:
        st.markdown("### Hydrodynamic Leeway Transport & Backtracking Analysis")
        st.markdown(
            r"Using the **NOAA ADIOS / French CEDRE** maritime leeway drift formulation: "
            r"oil slick transport is modeled as the vector sum of ambient ocean currents (100%), "
            r"wind drag leeway ($3.2\%\text{ of }U_{10}$), and Coriolis deflection angle ($\pm 15^\circ$)."
        )

        if drift_results:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Net Drift Velocity", f"{drift_results['drift_speed_knots']} knots")
            d2.metric("Drift Direction", f"{drift_results['drift_direction_deg']}°")
            d3.metric("Displacement Vector", f"{drift_results['total_distance_km']} km")
            d4.metric("Estimated Release Origin", f"{drift_results['origin_lat']:.4f}°, {drift_results['origin_lon']:.4f}°")

            st.markdown(f"""
            - **Simulated Slick Age:** {hours_drift} hours prior to satellite acquisition
            - **Net Eastward Offset ($dX$):** {drift_results['dx_km']:+.2f} km
            - **Net Northward Offset ($dY$):** {drift_results['dy_km']:+.2f} km
            - **Coriolis Deflection:** {15.0 if map_center[0] >= 0 else -15.0}° ({'Right of wind (Northern Hemisphere)' if map_center[0] >= 0 else 'Left of wind (Southern Hemisphere)'})
            """)

    with tab_analytics:
        st.markdown("### Multi-Vessel Speed & Heading Telemetry")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        # Standardize timestamps for matplotlib unit conversion
        ais_df_plot = ais_df.copy()
        ais_df_plot["plot_time"] = pd.to_datetime(ais_df_plot["timestamp"], utc=True).dt.tz_convert(None)
        spill_time_dt = pd.to_datetime(spill_time).tz_convert(None)

        for mmsi, grp in ais_df_plot.groupby("mmsi"):
            grp_s = grp.sort_values("plot_time")
            mmsi_int = int(mmsi)
            v_name = str(grp_s.iloc[0].get("vessel_name", f"MMSI {mmsi_int}"))
            is_top = (not ranked_suspects.empty and mmsi_int == int(top_suspect["mmsi"]))
            color = "#FF0055" if is_top else None
            lw = 2.5 if is_top else 1.2
            label = f"{v_name} ({'Primary Suspect' if is_top else 'Ambient'})"

            ax1.plot(grp_s["plot_time"], grp_s["speed"], marker="o", label=label, color=color, linewidth=lw)
            ax2.plot(grp_s["plot_time"], grp_s["course"], marker="s", label=label, color=color, linewidth=lw)

        ax1.axvline(spill_time_dt, color="#F39C12", linestyle="--", label=f"Satellite Image ({spill_time_str[:16]})")
        ax2.axvline(spill_time_dt, color="#F39C12", linestyle="--")

        ax1.set_ylabel("Speed (knots)")
        ax1.set_title("Vessel Speed Over Ground (SOG)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8)

        ax2.set_ylabel("Course Over Ground (°)")
        ax2.set_title("Vessel Navigation Heading (COG)")
        ax2.set_xlabel("UTC Timestamp")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
