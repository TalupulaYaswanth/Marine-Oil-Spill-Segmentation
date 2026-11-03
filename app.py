"""
app.py
Streamlit Interactive Dashboard for Marine Oil Spill Segmentation & Vessel Attribution.

Features:
  - Interactive Folium/Leaflet marine map.
  - Satellite SAR scene boundary and segmented oil spill polygon overlay.
  - Spatiotemporal AIS vessel tracks colored by suspect ranking.
  - Point of Closest Approach markers with maritime telemetry.
  - Sidebar controls for dynamic thresholding and attribution weighting.
  - Forensic suspect ranking table and deep-dive card.
"""

import os
import sys
import json
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


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Marine Oil Spill & Vessel Attribution Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #00ADB5;
        margin-bottom: 12px;
    }
    .suspect-card-rank1 {
        background-color: #2D1A22;
        border-radius: 8px;
        padding: 18px;
        border: 2px solid #FF0055;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Data Loading & Helper Functions
# ==============================================================================
DEFAULT_SPILL_POLYGON = [
    [-5.165, 36.215],
    [-5.150, 36.228],
    [-5.120, 36.235],
    [-5.110, 36.225],
    [-5.130, 36.212],
    [-5.165, 36.215]
]

DEFAULT_SAR_BOUNDS = [-5.50, 36.00, -4.80, 36.50]  # min_lon, min_lat, max_lon, max_lat
DEFAULT_TIMESTAMP = "2024-05-15T06:30:00Z"


@st.cache_data
def load_ais_data(ais_path: str) -> pd.DataFrame:
    """Loads historical AIS dataset."""
    if not os.path.exists(ais_path):
        st.error(f"AIS file not found at {ais_path}. Run Phase 1 data generation first.")
        return pd.DataFrame()
    df = pd.read_csv(ais_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def compute_attribution(
    ais_df: pd.DataFrame,
    spill_coords: List[List[float]],
    spill_time: datetime,
    threshold_km: float,
    spatial_w: float,
    temporal_w: float
) -> pd.DataFrame:
    """Executes attribution engine."""
    engine = VesselAttributionEngine(
        distance_threshold_km=threshold_km,
        spatial_weight=spatial_w,
        temporal_weight=temporal_w,
        temporal_sigma_hours=2.0
    )
    return engine.attribute_suspect_vessels(
        spill_polygon=spill_coords,
        ais_data=ais_df,
        spill_timestamp=spill_time
    )


# ==============================================================================
# Map Rendering Functions (Folium + HTML Fallback)
# ==============================================================================
def get_vessel_color(rank: int) -> str:
    """Returns distinctive visual colors for suspect rankings."""
    if rank == 1:
        return "#FF0055"  # Crimson Red (Primary Suspect)
    elif rank == 2:
        return "#FF8800"  # Amber Orange (Secondary Suspect)
    elif rank == 3:
        return "#FFD600"  # Yellow
    else:
        return "#3388FF"  # Blue


def create_folium_map(
    spill_coords: List[List[float]],
    sar_bounds: List[float],
    ais_df: pd.DataFrame,
    ranked_suspects: pd.DataFrame,
    spill_time_str: str
):
    """Constructs Folium map with layers for SAR bounds, spill mask, and AIS trajectories."""
    # Compute center
    lats = [c[1] for c in spill_coords]
    lons = [c[0] for c in spill_coords]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # Base layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Overlay Satellite Scene Footprint (Bounding Box)
    min_lon, min_lat, max_lon, max_lat = sar_bounds
    sar_box = [
        [min_lat, min_lon],
        [min_lat, max_lon],
        [max_lat, max_lon],
        [max_lat, min_lon],
        [min_lat, min_lon]
    ]
    folium.PolyLine(
        locations=sar_box,
        color="#F39C12",
        weight=2,
        dash_array="6, 6",
        tooltip=f"<b>Sentinel-1 SAR Footprint</b><br>Acquisition: {spill_time_str}"
    ).add_to(m)

    # 2. Overlay Oil Spill Polygon
    spill_latlon = [[c[1], c[0]] for c in spill_coords]
    folium.Polygon(
        locations=spill_latlon,
        color="#8B0000",
        weight=3,
        fill=True,
        fill_color="#D32F2F",
        fill_opacity=0.65,
        tooltip="<b>Detected Marine Oil Spill</b><br>Damped backscatter anomaly",
        popup=folium.Popup(
            f"<div style='font-family:sans-serif;'><b>Marine Oil Slick</b><br>"
            f"Observation: {spill_time_str}<br>"
            f"Centroid: {center_lat:.4f}°N, {center_lon:.4f}°W</div>",
            max_width=250
        )
    ).add_to(m)

    # Centroid Marker
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=4,
        color="#000000",
        fill=True,
        fill_color="#FFFFFF",
        tooltip="Spill Centroid"
    ).add_to(m)

    # Create lookup map for rankings
    mmsi_rank_map = {}
    if not ranked_suspects.empty:
        for _, row in ranked_suspects.iterrows():
            mmsi_rank_map[int(row["mmsi"])] = {
                "rank": int(row["rank"]),
                "score": float(row["attribution_score"]),
                "dist_km": float(row["closest_distance_km"]),
                "time_offset": float(row["time_delta_hours"]),
                "closest_ts": str(row["closest_timestamp"])
            }

    # 3. Plot AIS Vessel Tracks and Pings
    for mmsi, group in ais_df.groupby("mmsi"):
        mmsi = int(mmsi)
        group_sorted = group.sort_values("timestamp")
        track_points = [[r["latitude"], r["longitude"]] for _, r in group_sorted.iterrows()]

        info = mmsi_rank_map.get(mmsi)
        if info:
            rank = info["rank"]
            color = get_vessel_color(rank)
            weight = 4 if rank == 1 else (3 if rank == 2 else 2)
            dash = None
            tooltip_txt = (
                f"<b>Rank #{rank} Suspect (MMSI: {mmsi})</b><br>"
                f"Score: {info['score']:.4f}<br>"
                f"Min Dist: {info['dist_km']:.2f} km<br>"
                f"Time Delta: {info['time_offset']:+.2f} hrs"
            )
        else:
            color = "#7F8C8D"  # Neutral Gray
            weight = 2
            dash = "4, 4"
            tooltip_txt = f"Vessel MMSI: {mmsi} (Non-suspect)"

        # Trajectory Line
        folium.PolyLine(
            locations=track_points,
            color=color,
            weight=weight,
            dash_array=dash,
            tooltip=tooltip_txt
        ).add_to(m)

        # Plot individual pings
        for _, ping in group_sorted.iterrows():
            p_ts = str(ping["timestamp"]).replace("+00:00", "")
            popup_html = (
                f"<div style='font-family:sans-serif; min-width:160px;'>"
                f"<b>MMSI:</b> {mmsi}<br>"
                f"<b>Timestamp (UTC):</b> {p_ts}<br>"
                f"<b>Speed:</b> {ping['speed']:.1f} knots<br>"
                f"<b>Course:</b> {ping['course']:.0f}°<br>"
                + (f"<b>Rank:</b> #{info['rank']} ({info['score']:.3f})" if info else "") +
                f"</div>"
            )
            folium.CircleMarker(
                location=[ping["latitude"], ping["longitude"]],
                radius=4 if info and info["rank"] == 1 else 3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def render_leaflet_html(
    spill_coords: List[List[float]],
    sar_bounds: List[float],
    ais_df: pd.DataFrame,
    ranked_suspects: pd.DataFrame,
    spill_time_str: str
) -> str:
    """Self-contained Leaflet HTML generator for environments without streamlit-folium."""
    lats = [c[1] for c in spill_coords]
    lons = [c[0] for c in spill_coords]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    spill_geojson_coords = [[[c[0], c[1]] for c in spill_coords]]

    # Prepare suspect lookup
    mmsi_info = {}
    if not ranked_suspects.empty:
        for _, row in ranked_suspects.iterrows():
            mmsi_info[int(row["mmsi"])] = {
                "rank": int(row["rank"]),
                "score": float(row["attribution_score"]),
                "dist": float(row["closest_distance_km"]),
                "offset": float(row["time_delta_hours"])
            }

    # Prepare vessels data payload
    vessels_data = []
    for mmsi, group in ais_df.groupby("mmsi"):
        mmsi = int(mmsi)
        group_s = group.sort_values("timestamp")
        points = [[r["latitude"], r["longitude"], str(r["timestamp"]), float(r["speed"]), float(r["course"])] for _, r in group_s.iterrows()]
        vessels_data.append({
            "mmsi": mmsi,
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
        <style>
            #map {{ width: 100%; height: 620px; border-radius: 8px; }}
            .legend {{ background: white; padding: 10px; border-radius: 5px; font-family: sans-serif; font-size: 12px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }}
            .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
            .legend-color {{ width: 14px; height: 14px; margin-right: 6px; display: inline-block; border-radius: 2px; }}
        </style>
    </head>
    <body style="margin:0; padding:0;">
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
            
            var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 18, attribution: 'OpenStreetMap'
            }}).addTo(map);

            var esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                maxZoom: 18, attribution: 'Esri Satellite'
            }});

            var baseLayers = {{ "OpenStreetMap": osm, "Satellite View": esri }};
            L.control.layers(baseLayers).addTo(map);

            // 1. SAR Scene Bounds
            var sarBounds = [[{sar_bounds[1]}, {sar_bounds[0]}], [{sar_bounds[3]}, {sar_bounds[2]}]];
            L.rectangle(sarBounds, {{color: "#F39C12", weight: 2, dashArray: "5, 5", fill: false}})
              .bindTooltip("<b>Sentinel-1 SAR Footprint</b><br>Acquisition: {spill_time_str}")
              .addTo(map);

            // 2. Oil Spill Polygon
            var spillPolygon = L.polygon({json.dumps([[c[1], c[0]] for c in spill_coords])}, {{
                color: "#8B0000",
                fillColor: "#D32F2F",
                fillOpacity: 0.65,
                weight: 3
            }}).bindPopup("<b>Detected Marine Oil Spill</b><br>Centroid: {center_lat:.4f}°N, {center_lon:.4f}°W<br>Time: {spill_time_str}").addTo(map);

            // 3. Vessel Tracks
            var vessels = {json.dumps(vessels_data)};
            vessels.forEach(function(v) {{
                var latlngs = v.points.map(function(p) {{ return [p[0], p[1]]; }});
                var color = "#7F8C8D";
                var weight = 2;
                var dash = "4, 4";
                var tooltipTxt = "Vessel MMSI: " + v.mmsi;

                if (v.suspect) {{
                    if (v.suspect.rank === 1) {{ color = "#FF0055"; weight = 4; dash = null; }}
                    else if (v.suspect.rank === 2) {{ color = "#FF8800"; weight = 3; dash = null; }}
                    else {{ color = "#FFD600"; weight = 3; dash = null; }}
                    tooltipTxt = "<b>Rank #" + v.suspect.rank + " Suspect (MMSI: " + v.mmsi + ")</b><br>Score: " + v.suspect.score.toFixed(3) + "<br>Min Dist: " + v.suspect.dist.toFixed(2) + " km";
                }}

                L.polyline(latlngs, {{color: color, weight: weight, dashArray: dash}})
                 .bindTooltip(tooltipTxt)
                 .addTo(map);

                v.points.forEach(function(p) {{
                    var popup = "<b>MMSI:</b> " + v.mmsi + "<br><b>Time:</b> " + p[2].replace("+00:00", "") + "<br><b>Speed:</b> " + p[3].toFixed(1) + " kts<br><b>Course:</b> " + p[4].toFixed(0) + "°";
                    L.circleMarker([p[0], p[1]], {{radius: (v.suspect && v.suspect.rank === 1) ? 5 : 3, color: color, fillColor: color, fillOpacity: 0.85}})
                     .bindPopup(popup)
                     .addTo(map);
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html


# ==============================================================================
# Main Streamlit Application Layout
# ==============================================================================
def main():
    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------
    st.title("🛰️ Marine Oil Spill Segmentation & Vessel Attribution Dashboard")
    st.markdown(
        "**Deep Learning-Based Satellite Remote Sensing & AIS Telemetry Pipeline** | "
        "*Integrating Sentinel-1 SAR imagery with maritime spatiotemporal trajectory attribution.*"
    )

    # -------------------------------------------------------------
    # Sidebar: Configurations & Controls
    # -------------------------------------------------------------
    st.sidebar.header("⚙️ Investigation Parameters")

    ais_path = st.sidebar.text_input(
        "Historical AIS CSV Path",
        value="data/sample_ais.csv",
        help="Path to AIS log CSV (columns: mmsi, timestamp, latitude, longitude, speed, course)."
    )

    spill_time_input = st.sidebar.text_input(
        "Satellite Image Acquisition (UTC)",
        value=DEFAULT_TIMESTAMP,
        help="ISO 8601 acquisition timestamp of the Sentinel-1 SAR scene."
    )

    st.sidebar.subheader("Attribution Engine Filters")
    threshold_km = st.sidebar.slider(
        "Candidate Search Radius (km)",
        min_value=1.0,
        max_value=50.0,
        value=25.0,
        step=1.0,
        help="Maximum distance between vessel track and oil spill boundary to classify as candidate."
    )

    spatial_w = st.sidebar.slider(
        "Spatial Distance Weight",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Weight assigned to spatial proximity in composite scoring."
    )
    temporal_w = round(1.0 - spatial_w, 2)
    st.sidebar.caption(f"Temporal Alignment Weight: **{temporal_w}**")

    # Drift Vector Simulation
    with st.sidebar.expander("🌊 Ocean Current & Wind Drift Modeling"):
        drift_enabled = st.checkbox("Enable Spill Backtracking Drift", value=False)
        drift_x = st.number_input("Drift dX (Eastward km)", value=1.2, step=0.1) if drift_enabled else 0.0
        drift_y = st.number_input("Drift dY (Northward km)", value=0.5, step=0.1) if drift_enabled else 0.0

    # Load Data & Run Attribution
    spill_time = datetime.fromisoformat(spill_time_input.replace("Z", "+00:00"))
    ais_df = load_ais_data(ais_path)

    if ais_df.empty:
        st.stop()

    drift_vec = (drift_x, drift_y) if drift_enabled else None
    ranked_suspects = compute_attribution(
        ais_df=ais_df,
        spill_coords=DEFAULT_SPILL_POLYGON,
        spill_time=spill_time,
        threshold_km=threshold_km,
        spatial_w=spatial_w,
        temporal_w=temporal_w
    )

    # -------------------------------------------------------------
    # Top Metrics Row
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    total_vessels = ais_df["mmsi"].nunique()
    candidate_count = len(ranked_suspects) if not ranked_suspects.empty else 0

    col1.metric("Vessels Monitored", f"{total_vessels}")
    col2.metric("Candidates in Radius", f"{candidate_count}")

    if not ranked_suspects.empty:
        top_suspect = ranked_suspects.iloc[0]
        col3.metric("Primary Suspect MMSI", f"{int(top_suspect['mmsi'])}")
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
    # Main Body: Map & Forensic Attribution Report
    # -------------------------------------------------------------
    tab_map, tab_report, tab_analytics = st.tabs(["🗺️ Geospatial Map View", "📋 Forensic Attribution Report", "📊 Trajectory Analytics"])

    with tab_map:
        st.markdown("### Interactive Maritime Situation Map")
        st.caption(
            "🔴 **Crimson Polygon**: Detected oil spill | "
            "🟡 **Dashed Box**: Sentinel-1 SAR footprint | "
            "🟣 **Magenta Line**: Rank #1 Suspect track | "
            "🟠 **Orange Line**: Rank #2 Suspect track | "
            "⚪ **Gray Lines**: Ambient vessel traffic"
        )

        if HAS_FOLIUM:
            m = create_folium_map(
                spill_coords=DEFAULT_SPILL_POLYGON,
                sar_bounds=DEFAULT_SAR_BOUNDS,
                ais_df=ais_df,
                ranked_suspects=ranked_suspects,
                spill_time_str=spill_time_input
            )
            st_folium(m, width="100%", height=620, returned_objects=[])
        else:
            # Fallback to embedded Leaflet.js
            leaflet_html = render_leaflet_html(
                spill_coords=DEFAULT_SPILL_POLYGON,
                sar_bounds=DEFAULT_SAR_BOUNDS,
                ais_df=ais_df,
                ranked_suspects=ranked_suspects,
                spill_time_str=spill_time_input
            )
            st.components.v1.html(leaflet_html, height=640)

    with tab_report:
        st.markdown("### Suspect Vessel Attribution Rankings")

        if ranked_suspects.empty:
            st.warning(f"No vessels found within the {threshold_km} km threshold radius.")
        else:
            # Highlight top suspect
            st.markdown(f"""
            <div class="suspect-card-rank1">
                <h3 style="color:#FF0055; margin-top:0;">🚨 Primary Suspect Identified: MMSI {int(top_suspect['mmsi'])}</h3>
                <p><b>Attribution Confidence Score:</b> <span style="font-size:18px; color:#00ADB5;">{top_suspect['attribution_score']:.4f} / 1.0000</span></p>
                <ul>
                    <li><b>Closest Approach Distance:</b> {top_suspect['closest_distance_km']:.3f} km from spill perimeter</li>
                    <li><b>Temporal Offset:</b> {top_suspect['time_delta_hours']:+.2f} hours from satellite observation time</li>
                    <li><b>Timestamp of Closest Ping:</b> {top_suspect['closest_timestamp']}</li>
                    <li><b>Vessel Speed & Heading:</b> {top_suspect['closest_speed_knots']:.1f} knots at course {top_suspect['closest_course_deg']:.0f}°</li>
                    <li><b>Physical Causality Verified:</b> {'✅ Valid (Vessel present prior to/during satellite imaging)' if top_suspect['is_causally_valid'] else '❌ Non-causal'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Table of all candidates
            display_df = ranked_suspects.copy()
            if "geometry" in display_df.columns:
                display_df = display_df.drop(columns=["geometry"])

            display_df["mmsi"] = display_df["mmsi"].astype(str)
            display_df["attribution_score"] = display_df["attribution_score"].map(lambda x: f"{x:.4f}")
            display_df["closest_distance_km"] = display_df["closest_distance_km"].map(lambda x: f"{x:.2f} km")
            display_df["time_delta_hours"] = display_df["time_delta_hours"].map(lambda x: f"{x:+.2f} h")
            display_df["closest_speed_knots"] = display_df["closest_speed_knots"].map(lambda x: f"{x:.1f} kts")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # Export button
            csv_data = ranked_suspects.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Attribution Report CSV",
                data=csv_data,
                file_name="spill_attribution_report.csv",
                mime="text/csv"
            )

    with tab_analytics:
        st.markdown("### AIS Spatiotemporal Trajectory Analysis")

        # Telemetry Chart: Speed vs Time per vessel
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for mmsi, grp in ais_df.groupby("mmsi"):
            grp_s = grp.sort_values("timestamp")
            mmsi_int = int(mmsi)
            is_top = (not ranked_suspects.empty and mmsi_int == int(top_suspect["mmsi"]))
            label = f"MMSI {mmsi_int} ({'Primary Suspect' if is_top else 'Ambient'})"
            color = "#FF0055" if is_top else None
            lw = 2.5 if is_top else 1.2
            ax.plot(grp_s["timestamp"], grp_s["speed"], marker="o", label=label, color=color, linewidth=lw)

        ax.axvline(spill_time, color="#F39C12", linestyle="--", label=f"SAR Image Time ({spill_time_input[:16]})")
        ax.set_ylabel("Speed Over Ground (knots)")
        ax.set_title("Vessel Speed Profiles Over Satellite Acquisition Window")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()

