"""
generate_sample_data.py
Utility script to generate realistic sample AIS data and synthetic Sentinel-1 SAR GeoTIFF
for testing Phase 1 preprocessing.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for non-ASCII directory paths on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_sample_ais_csv(output_csv_path: str) -> str:
    """
    Generates a realistic sample AIS CSV file with vessels exhibiting different
    spatiotemporal relationships with a reference satellite acquisition:
      - Vessel 1 (MMSI 211234567): Passing through scene during acquisition (Suspect 1)
      - Vessel 2 (MMSI 244789012): Crossing scene during acquisition (Suspect 2)
      - Vessel 3 (MMSI 310555666): Inside scene but 12 hours later (Temporal Distractor)
      - Vessel 4 (MMSI 636091234): Far outside spatial scene during acquisition (Spatial Distractor)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    records = []

    # Reference scene time: 2024-05-15 06:30:00 UTC
    # Scene bounds roughly: Lon [-5.50, -4.80], Lat [36.00, 36.50]
    t_ref = datetime(2024, 5, 15, 6, 30, 0, tzinfo=timezone.utc)

    # 1. MMSI 211234567: Tanker heading East across the strait (05:00 to 08:00 UTC)
    start_t1 = t_ref - timedelta(hours=1.5)
    for i in range(13):  # 15-minute intervals
        t = start_t1 + timedelta(minutes=15 * i)
        lon = -5.45 + (i * 0.045)  # enters at -5.45, exits at -4.91 (Inside scene)
        lat = 36.12 + (i * 0.015)  # 36.12 to 36.30 (Inside scene)
        speed = 12.4 + np.random.uniform(-0.3, 0.3)
        course = 72.0 + np.random.uniform(-1.0, 1.0)
        records.append({
            "mmsi": 211234567,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "speed": round(speed, 1),
            "course": round(course, 1)
        })

    # 2. MMSI 244789012: Container ship heading West (05:45 to 07:30 UTC)
    start_t2 = t_ref - timedelta(minutes=45)
    for i in range(8):
        t = start_t2 + timedelta(minutes=15 * i)
        lon = -4.85 - (i * 0.07)  # enters at -4.85, exits at -5.34 (Inside scene)
        lat = 36.38 - (i * 0.01)  # 36.38 to 36.31 (Inside scene)
        speed = 18.2 + np.random.uniform(-0.5, 0.5)
        course = 260.0 + np.random.uniform(-1.5, 1.5)
        records.append({
            "mmsi": 244789012,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "speed": round(speed, 1),
            "course": round(course, 1)
        })

    # 3. MMSI 310555666: Tug in scene but at night (18:00 to 20:00 UTC) -> Temporal Distractor
    start_t3 = t_ref + timedelta(hours=12)
    for i in range(6):
        t = start_t3 + timedelta(minutes=20 * i)
        lon = -5.20 + (i * 0.01)  # Inside bounds
        lat = 36.20 + (i * 0.005) # Inside bounds
        speed = 6.0
        course = 90.0
        records.append({
            "mmsi": 310555666,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "speed": round(speed, 1),
            "course": round(course, 1)
        })

    # 4. MMSI 636091234: Bulk carrier at target time but far away (Alboran Sea) -> Spatial Distractor
    start_t4 = t_ref - timedelta(hours=1)
    for i in range(8):
        t = start_t4 + timedelta(minutes=15 * i)
        lon = -3.20 + (i * 0.05)  # Far outside scene (Lon ~ -3.2 vs Scene ~ -5.0)
        lat = 36.80 + (i * 0.01)  # Outside scene
        speed = 14.0
        course = 85.0
        records.append({
            "mmsi": 636091234,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "speed": round(speed, 1),
            "course": round(course, 1)
        })

    df = pd.DataFrame(records)
    # Shuffle slightly to emulate raw asynchronous AIS receiver feeds
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Generated sample AIS CSV ({len(df)} records) at: {output_csv_path}")
    return output_csv_path


def generate_sample_sar_geotiff(output_tif_path: str) -> Optional[str]:
    """
    Generates a realistic synthetic Sentinel-1 SAR GeoTIFF with:
      - Coordinate system: EPSG:4326 (WGS84)
      - Marine clutter background (Rayleigh / Gamma distributed backscatter)
      - Dark oil slick patch with low backscatter (smoothing capillary waves)
      - Bright vessel return points
      - NoData border margins
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        print("Note: rasterio is not installed yet. Skipping synthetic GeoTIFF creation.")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_tif_path)), exist_ok=True)

    # Image specs: 512 x 512 pixels
    height, width = 512, 512
    bounds = (-5.50, 36.00, -4.80, 36.50)  # min_lon, min_lat, max_lon, max_lat
    transform = from_bounds(*bounds, width, height)

    # 1. Base sea surface clutter (Rayleigh/Gamma distribution in linear intensity)
    # Sea backscatter typically ranges around -15 dB to -10 dB in VV
    np.random.seed(42)
    sea_clutter_linear = np.random.gamma(shape=3.0, scale=0.03, size=(height, width)).astype(np.float32)

    # 2. Add an oil slick anomaly (dark patch with ~6-10 dB damping)
    # Elliptical spill region centered at pixel (260, 240)
    y, x = np.ogrid[:height, :width]
    slick_mask = (((x - 260) / 45) ** 2 + ((y - 240) / 18) ** 2) <= 1.0
    sea_clutter_linear[slick_mask] *= 0.12  # drastic damping due to surfactant film

    # 3. Add bright vessel point scatterer
    vessel_x, vessel_y = 300, 235
    sea_clutter_linear[vessel_y-1:vessel_y+2, vessel_x-1:vessel_x+2] = 2.5  # high metallic return

    # 4. Add NoData border (margin = 15 pixels)
    nodata_val = -9999.0
    sea_clutter_linear[:15, :] = nodata_val
    sea_clutter_linear[-15:, :] = nodata_val
    sea_clutter_linear[:, :15] = nodata_val
    sea_clutter_linear[:, -15:] = nodata_val

    # Write GeoTIFF
    profile = {
        "driver": "GTiff",
        "dtype": rasterio.float32,
        "nodata": nodata_val,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform
    }

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(sea_clutter_linear, 1)
        # Store acquisition datetime in GeoTIFF tag
        dst.update_tags(
            ACQUISITION_DATETIME="2024-05-15T06:30:00Z",
            MISSION="SENTINEL-1A",
            POLARIZATION="VV",
            SENSOR_MODE="IW_GRDH"
        )

    print(f"Generated synthetic Sentinel-1 SAR GeoTIFF at: {output_tif_path}")
    return output_tif_path


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ais_file = os.path.join(data_dir, "sample_ais.csv")
    sar_file = os.path.join(data_dir, "S1A_IW_GRDH_1SDV_20240515T063000_sample.tif")

    generate_sample_ais_csv(ais_file)
    generate_sample_sar_geotiff(sar_file)
