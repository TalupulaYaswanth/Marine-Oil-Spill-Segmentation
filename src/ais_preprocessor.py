"""
Module: ais_preprocessor.py
Description: Ingestion, validation, coordinate transformation, and spatiotemporal 
             filtering of Automatic Identification System (AIS) vessel trajectories.
             Engineered with multi-backend support (GeoPandas + pure Pandas fallbacks)
             to guarantee 100% operational execution on any platform.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union, Dict, Any, List

import pandas as pd
import numpy as np

# Try importing geopandas and shapely
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon, box
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False

    class Point:  # type: ignore
        def __init__(self, x_or_coord, y=None):
            if y is None:
                self.x, self.y = x_or_coord[0], x_or_coord[1]
            else:
                self.x, self.y = float(x_or_coord), float(y)
            self.coords = [(self.x, self.y)]
        def __repr__(self):
            return f"POINT ({self.x} {self.y})"

    class LineString:  # type: ignore
        def __init__(self, coords):
            self.coords = list(coords)
        def __repr__(self):
            return f"LINESTRING ({', '.join(f'{c[0]} {c[1]}' for c in self.coords)})"

    class Polygon:  # type: ignore
        def __init__(self, coords):
            self.coords = list(coords)
        def __repr__(self):
            return f"POLYGON ({len(self.coords)} vertices)"


def export_gdf_or_df_to_geojson(df: pd.DataFrame, output_path: str) -> None:
    """Exports GeoDataFrame or fallback Pandas DataFrame with geometry to GeoJSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if HAS_GEOSPATIAL and hasattr(df, "to_file"):
        try:
            df.to_file(output_path, driver="GeoJSON")
            return
        except Exception:
            pass

    features = []
    for _, row in df.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue

        if hasattr(geom, "coords"):
            coords = list(geom.coords)
        else:
            coords = []

        if hasattr(geom, "__class__") and "LineString" in geom.__class__.__name__:
            geom_type = "LineString"
            geometry_dict = {"type": "LineString", "coordinates": coords}
        elif hasattr(geom, "__class__") and "Point" in geom.__class__.__name__:
            geom_type = "Point"
            geometry_dict = {"type": "Point", "coordinates": [geom.x, geom.y]}
        else:
            geom_type = "Point"
            geometry_dict = {"type": "Point", "coordinates": [float(row.get("longitude", 0)), float(row.get("latitude", 0))]}

        props = {}
        for col in df.columns:
            if col == "geometry":
                continue
            val = row[col]
            if isinstance(val, (datetime, pd.Timestamp)):
                props[col] = val.isoformat()
            elif isinstance(val, (np.integer, int)):
                props[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                props[col] = float(val) if not np.isnan(val) else None
            else:
                props[col] = str(val) if pd.notna(val) else None

        features.append({
            "type": "Feature",
            "geometry": geometry_dict,
            "properties": props
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, indent=2)


class AISPreprocessor:
    """
    Handles parsing historical AIS CSV logs, converting them into standardized
    GeoDataFrames (or Pandas DataFrame fallbacks), and performing spatiotemporal
    window filtering against satellite imagery acquisitions.
    """

    REQUIRED_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude", "speed", "course"]

    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs

    def load_ais_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Loads AIS CSV file, validates required schema, standardizes timestamps,
        and constructs GeoDataFrame (or DataFrame).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"AIS CSV file not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in AIS CSV: {missing_cols}. Expected: {self.REQUIRED_COLUMNS}")

        df = df.dropna(subset=["mmsi", "timestamp", "latitude", "longitude"])
        df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

        df["mmsi"] = df["mmsi"].astype(int)
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
        df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if HAS_GEOSPATIAL:
            try:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                    crs=self.crs
                )
                return gdf
            except Exception:
                pass

        # Fallback to pure pandas DataFrame with Point objects
        df["geometry"] = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
        return df

    def filter_spatiotemporal(
        self,
        ais_gdf: pd.DataFrame,
        roi_bounds_wgs84: Tuple[float, float, float, float],
        roi_polygon: Optional[Any] = None,
        target_timestamp: Optional[datetime] = None,
        time_window_hours: float = 3.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        spatial_buffer_deg: float = 0.05
    ) -> pd.DataFrame:
        """
        Filters AIS pings that intersect the satellite scene spatially and temporally.
        """
        if ais_gdf.empty:
            return ais_gdf.copy()

        # 1. Temporal Filtering
        if start_time is None or end_time is None:
            if target_timestamp is None:
                raise ValueError("Must provide either (start_time, end_time) or target_timestamp.")
            if target_timestamp.tzinfo is None:
                target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
            delta = timedelta(hours=time_window_hours)
            t_min = target_timestamp - delta
            t_max = target_timestamp + delta
        else:
            t_min = start_time.replace(tzinfo=timezone.utc) if start_time.tzinfo is None else start_time
            t_max = end_time.replace(tzinfo=timezone.utc) if end_time.tzinfo is None else end_time

        temporal_mask = (ais_gdf["timestamp"] >= t_min) & (ais_gdf["timestamp"] <= t_max)
        temp_filtered = ais_gdf[temporal_mask].copy()

        if temp_filtered.empty:
            return temp_filtered

        # 2. Spatial Filtering
        min_lon, min_lat, max_lon, max_lat = roi_bounds_wgs84
        b_min_lon = min_lon - spatial_buffer_deg
        b_min_lat = min_lat - spatial_buffer_deg
        b_max_lon = max_lon + spatial_buffer_deg
        b_max_lat = max_lat + spatial_buffer_deg

        if HAS_GEOSPATIAL and hasattr(temp_filtered, "cx"):
            try:
                spatial_filtered = temp_filtered.cx[b_min_lon:b_max_lon, b_min_lat:b_max_lat].copy()
                if roi_polygon is not None and hasattr(roi_polygon, "within") and not spatial_filtered.empty:
                    search_poly = roi_polygon.buffer(spatial_buffer_deg) if spatial_buffer_deg > 0 else roi_polygon
                    spatial_filtered = spatial_filtered[spatial_filtered.geometry.within(search_poly)].copy()
                return spatial_filtered.sort_values(by=["mmsi", "timestamp"]).reset_index(drop=True)
            except Exception:
                pass

        # Robust coordinate bounding box mask
        spatial_mask = (
            (temp_filtered["longitude"] >= b_min_lon) &
            (temp_filtered["longitude"] <= b_max_lon) &
            (temp_filtered["latitude"] >= b_min_lat) &
            (temp_filtered["latitude"] <= b_max_lat)
        )
        spatial_filtered = temp_filtered[spatial_mask].copy()

        return spatial_filtered.sort_values(by=["mmsi", "timestamp"]).reset_index(drop=True)

    def reconstruct_trajectories(self, filtered_ais_gdf: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates individual AIS point pings into continuous LineString trajectories per vessel MMSI.
        """
        if filtered_ais_gdf.empty:
            cols = ["mmsi", "ping_count", "start_time", "end_time", "avg_speed", "max_speed", "geometry"]
            if HAS_GEOSPATIAL:
                try:
                    return gpd.GeoDataFrame(columns=cols, crs=self.crs)
                except Exception:
                    pass
            return pd.DataFrame(columns=cols)

        trajectory_records = []
        for mmsi, group in filtered_ais_gdf.groupby("mmsi"):
            sorted_group = group.sort_values("timestamp")
            coords = list(zip(sorted_group["longitude"], sorted_group["latitude"]))

            if len(coords) >= 2:
                geom = LineString(coords)
            else:
                geom = Point(coords[0])

            trajectory_records.append({
                "mmsi": int(mmsi),
                "ping_count": len(sorted_group),
                "start_time": sorted_group["timestamp"].min(),
                "end_time": sorted_group["timestamp"].max(),
                "avg_speed": float(sorted_group["speed"].mean()),
                "max_speed": float(sorted_group["speed"].max()),
                "geometry": geom
            })

        if HAS_GEOSPATIAL:
            try:
                return gpd.GeoDataFrame(trajectory_records, crs=self.crs)
            except Exception:
                pass

        return pd.DataFrame(trajectory_records)
