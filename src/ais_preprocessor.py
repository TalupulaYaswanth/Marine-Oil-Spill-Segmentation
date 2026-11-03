"""
Module: ais_preprocessor.py
Description: Ingestion, validation, coordinate transformation, and spatiotemporal 
             filtering of Automatic Identification System (AIS) vessel trajectories.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union, Dict, Any, List

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box


class AISPreprocessor:
    """
    Handles parsing historical AIS CSV logs, converting them into GeoPandas GeoDataFrames,
    and performing spatiotemporal window filtering against satellite imagery acquisitions.
    """

    REQUIRED_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude", "speed", "course"]

    def __init__(self, crs: str = "EPSG:4326"):
        """
        Args:
            crs: Target coordinate reference system for AIS data (default: EPSG:4326 WGS84).
        """
        self.crs = crs

    def load_ais_csv(self, csv_path: str) -> gpd.GeoDataFrame:
        """
        Loads AIS CSV file, validates required schema, standardizes timestamps,
        and constructs a GeoPandas GeoDataFrame.

        Args:
            csv_path: Path to the AIS CSV file.

        Returns:
            gpd.GeoDataFrame with Point geometries in EPSG:4326.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"AIS CSV file not found at: {csv_path}")

        df = pd.read_csv(csv_path)

        # Standardize column names (lowercase, strip whitespace)
        df.columns = [c.strip().lower() for c in df.columns]

        # Verify required columns exist
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in AIS CSV: {missing_cols}. Expected: {self.REQUIRED_COLUMNS}")

        # Drop invalid coordinates and nulls
        df = df.dropna(subset=["mmsi", "timestamp", "latitude", "longitude"])
        df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

        # Ensure correct data types
        df["mmsi"] = df["mmsi"].astype(int)
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
        df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)

        # Convert timestamps to UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Build GeoDataFrame with Point geometry
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=self.crs
        )

        return gdf

    def filter_spatiotemporal(
        self,
        ais_gdf: gpd.GeoDataFrame,
        roi_bounds_wgs84: Tuple[float, float, float, float],
        roi_polygon: Optional[Polygon] = None,
        target_timestamp: Optional[datetime] = None,
        time_window_hours: float = 3.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        spatial_buffer_deg: float = 0.05
    ) -> gpd.GeoDataFrame:
        """
        Filters AIS pings that intersect the satellite scene spatially and temporally.

        Args:
            ais_gdf: Raw AIS GeoDataFrame in EPSG:4326.
            roi_bounds_wgs84: Bounding box tuple (min_lon, min_lat, max_lon, max_lat).
            roi_polygon: Optional exact scene polygon footprint for fine intersection.
            target_timestamp: Satellite image acquisition timestamp.
            time_window_hours: Half-window in hours around target_timestamp [T - w, T + w].
            start_time: Explicit start datetime (overrides time_window_hours if provided).
            end_time: Explicit end datetime (overrides time_window_hours if provided).
            spatial_buffer_deg: Optional geographic buffer in degrees to include vessels
                                just outside the frame (critical for drift-back attribution).

        Returns:
            Filtered gpd.GeoDataFrame containing only relevant AIS records.
        """
        if ais_gdf.empty:
            return ais_gdf.copy()

        # -------------------------------------------------------------
        # 1. Temporal Filtering
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 2. Spatial Filtering
        # -------------------------------------------------------------
        min_lon, min_lat, max_lon, max_lat = roi_bounds_wgs84
        
        # Apply spatial buffer (expanded bounding box)
        buffered_bounds = (
            min_lon - spatial_buffer_deg,
            min_lat - spatial_buffer_deg,
            max_lon + spatial_buffer_deg,
            max_lat + spatial_buffer_deg
        )
        
        # Fast spatial index bounding box filtering
        spatial_filtered = temp_filtered.cx[
            buffered_bounds[0]:buffered_bounds[2],
            buffered_bounds[1]:buffered_bounds[3]
        ].copy()

        # If an exact footprint polygon is provided, refine intersection
        if roi_polygon is not None and not spatial_filtered.empty:
            search_poly = roi_polygon.buffer(spatial_buffer_deg) if spatial_buffer_deg > 0 else roi_polygon
            spatial_filtered = spatial_filtered[spatial_filtered.geometry.within(search_poly)].copy()

        return spatial_filtered.sort_values(by=["mmsi", "timestamp"]).reset_index(drop=True)

    def reconstruct_trajectories(self, filtered_ais_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregates individual AIS point pings into continuous LineString trajectories per vessel MMSI.

        Args:
            filtered_ais_gdf: Spatiotemporally filtered AIS GeoDataFrame.

        Returns:
            gpd.GeoDataFrame containing MMSI, trajectory LineString, start/end times, and ping counts.
        """
        if filtered_ais_gdf.empty:
            return gpd.GeoDataFrame(
                columns=["mmsi", "ping_count", "start_time", "end_time", "avg_speed", "geometry"],
                crs=self.crs
            )

        trajectory_records = []
        for mmsi, group in filtered_ais_gdf.groupby("mmsi"):
            sorted_group = group.sort_values("timestamp")
            coords = list(zip(sorted_group.geometry.x, sorted_group.geometry.y))

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

        return gpd.GeoDataFrame(trajectory_records, crs=self.crs)

