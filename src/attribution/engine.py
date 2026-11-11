"""
Module: engine.py
Description: Geospatial vessel detection, metric distance calculation, 
             and spatiotemporal attribution scoring engine for marine oil spills.
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Check availability of geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon
    from shapely.ops import nearest_points
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two geographic coordinates in kilometers.
    Used for geodesic calculations and fallback metric validation.
    """
    r = 6371.0088  # Mean Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class VesselAttributionEngine:
    """
    Correlates segmented oil spill geometries with spatiotemporal AIS vessel tracks
    to identify, score, and rank potential polluter vessels.
    """

    def __init__(
        self,
        distance_threshold_km: float = 5.0,
        spatial_weight: float = 0.6,
        temporal_weight: float = 0.4,
        temporal_sigma_hours: float = 2.0,
        crs_wgs84: str = "EPSG:4326"
    ):
        """
        Args:
            distance_threshold_km: Maximum radius (in km) from spill polygon to consider a vessel.
            spatial_weight: Weight of spatial proximity in the composite attribution score [0, 1].
            temporal_weight: Weight of temporal alignment in the composite attribution score [0, 1].
            temporal_sigma_hours: Scale parameter (in hours) for Gaussian temporal decay.
            crs_wgs84: Base CRS for coordinates (EPSG:4326).
        """
        self.distance_threshold_km = distance_threshold_km
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.temporal_sigma_hours = temporal_sigma_hours
        self.crs_wgs84 = crs_wgs84

    def attribute_suspect_vessels(
        self,
        spill_polygon: Any,
        ais_data: pd.DataFrame,
        spill_timestamp: datetime,
        drift_vector_km: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Calculates metric distances between vessel trajectories and the oil spill polygon,
        filters candidates within distance_threshold_km, and ranks them by attribution score.

        Args:
            spill_polygon: Shapely Polygon/MultiPolygon (or tuple/list of bounding coordinates).
            ais_data: DataFrame containing AIS tracks (mmsi, timestamp, latitude, longitude, speed, course).
            spill_timestamp: Acquisition datetime of the satellite image (UTC).
            drift_vector_km: Optional tuple of (dx_km, dy_km) for ocean/wind drift backtracking.

        Returns:
            Ranked DataFrame (or GeoDataFrame if geopandas is present) of suspect vessels.
        """
        if spill_timestamp.tzinfo is None:
            spill_timestamp = spill_timestamp.replace(tzinfo=timezone.utc)

        df = ais_data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if HAS_GEOSPATIAL and hasattr(spill_polygon, "is_empty") and not spill_polygon.is_empty:
            return self._attribute_with_geopandas(spill_polygon, df, spill_timestamp, drift_vector_km)
        else:
            return self._attribute_with_haversine(spill_polygon, df, spill_timestamp, drift_vector_km)

    def _attribute_with_geopandas(
        self,
        spill_polygon: "Polygon",
        df: pd.DataFrame,
        spill_timestamp: datetime,
        drift_vector_km: Optional[Tuple[float, float]]
    ) -> "gpd.GeoDataFrame":
        """Implementation using GeoPandas and local UTM projection for millimetric accuracy."""
        ais_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=self.crs_wgs84
        )

        local_utm = ais_gdf.estimate_utm_crs() or "EPSG:3857"
        spill_gdf = gpd.GeoDataFrame(geometry=[spill_polygon], crs=self.crs_wgs84)
        spill_projected = spill_gdf.to_crs(local_utm).geometry.iloc[0]
        ais_projected = ais_gdf.to_crs(local_utm)

        if drift_vector_km is not None:
            from shapely.affinity import translate
            origin_projected = translate(
                spill_projected,
                xoff=-drift_vector_km[0] * 1000.0,
                yoff=-drift_vector_km[1] * 1000.0
            )
        else:
            origin_projected = spill_projected

        suspect_records = []
        for mmsi, group in ais_projected.groupby("mmsi"):
            group_sorted = group.sort_values("timestamp")
            pings_geom = list(group_sorted.geometry)

            vessel_track = LineString([p.coords[0] for p in pings_geom]) if len(pings_geom) >= 2 else pings_geom[0]
            track_dist_km = vessel_track.distance(origin_projected) / 1000.0

            ping_distances = [p.distance(origin_projected) / 1000.0 for p in pings_geom]
            min_ping_idx = int(np.argmin(ping_distances))
            closest_ping = group_sorted.iloc[min_ping_idx]
            closest_dist_km = min(track_dist_km, ping_distances[min_ping_idx])

            if closest_dist_km > self.distance_threshold_km:
                continue

            time_delta_hours = (spill_timestamp - closest_ping["timestamp"]).total_seconds() / 3600.0
            causality_factor = 0.2 if time_delta_hours < -0.25 else 1.0

            spatial_score = float(np.exp(-0.5 * (closest_dist_km / (self.distance_threshold_km * 0.4)) ** 2))
            temporal_score = float(np.exp(-0.5 * (abs(time_delta_hours) / self.temporal_sigma_hours) ** 2))
            composite_score = (self.spatial_weight * spatial_score + self.temporal_weight * temporal_score) * causality_factor

            wgs84_coords = list(zip(group_sorted["longitude"], group_sorted["latitude"]))
            wgs84_geom = LineString(wgs84_coords) if len(wgs84_coords) >= 2 else Point(wgs84_coords[0])

            suspect_records.append({
                "mmsi": int(mmsi),
                "vessel_name": str(closest_ping.get("vessel_name", f"Vessel-{str(mmsi)[-4:]}")),
                "imo": int(closest_ping.get("imo", 0)) if pd.notna(closest_ping.get("imo")) else 0,
                "vessel_type": str(closest_ping.get("vessel_type", "Commercial Vessel")),
                "status": str(closest_ping.get("status", "Underway")),
                "attribution_score": round(float(composite_score), 4),
                "closest_distance_km": round(float(closest_dist_km), 3),
                "time_delta_hours": round(float(time_delta_hours), 2),
                "closest_timestamp": closest_ping["timestamp"].isoformat(),
                "closest_speed_knots": float(closest_ping.get("speed", 0.0)),
                "closest_course_deg": float(closest_ping.get("course", 0.0)),
                "total_pings": len(group_sorted),
                "is_causally_valid": time_delta_hours >= -0.25,
                "geometry": wgs84_geom
            })

        return self._finalize_rankings(suspect_records, as_gdf=True)

    def _attribute_with_haversine(
        self,
        spill_geom: Any,
        df: pd.DataFrame,
        spill_timestamp: datetime,
        drift_vector_km: Optional[Tuple[float, float]]
    ) -> pd.DataFrame:
        """Lightweight haversine geodesic calculation fallback."""
        # Determine center coordinates of the spill
        if hasattr(spill_geom, "centroid"):
            spill_lat, spill_lon = float(spill_geom.centroid.y), float(spill_geom.centroid.x)
        elif isinstance(spill_geom, (list, tuple)) and len(spill_geom) > 0:
            if isinstance(spill_geom[0], (list, tuple)):
                # List of (lon, lat) polygon ring vertices
                spill_lon = float(sum(p[0] for p in spill_geom) / len(spill_geom))
                spill_lat = float(sum(p[1] for p in spill_geom) / len(spill_geom))
            else:
                spill_lon, spill_lat = float(spill_geom[0]), float(spill_geom[1])
        else:
            spill_lat, spill_lon = 36.225, -5.137

        if drift_vector_km is not None:
            # Approx 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat)
            dlat = drift_vector_km[1] / 111.0
            dlon = drift_vector_km[0] / (111.0 * math.cos(math.radians(spill_lat)))
            spill_lat -= dlat
            spill_lon -= dlon

        suspect_records = []
        for mmsi, group in df.groupby("mmsi"):
            group_sorted = group.sort_values("timestamp")
            
            distances = [
                haversine_km(row["latitude"], row["longitude"], spill_lat, spill_lon)
                for _, row in group_sorted.iterrows()
            ]
            min_idx = int(np.argmin(distances))
            closest_dist_km = distances[min_idx]
            closest_ping = group_sorted.iloc[min_idx]

            if closest_dist_km > self.distance_threshold_km:
                continue

            time_delta_hours = (spill_timestamp - closest_ping["timestamp"]).total_seconds() / 3600.0
            causality_factor = 0.2 if time_delta_hours < -0.25 else 1.0

            spatial_score = float(np.exp(-0.5 * (closest_dist_km / (self.distance_threshold_km * 0.4)) ** 2))
            temporal_score = float(np.exp(-0.5 * (abs(time_delta_hours) / self.temporal_sigma_hours) ** 2))
            composite_score = (self.spatial_weight * spatial_score + self.temporal_weight * temporal_score) * causality_factor

            suspect_records.append({
                "mmsi": int(mmsi),
                "vessel_name": str(closest_ping.get("vessel_name", f"Vessel-{str(mmsi)[-4:]}")),
                "imo": int(closest_ping.get("imo", 0)) if pd.notna(closest_ping.get("imo")) else 0,
                "vessel_type": str(closest_ping.get("vessel_type", "Commercial Vessel")),
                "status": str(closest_ping.get("status", "Underway")),
                "attribution_score": round(float(composite_score), 4),
                "closest_distance_km": round(float(closest_dist_km), 3),
                "time_delta_hours": round(float(time_delta_hours), 2),
                "closest_timestamp": closest_ping["timestamp"].isoformat(),
                "closest_speed_knots": float(closest_ping.get("speed", 0.0)),
                "closest_course_deg": float(closest_ping.get("course", 0.0)),
                "total_pings": len(group_sorted),
                "is_causally_valid": time_delta_hours >= -0.25
            })

        return self._finalize_rankings(suspect_records, as_gdf=False)

    def _finalize_rankings(self, records: List[Dict[str, Any]], as_gdf: bool = True) -> Any:
        cols = [
            "rank", "mmsi", "vessel_name", "vessel_type", "attribution_score", "closest_distance_km",
            "time_delta_hours", "closest_speed_knots", "closest_course_deg",
            "is_causally_valid", "closest_timestamp", "imo", "status", "total_pings"
        ]
        if as_gdf:
            cols.append("geometry")

        if not records:
            empty_df = pd.DataFrame(columns=cols)
            return gpd.GeoDataFrame(empty_df, crs=self.crs_wgs84) if as_gdf and HAS_GEOSPATIAL else empty_df

        ranked = pd.DataFrame(records).sort_values(
            by=["attribution_score", "closest_distance_km"],
            ascending=[False, True]
        ).reset_index(drop=True)

        ranked["rank"] = range(1, len(ranked) + 1)
        ranked = ranked[cols]

        if as_gdf and HAS_GEOSPATIAL:
            return gpd.GeoDataFrame(ranked, crs=self.crs_wgs84)
        return ranked
