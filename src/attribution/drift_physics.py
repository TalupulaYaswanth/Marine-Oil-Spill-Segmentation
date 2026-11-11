"""
Module: drift_physics.py
Description: Physical oil spill drift trajectory and backtracking model based on 
             NOAA ADIOS / CEDRE hydrodynamic leeway transport physics.
"""

import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class HydrodynamicDriftModel:
    """
    Simulates maritime surface oil transport incorporating:
      - Ocean surface currents (100% transport vector)
      - Atmospheric wind leeway (typically 3.0% - 3.5% of 10m wind speed)
      - Coriolis deflection angle (approx. +15° in Northern Hemisphere, -15° in Southern Hemisphere)
    """

    def __init__(self, wind_leeway_factor: float = 0.032, coriolis_enabled: bool = True):
        """
        Args:
            wind_leeway_factor: Ratio of surface slick drift speed to 10m wind speed (standard 0.03 - 0.035).
            coriolis_enabled: Whether to apply Coriolis angular deflection to the wind-driven component.
        """
        self.wind_leeway_factor = wind_leeway_factor
        self.coriolis_enabled = coriolis_enabled

    def compute_drift_velocity(
        self,
        current_speed_ms: float,
        current_direction_deg: float,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        latitude: float
    ) -> Tuple[float, float, float, float]:
        """
        Computes the net hydrodynamic transport velocity vector (u_east, v_north).

        Args:
            current_speed_ms: Ocean current velocity in m/s.
            current_direction_deg: Direction current is flowing TOWARDS (0° North, 90° East).
            wind_speed_ms: Wind speed in m/s.
            wind_direction_from_deg: Direction wind is coming FROM (meteorological convention).
            latitude: Geographic latitude in degrees (determines Coriolis deflection sign).

        Returns:
            Tuple of (net_speed_ms, net_direction_deg, u_east_ms, v_north_ms)
        """
        # 1. Current velocity vector (towards direction)
        curr_rad = math.radians(current_direction_deg)
        u_curr = current_speed_ms * math.sin(curr_rad)
        v_curr = current_speed_ms * math.cos(curr_rad)

        # 2. Wind drift vector (wind pushes oil DOWNWIND, so direction towards is direction_from + 180)
        wind_towards_deg = (wind_direction_from_deg + 180.0) % 360.0

        # Coriolis deflection: Wind-driven surface layer deflects to the right in NH (+15°), left in SH (-15°)
        if self.coriolis_enabled:
            coriolis_deflection = 15.0 if latitude >= 0 else -15.0
            wind_towards_deg = (wind_towards_deg + coriolis_deflection) % 360.0

        wind_rad = math.radians(wind_towards_deg)
        wind_drift_speed = wind_speed_ms * self.wind_leeway_factor

        u_wind = wind_drift_speed * math.sin(wind_rad)
        v_wind = wind_drift_speed * math.cos(wind_rad)

        # 3. Superposition: Net drift velocity
        u_net = u_curr + u_wind
        v_net = v_curr + v_wind

        net_speed = math.sqrt(u_net ** 2 + v_net ** 2)
        net_dir_rad = math.atan2(u_net, v_net)
        net_dir_deg = (math.degrees(net_dir_rad) + 360.0) % 360.0

        return net_speed, net_dir_deg, u_net, v_net

    def backtrack_slick_origin(
        self,
        center_lon: float,
        center_lat: float,
        hours_elapsed: float,
        current_speed_ms: float,
        current_direction_deg: float,
        wind_speed_ms: float,
        wind_direction_from_deg: float
    ) -> Dict[str, Any]:
        """
        Backtracks the observed slick centroid back in time to estimate the original discharge location.

        Returns:
            Dictionary with estimated release coordinates (origin_lon, origin_lat), displacement vector,
            and intermediate trajectory points.
        """
        net_speed, net_dir, u_net, v_net = self.compute_drift_velocity(
            current_speed_ms=current_speed_ms,
            current_direction_deg=current_direction_deg,
            wind_speed_ms=wind_speed_ms,
            wind_direction_from_deg=wind_direction_from_deg,
            latitude=center_lat
        )

        # Net displacement in meters over hours_elapsed
        total_seconds = hours_elapsed * 3600.0
        dx_m = u_net * total_seconds
        dy_m = v_net * total_seconds

        # Conversion to geographic degrees
        # 1 deg lat ~ 111,139 m; 1 deg lon ~ 111,139 * cos(lat) m
        lat_rad = math.radians(center_lat)
        dlat = dy_m / 111139.0
        dlon = dx_m / (111139.0 * math.cos(lat_rad))

        # Origin position (subtract displacement because backtracking into the past)
        origin_lat = center_lat - dlat
        origin_lon = center_lon - dlon

        # Discrete steps for path visualization
        steps = 10
        trajectory = []
        for i in range(steps + 1):
            fraction = i / float(steps)
            step_lat = center_lat - (dlat * fraction)
            step_lon = center_lon - (dlon * fraction)
            trajectory.append([step_lat, step_lon])

        return {
            "origin_lon": round(origin_lon, 5),
            "origin_lat": round(origin_lat, 5),
            "drift_speed_knots": round(net_speed * 1.94384, 2),
            "drift_direction_deg": round(net_dir, 1),
            "total_distance_km": round(math.sqrt(dx_m ** 2 + dy_m ** 2) / 1000.0, 2),
            "dx_km": round(dx_m / 1000.0, 3),
            "dy_km": round(dy_m / 1000.0, 3),
            "backtrack_path": trajectory
        }

