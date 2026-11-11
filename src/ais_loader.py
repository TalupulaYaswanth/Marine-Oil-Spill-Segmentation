"""
Module: ais_loader.py
Description: Universal multi-format loader and schema normalizer for real-world AIS data
             (NOAA Marine Cadastre, Danish Maritime Authority, Spire, MarineTraffic, generic).
"""

import os
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class UniversalAISLoader:
    """
    Auto-detects and standardizes real-world AIS telemetry datasets into a unified schema:
    [mmsi, timestamp, latitude, longitude, speed, course, vessel_name, imo, vessel_type]
    """

    COLUMN_MAPS = {
        # NOAA Marine Cadastre schema
        "noaa": {
            "mmsi": ["mmsi"],
            "timestamp": ["basedatetime", "datetime", "timestamp"],
            "latitude": ["lat", "latitude"],
            "longitude": ["lon", "longitude"],
            "speed": ["sog", "speed"],
            "course": ["cog", "course"],
            "heading": ["heading"],
            "vessel_name": ["vesselname", "name"],
            "imo": ["imo"],
            "vessel_type": ["vesseltype", "shiptype"]
        },
        # Danish Maritime Authority (DMA) schema
        "dma": {
            "mmsi": ["mmsi"],
            "timestamp": ["# timestamp", "timestamp"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon"],
            "speed": ["sog"],
            "course": ["cog"],
            "heading": ["heading"],
            "vessel_name": ["name"],
            "imo": ["imo"],
            "vessel_type": ["ship type", "shiptype"]
        }
    }

    VESSEL_TYPE_CODES = {
        70: "Cargo Vessel",
        71: "Cargo - Hazardous A",
        72: "Cargo - Hazardous B",
        79: "Cargo Vessel",
        80: "Oil/Chemical Tanker",
        81: "Tanker - Hazardous A",
        82: "Tanker - Hazardous B",
        89: "Tanker",
        30: "Fishing Vessel",
        31: "Towing / Tug",
        32: "Tug / Towing",
        50: "Pilot Vessel",
        51: "Search and Rescue (SAR)",
        52: "Tugboat",
        55: "Law Enforcement / Coast Guard",
        60: "Passenger Ship"
    }

    @classmethod
    def load(cls, file_path_or_df: Any) -> pd.DataFrame:
        """
        Loads CSV, Excel, or DataFrame, normalizes columns to standard lowercase,
        maps heterogeneous naming conventions, cleans outliers, and enforces UTC timestamps.
        """
        if isinstance(file_path_or_df, str):
            if not os.path.exists(file_path_or_df):
                raise FileNotFoundError(f"AIS file not found: {file_path_or_df}")
            df = pd.read_csv(file_path_or_df, low_memory=False)
        elif isinstance(file_path_or_df, pd.DataFrame):
            df = file_path_or_df.copy()
        else:
            raise ValueError("Input must be a valid file path string or pandas DataFrame.")

        # Clean column names (lowercase, strip whitespace)
        col_clean = {c: c.strip().lower() for c in df.columns}
        df.rename(columns=col_clean, inplace=True)

        standard_cols = {}
        # Auto-match columns
        for target, aliases in [
            ("mmsi", ["mmsi", "vessel_mmsi"]),
            ("timestamp", ["timestamp", "basedatetime", "# timestamp", "date_time_utc", "datetime"]),
            ("latitude", ["latitude", "lat", "y"]),
            ("longitude", ["longitude", "lon", "x"]),
            ("speed", ["speed", "sog", "speed_knots"]),
            ("course", ["course", "cog", "course_deg"]),
            ("vessel_name", ["vessel_name", "vesselname", "name", "shipname"]),
            ("imo", ["imo", "imo_number"]),
            ("vessel_type", ["vessel_type", "vesseltype", "ship_type", "shiptype", "cargo"])
        ]:
            found = False
            for alias in aliases:
                if alias in df.columns:
                    standard_cols[target] = df[alias]
                    found = True
                    break
            if not found and target in ["mmsi", "timestamp", "latitude", "longitude"]:
                raise ValueError(f"Required AIS field '{target}' not found in file columns: {list(df.columns)}")

        out_df = pd.DataFrame(standard_cols)

        # Drop invalid coordinates
        out_df = out_df.dropna(subset=["mmsi", "timestamp", "latitude", "longitude"])
        out_df["latitude"] = pd.to_numeric(out_df["latitude"], errors="coerce")
        out_df["longitude"] = pd.to_numeric(out_df["longitude"], errors="coerce")
        out_df = out_df[(out_df["latitude"].between(-90, 90)) & (out_df["longitude"].between(-180, 180))]

        # Numeric conversions
        out_df["mmsi"] = pd.to_numeric(out_df["mmsi"], errors="coerce").fillna(0).astype(np.int64)
        out_df["speed"] = pd.to_numeric(out_df.get("speed", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=60.0)
        out_df["course"] = pd.to_numeric(out_df.get("course", 0.0), errors="coerce").fillna(0.0) % 360.0

        # Enforce UTC timestamps
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True, errors="coerce")
        out_df = out_df.dropna(subset=["timestamp"])

        # Populate or format optional metadata
        if "vessel_name" not in out_df.columns:
            out_df["vessel_name"] = "Vessel-" + out_df["mmsi"].astype(str).str[-4:]
        else:
            out_df["vessel_name"] = out_df["vessel_name"].fillna("Unknown Vessel")

        if "imo" not in out_df.columns:
            out_df["imo"] = 0
        else:
            out_df["imo"] = pd.to_numeric(out_df["imo"], errors="coerce").fillna(0).astype(np.int64)

        if "vessel_type" not in out_df.columns:
            out_df["vessel_type"] = "Commercial Vessel"
        else:
            # Map numeric AIS type codes if present
            def map_vessel_type(val):
                if pd.isna(val):
                    return "Commercial Vessel"
                try:
                    num = int(val)
                    return cls.VESSEL_TYPE_CODES.get(num, f"Type {num}")
                except (ValueError, TypeError):
                    return str(val)
            out_df["vessel_type"] = out_df["vessel_type"].apply(map_vessel_type)

        return out_df.sort_values(by=["mmsi", "timestamp"]).reset_index(drop=True)

