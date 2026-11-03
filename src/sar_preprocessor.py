"""
Module: sar_preprocessor.py
Description: Ingestion, metadata extraction, and radiometric preprocessing for 
             Sentinel-1 Synthetic Aperture Radar (SAR) imagery.
"""

import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box, Polygon


@dataclass
class SARMetadata:
    """Metadata container for Sentinel-1 SAR imagery."""
    crs: str
    epsg: Optional[int]
    bounds_native: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in native CRS
    bounds_wgs84: Tuple[float, float, float, float]   # (min_lon, min_lat, max_lon, max_lat)
    footprint_wgs84: Polygon                          # Shapely geometry in EPSG:4326
    timestamp: datetime                               # Image acquisition timestamp (UTC)
    width: int
    height: int
    num_bands: int
    nodata_value: Optional[float]
    resolution: Tuple[float, float]                   # (pixel_width, pixel_height)
    tags: Dict[str, Any]


class SARImagePreprocessor:
    """
    Handles Sentinel-1 SAR GeoTIFF ingestion, radiometric conversion (dB scale),
    NoData masking, outlier clipping, and geographic footprint metadata extraction.
    """

    def __init__(self, epsilon: float = 1e-7):
        """
        Args:
            epsilon: Small offset to avoid log(0) during dB conversion.
        """
        self.epsilon = epsilon

    def parse_timestamp_from_path(self, filepath: str) -> Optional[datetime]:
        """
        Attempts to parse acquisition timestamp from standard Sentinel-1 SAFE filename naming convention.
        Standard format: S1[A|B]_[Mode]_[Type]___[Polarization]_[StartDateTime]_[StopDateTime]...
        Example: S1A_IW_GRDH_1SDV_20231015T054512_...
        """
        filename = os.path.basename(filepath)
        # Search for pattern: YYYYMMDDTHHMMSS
        matches = re.findall(r"(\d{8}T\d{6})", filename)
        if matches:
            try:
                # Use the start acquisition time (first match)
                return datetime.strptime(matches[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return None

    def extract_metadata(
        self, 
        tif_path: str, 
        fallback_timestamp: Optional[datetime] = None
    ) -> SARMetadata:
        """
        Extracts native and reprojected bounding boxes, CRS, acquisition time, and dimensions.

        Args:
            tif_path: Path to the Sentinel-1 GeoTIFF image.
            fallback_timestamp: Optional timestamp to use if none is found in metadata or filename.

        Returns:
            SARMetadata dataclass.
        """
        with rasterio.open(tif_path) as src:
            native_crs = src.crs
            epsg = native_crs.to_epsg() if native_crs else None
            bounds_native = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

            # Transform native bounding box to WGS84 (EPSG:4326) for AIS coordinate alignment
            if native_crs and native_crs != CRS.from_epsg(4326):
                wgs84_bounds = transform_bounds(
                    native_crs, 
                    CRS.from_epsg(4326), 
                    src.bounds.left, 
                    src.bounds.bottom, 
                    src.bounds.right, 
                    src.bounds.top
                )
            else:
                wgs84_bounds = bounds_native

            footprint_polygon = box(*wgs84_bounds)

            # Extract acquisition timestamp
            timestamp = None
            tags = src.tags()
            
            # Check common GeoTIFF metadata tags
            for key in ["TIFFTAG_DATETIME", "ACQUISITION_DATETIME", "acquisition_time", "system:time_start"]:
                if key in tags:
                    val = tags[key]
                    try:
                        # Try parsing ISO 8601 or millisecond timestamp
                        if str(val).isdigit():
                            timestamp = datetime.fromtimestamp(int(val) / 1000.0, tz=timezone.utc)
                        else:
                            timestamp = datetime.fromisoformat(str(val)).replace(tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass

            # Fallback to filename parsing
            if timestamp is None:
                timestamp = self.parse_timestamp_from_path(tif_path)

            # Fallback to caller-specified or current time
            if timestamp is None:
                if fallback_timestamp is not None:
                    timestamp = fallback_timestamp
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

            res_x, res_y = src.res

            metadata = SARMetadata(
                crs=native_crs.to_string() if native_crs else "Unknown",
                epsg=epsg,
                bounds_native=bounds_native,
                bounds_wgs84=wgs84_bounds,
                footprint_wgs84=footprint_polygon,
                timestamp=timestamp,
                width=src.width,
                height=src.height,
                num_bands=src.count,
                nodata_value=src.nodata,
                resolution=(abs(res_x), abs(res_y)),
                tags=tags
            )

            return metadata

    def load_and_preprocess(
        self,
        tif_path: str,
        convert_to_db: bool = True,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        band_indices: Optional[Tuple[int, ...]] = None,
        fill_nodata_val: float = 0.0
    ) -> Tuple[np.ndarray, SARMetadata, Dict[str, Any]]:
        """
        Loads Sentinel-1 SAR bands, handles NoData/ocean boundary masks, 
        converts linear intensity to decibels (dB), and normalizes values to [0, 1].

        Args:
            tif_path: Path to the GeoTIFF file.
            convert_to_db: Whether to convert linear radar amplitude/intensity to decibel (dB) scale.
            clip_percentiles: Percentiles (min, max) for robust outlier suppression against speckle noise.
            band_indices: 1-based indices of bands to read. Defaults to all bands.
            fill_nodata_val: Value assigned to masked/NoData pixels post-normalization.

        Returns:
            Tuple of:
                - preprocessed_data: np.ndarray of shape (C, H, W) normalized to [0, 1].
                - metadata: SARMetadata.
                - profile: Rasterio profile for subsequent georeferenced output writing.
        """
        metadata = self.extract_metadata(tif_path)

        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            indexes = band_indices if band_indices is not None else tuple(range(1, src.count + 1))
            
            # Read bands as float32
            raw_data = src.read(indexes, masked=True).astype(np.float32)
            # raw_data shape: (C, H, W)

        num_bands, height, width = raw_data.shape
        normalized_bands = np.zeros((num_bands, height, width), dtype=np.float32)

        for b in range(num_bands):
            band = raw_data[b]

            # 1. Mask NoData, zero-fill, and NaNs
            valid_mask = ~band.mask if hasattr(band, "mask") else np.ones((height, width), dtype=bool)
            if metadata.nodata_value is not None:
                valid_mask = valid_mask & (band.data != metadata.nodata_value)
            valid_mask = valid_mask & (~np.isnan(band.data)) & (~np.isinf(band.data)) & (band.data > 0)

            band_data = band.data.copy()

            # 2. Radiometric Decibel Conversion (Intensity -> dB: 10 * log10(I))
            if convert_to_db:
                # Oil spills dampen capillary waves, lowering backscatter into deep negative dB
                band_data[valid_mask] = 10.0 * np.log10(band_data[valid_mask] + self.epsilon)

            # 3. Percentile Clipping (removes extreme speckle noise and high-backscatter metal hulls)
            if np.any(valid_mask):
                valid_pixels = band_data[valid_mask]
                p_min, p_max = np.percentile(valid_pixels, clip_percentiles)
                
                # Clip values to percentile boundaries
                clipped = np.clip(band_data, p_min, p_max)

                # 4. Min-Max Normalization to [0, 1] for neural network ingestion
                denom = (p_max - p_min) if (p_max - p_min) > 1e-5 else 1.0
                norm = (clipped - p_min) / denom
                norm[~valid_mask] = fill_nodata_val
                normalized_bands[b] = norm
            else:
                normalized_bands[b] = fill_nodata_val

        # Update profile for output
        profile.update(
            dtype=rasterio.float32,
            count=num_bands,
            nodata=fill_nodata_val
        )

        return normalized_bands, metadata, profile

    def save_preprocessed(
        self,
        output_path: str,
        data: np.ndarray,
        profile: Dict[str, Any]
    ) -> None:
        """Saves preprocessed array to a new georeferenced GeoTIFF."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32))

