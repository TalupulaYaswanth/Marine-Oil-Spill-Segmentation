"""
Module: pipeline.py
Description: End-to-end Phase 1 orchestration pipeline integrating SAR preprocessing
             and AIS spatiotemporal trajectory filtering.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import geopandas as gpd
import numpy as np

from .sar_preprocessor import SARImagePreprocessor, SARMetadata
from .ais_preprocessor import AISPreprocessor


class Phase1PreprocessingPipeline:
    """
    Coordinates Sentinel-1 SAR ingestion/normalization with historical AIS trajectory
    filtering against the satellite scene's spatial footprint and acquisition timestamp.
    """

    def __init__(
        self,
        time_window_hours: float = 3.0,
        spatial_buffer_deg: float = 0.05,
        convert_to_db: bool = True,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    ):
        self.time_window_hours = time_window_hours
        self.spatial_buffer_deg = spatial_buffer_deg
        self.convert_to_db = convert_to_db
        self.clip_percentiles = clip_percentiles

        self.sar_processor = SARImagePreprocessor()
        self.ais_processor = AISPreprocessor()

    def run(
        self,
        sar_path: str,
        ais_path: str,
        output_dir: str,
        fallback_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Executes the Phase 1 preprocessing pipeline.

        Args:
            sar_path: Path to raw Sentinel-1 GeoTIFF.
            ais_path: Path to historical AIS CSV.
            output_dir: Target directory for preprocessed GeoTIFF and filtered AIS GeoJSONs.
            fallback_timestamp: Optional timestamp if GeoTIFF lacks timestamp metadata.

        Returns:
            Dictionary containing pipeline execution metrics, metadata, and output artifact paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"=== Starting Phase 1 Preprocessing Pipeline ===")
        print(f"Input SAR: {sar_path}")
        print(f"Input AIS: {ais_path}")
        print(f"Output Directory: {output_dir}")

        # -------------------------------------------------------------
        # Step 1: Sentinel-1 SAR Ingestion & Preprocessing
        # -------------------------------------------------------------
        print("\n[1/4] Loading and Preprocessing Sentinel-1 SAR Image...")
        norm_data, metadata, profile = self.sar_processor.load_and_preprocess(
            tif_path=sar_path,
            convert_to_db=self.convert_to_db,
            clip_percentiles=self.clip_percentiles
        )

        print(f"  -> Dimensions: {metadata.width} x {metadata.height} (Bands: {metadata.num_bands})")
        print(f"  -> Native CRS: {metadata.crs} (EPSG: {metadata.epsg})")
        print(f"  -> WGS84 Bounds: {metadata.bounds_wgs84}")
        print(f"  -> Acquisition Timestamp (UTC): {metadata.timestamp.isoformat()}")

        # Save preprocessed normalized raster
        preprocessed_tif_path = os.path.join(output_dir, "preprocessed_sar.tif")
        self.sar_processor.save_preprocessed(preprocessed_tif_path, norm_data, profile)
        print(f"  -> Saved preprocessed SAR raster to: {preprocessed_tif_path}")

        # -------------------------------------------------------------
        # Step 2: Load Historical AIS Data
        # -------------------------------------------------------------
        print("\n[2/4] Parsing and Validating Historical AIS CSV Data...")
        raw_ais_gdf = self.ais_processor.load_ais_csv(ais_path)
        print(f"  -> Total AIS Pings Loaded: {len(raw_ais_gdf)}")
        print(f"  -> Unique Vessels (MMSIs): {raw_ais_gdf['mmsi'].nunique()}")

        # -------------------------------------------------------------
        # Step 3: Spatiotemporal Filtering
        # -------------------------------------------------------------
        print(f"\n[3/4] Spatiotemporal Filtering (Window: +/-{self.time_window_hours} hrs, Buffer: {self.spatial_buffer_deg} deg)...")
        filtered_ais_gdf = self.ais_processor.filter_spatiotemporal(
            ais_gdf=raw_ais_gdf,
            roi_bounds_wgs84=metadata.bounds_wgs84,
            roi_polygon=metadata.footprint_wgs84,
            target_timestamp=metadata.timestamp,
            time_window_hours=self.time_window_hours,
            spatial_buffer_deg=self.spatial_buffer_deg
        )

        print(f"  -> Candidate Pings within Spatiotemporal Window: {len(filtered_ais_gdf)}")
        print(f"  -> Candidate Vessels in Scene: {filtered_ais_gdf['mmsi'].nunique() if not filtered_ais_gdf.empty else 0}")

        # -------------------------------------------------------------
        # Step 4: Reconstruct Vessel Trajectories & Save Artifacts
        # -------------------------------------------------------------
        print("\n[4/4] Reconstructing Vessel Trajectories and Exporting Artifacts...")
        trajectories_gdf = self.ais_processor.reconstruct_trajectories(filtered_ais_gdf)

        # Export filtered AIS points
        filtered_pings_path = os.path.join(output_dir, "filtered_ais_pings.geojson")
        filtered_pings_csv_path = os.path.join(output_dir, "filtered_ais_pings.csv")
        filtered_ais_gdf.to_file(filtered_pings_path, driver="GeoJSON")
        # Also export CSV for convenient inspection
        filtered_ais_gdf.drop(columns=["geometry"], errors="ignore").to_csv(filtered_pings_csv_path, index=False)

        # Export trajectories
        trajectories_path = os.path.join(output_dir, "vessel_trajectories.geojson")
        trajectories_gdf.to_file(trajectories_path, driver="GeoJSON")

        print(f"  -> Saved filtered AIS pings to: {filtered_pings_path}")
        print(f"  -> Saved vessel trajectories to: {trajectories_path}")
        print("=== Phase 1 Preprocessing Complete! ===\n")

        return {
            "metadata": metadata,
            "preprocessed_sar_path": preprocessed_tif_path,
            "filtered_ais_pings_path": filtered_pings_path,
            "filtered_ais_csv_path": filtered_pings_csv_path,
            "trajectories_path": trajectories_path,
            "num_raw_pings": len(raw_ais_gdf),
            "num_filtered_pings": len(filtered_ais_gdf),
            "num_candidate_vessels": len(trajectories_gdf),
            "candidate_mmsis": trajectories_gdf["mmsi"].tolist() if not trajectories_gdf.empty else []
        }

