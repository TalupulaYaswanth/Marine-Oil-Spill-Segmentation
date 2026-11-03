"""
run_phase1.py
Main driver script for executing Phase 1: Geospatial Data Preprocessing.

Loads a Sentinel-1 SAR GeoTIFF image, applies radiometric preprocessing,
extracts spatial/temporal metadata, parses historical AIS records,
and filters vessel trajectories matching the satellite footprint and acquisition time.
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure UTF-8 output encoding for Windows command line environments
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sar_preprocessor import SARImagePreprocessor
from src.ais_preprocessor import AISPreprocessor
from src.pipeline import Phase1PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Sentinel-1 SAR & AIS Geospatial Preprocessing Pipeline"
    )
    parser.add_argument(
        "--sar-path",
        type=str,
        default="data/S1A_IW_GRDH_1SDV_20240515T063000_sample.tif",
        help="Path to Sentinel-1 SAR GeoTIFF image."
    )
    parser.add_argument(
        "--ais-path",
        type=str,
        default="data/sample_ais.csv",
        help="Path to historical AIS CSV data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/phase1",
        help="Directory where preprocessed rasters and filtered AIS GeoJSON files will be stored."
    )
    parser.add_argument(
        "--time-window-hours",
        type=float,
        default=3.0,
        help="Symmetric temporal filter window [T - w, T + w] in hours around acquisition time (default: 3.0)."
    )
    parser.add_argument(
        "--spatial-buffer-deg",
        type=float,
        default=0.05,
        help="Spatial buffer in degrees around SAR bounding box (default: 0.05 deg, ~5.5 km)."
    )
    parser.add_argument(
        "--convert-to-db",
        action="store_true",
        default=True,
        help="Convert linear intensity to decibels (dB)."
    )

    args = parser.parse_args()

    # Initialize and execute pipeline
    pipeline = Phase1PreprocessingPipeline(
        time_window_hours=args.time_window_hours,
        spatial_buffer_deg=args.spatial_buffer_deg,
        convert_to_db=args.convert_to_db,
        clip_percentiles=(1.0, 99.0)
    )

    results = pipeline.run(
        sar_path=args.sar_path,
        ais_path=args.ais_path,
        output_dir=args.output_dir
    )

    print("\n--- Phase 1 Execution Summary ---")
    meta = results["metadata"]
    print(f"Image Resolution: {meta.width} x {meta.height} (Pixel Size: {meta.resolution})")
    print(f"Native CRS: {meta.crs}")
    print(f"EPSG Code: {meta.epsg}")
    print(f"Bounding Box (WGS84 [min_lon, min_lat, max_lon, max_lat]): {meta.bounds_wgs84}")
    print(f"Acquisition Timestamp: {meta.timestamp.isoformat()}")
    print(f"Total AIS Pings Evaluated: {results['num_raw_pings']}")
    print(f"Candidate Pings within Scene Window: {results['num_filtered_pings']}")
    print(f"Candidate Vessels (MMSI count): {results['num_candidate_vessels']}")
    print(f"Candidate MMSIs: {results['candidate_mmsis']}")
    print(f"Preprocessed GeoTIFF: {results['preprocessed_sar_path']}")
    print(f"Filtered Pings GeoJSON: {results['filtered_ais_pings_path']}")
    print(f"Reconstructed Trajectories: {results['trajectories_path']}")


if __name__ == "__main__":
    main()

