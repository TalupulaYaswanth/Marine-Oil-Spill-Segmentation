"""
run_phase3.py
Main driver script for Phase 3: Vessel Detection and Attribution Engine.

Demonstrates:
  - Defining or vectorizing an oil spill polygon in EPSG:4326.
  - Ingesting filtered AIS tracks.
  - Computing metric UTM / geodesic distances (in kilometers) to the spill boundary.
  - Filtering candidates within a configurable radius (e.g., 15.0 km).
  - Scoring and ranking suspect polluter vessels based on spatial proximity,
    temporal offset, and physical causality.
  - Exporting GeoJSON and CSV attribution reports.
"""

import os
import sys
import argparse
from datetime import datetime, timezone

# Ensure UTF-8 output encoding for non-ASCII console environments
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from src.attribution.engine import VesselAttributionEngine

# Check if shapely / geopandas are present
try:
    from shapely.geometry import Polygon
    import geopandas as gpd
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False


def create_sample_spill_polygon():
    """
    Creates a realistic sample oil spill polygon centered in the Strait of Gibraltar
    at (-5.14, 36.22) matching the simulated slick footprint from Phase 1.
    """
    coords = [
        (-5.165, 36.215),
        (-5.150, 36.228),
        (-5.120, 36.235),
        (-5.110, 36.225),
        (-5.130, 36.212),
        (-5.165, 36.215)
    ]
    if HAS_GEOSPATIAL:
        return Polygon(coords)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Vessel Detection & Attribution Engine")
    parser.add_argument(
        "--ais-path",
        type=str,
        default="data/sample_ais.csv",
        help="Path to AIS CSV file containing vessel positions."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/phase3",
        help="Target directory for attribution reports and suspect GeoJSON."
    )
    parser.add_argument(
        "--threshold-km",
        type=float,
        default=25.0,
        help="Configurable suspect radius in kilometers (default: 25.0 km)."
    )
    parser.add_argument(
        "--spill-timestamp",
        type=str,
        default="2024-05-15T06:30:00Z",
        help="Satellite acquisition timestamp in ISO 8601 (default: 2024-05-15T06:30:00Z)."
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=0.6,
        help="Attribution weight for spatial distance (default: 0.6)."
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.4,
        help="Attribution weight for temporal alignment (default: 0.4)."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    spill_time = datetime.fromisoformat(args.spill_timestamp.replace("Z", "+00:00"))
    spill_poly = create_sample_spill_polygon()

    print(f"=== Starting Phase 3: Vessel Attribution Engine ===")
    print(f"Satellite Observation Time (UTC): {spill_time.isoformat()}")
    print(f"Distance Threshold Radius: {args.threshold_km} km")

    # Ingest AIS Data
    print(f"\n[1/3] Loading AIS Vessel Tracks from: {args.ais_path}")
    ais_df = pd.read_csv(args.ais_path)
    print(f"  -> Total AIS Pings: {len(ais_df)} across {ais_df['mmsi'].nunique()} unique vessels")

    # Initialize Attribution Engine
    print("\n[2/3] Computing Metric Track Distances and Spatiotemporal Attribution Scores...")
    engine = VesselAttributionEngine(
        distance_threshold_km=args.threshold_km,
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
        temporal_sigma_hours=2.0
    )

    ranked_suspects = engine.attribute_suspect_vessels(
        spill_polygon=spill_poly,
        ais_data=ais_df,
        spill_timestamp=spill_time
    )

    # Export Results
    print("\n[3/3] Exporting Attribution Artifacts...")
    csv_out = os.path.join(args.output_dir, "attribution_report.csv")
    
    if HAS_GEOSPATIAL and hasattr(ranked_suspects, "to_file"):
        geojson_out = os.path.join(args.output_dir, "suspect_vessels.geojson")
        ranked_suspects.to_file(geojson_out, driver="GeoJSON")
        ranked_suspects.drop(columns=["geometry"], errors="ignore").to_csv(csv_out, index=False)
        print(f"  -> Saved ranked suspect GeoJSON to: {geojson_out}")
    else:
        ranked_suspects.to_csv(csv_out, index=False)
    
    print(f"  -> Saved tabular attribution report to: {csv_out}")

    print("\n========================= SUSPECT VESSEL RANKINGS =========================")
    if ranked_suspects.empty:
        print("No vessels detected within the specified threshold radius.")
    else:
        for _, row in ranked_suspects.iterrows():
            print(
                f"Rank #{int(row['rank'])}: MMSI {int(row['mmsi'])} | "
                f"Attribution Score: {row['attribution_score']:.4f} | "
                f"Min Distance: {row['closest_distance_km']:.2f} km | "
                f"Time Offset: {row['time_delta_hours']:+.2f} hrs | "
                f"Speed: {row['closest_speed_knots']:.1f} kts | "
                f"Course: {row['closest_course_deg']:.0f}° | "
                f"Causal: {'YES' if row['is_causally_valid'] else 'NO'}"
            )
    print("===========================================================================\n")


if __name__ == "__main__":
    main()
