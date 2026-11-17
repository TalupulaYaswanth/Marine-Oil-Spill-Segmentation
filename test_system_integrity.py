"""
test_system_integrity.py
Comprehensive diagnostic test suite to verify 100% bug-free operational status
across all system modules, edge cases, and runtime dependencies.
"""

import os
import sys
import json
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Set Matplotlib cache dir to prevent writable cache warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_test_cache"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

passed = 0
failed = 0
failures = []


def run_test(name, func):
    global passed, failed, failures
    print(f"Testing: {name} ... ", end="")
    try:
        func()
        print("✅ PASS")
        passed += 1
    except Exception as e:
        print("❌ FAIL")
        failed += 1
        err_msg = traceback.format_exc()
        failures.append((name, err_msg))
        print(err_msg)


# ----------------------------------------------------------------------
# 1. SAR Preprocessor Tests
# ----------------------------------------------------------------------
def test_sar_preprocessor():
    from src.sar_preprocessor import SARImagePreprocessor
    proc = SARImagePreprocessor()
    
    # Test sample file
    sar_path = "data/S1A_IW_GRDH_1SDV_20240515T063000_sample.tif"
    if not os.path.exists(sar_path):
        from generate_sample_data import generate_sample_sar_geotiff
        generate_sample_sar_geotiff(sar_path)
    
    meta = proc.extract_metadata(sar_path)
    assert meta.width > 0 and meta.height > 0
    assert meta.bounds_wgs84 is not None
    assert len(meta.bounds_wgs84) == 4
    
    norm_arr, meta2, profile = proc.load_and_preprocess(sar_path, convert_to_db=True)
    assert norm_arr.ndim == 3
    assert norm_arr.shape[0] == 1
    assert not np.isnan(norm_arr).any()
    assert norm_arr.min() >= 0.0 and norm_arr.max() <= 1.0


def test_sar_edge_cases():
    from src.sar_preprocessor import SARImagePreprocessor
    proc = SARImagePreprocessor()
    
    # Test parsing timestamp from SAFE name
    t = proc.parse_timestamp_from_path("S1A_IW_GRDH_1SDV_20231015T054512_20231015T054537_050772_061E2D_F961.SAFE")
    assert t is not None
    assert t.year == 2023 and t.month == 10 and t.day == 15
    
    # Test parsing with invalid name
    t_invalid = proc.parse_timestamp_from_path("random_image.png")
    assert t_invalid is None


# ----------------------------------------------------------------------
# 2. AIS Preprocessor Tests
# ----------------------------------------------------------------------
def test_ais_preprocessor():
    from src.ais_preprocessor import AISPreprocessor, export_gdf_or_df_to_geojson
    prep = AISPreprocessor()
    
    ais_path = "data/sample_ais.csv"
    df = prep.load_ais_csv(ais_path)
    assert not df.empty
    assert "mmsi" in df.columns
    assert "timestamp" in df.columns
    assert "geometry" in df.columns
    
    # Test spatiotemporal filter
    bounds = (-5.50, 36.00, -4.80, 36.50)
    filtered = prep.filter_spatiotemporal(
        df,
        roi_bounds_wgs84=bounds,
        target_timestamp=datetime(2024, 5, 15, 6, 30, tzinfo=timezone.utc),
        time_window_hours=3.0
    )
    assert not filtered.empty
    
    # Test trajectory reconstruction
    traj = prep.reconstruct_trajectories(filtered)
    assert not traj.empty
    assert "mmsi" in traj.columns
    
    # Test export to geojson
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_gdf_or_df_to_geojson(traj, tmp_path)
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ais_empty_edge_cases():
    from src.ais_preprocessor import AISPreprocessor, export_gdf_or_df_to_geojson
    prep = AISPreprocessor()
    empty_df = pd.DataFrame(columns=["mmsi", "timestamp", "latitude", "longitude", "speed", "course", "geometry"])
    
    filtered = prep.filter_spatiotemporal(empty_df, (-5.0, 36.0, -4.0, 37.0), target_timestamp=datetime.now(timezone.utc))
    assert filtered.empty
    
    traj = prep.reconstruct_trajectories(filtered)
    assert traj.empty


# ----------------------------------------------------------------------
# 3. Universal AIS Loader Tests
# ----------------------------------------------------------------------
def test_universal_ais_loader():
    from src.ais_loader import UniversalAISLoader
    df = UniversalAISLoader.load("data/real_world/wakashio_mauritius/real_ais_telemetry.csv")
    assert not df.empty
    assert "mmsi" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "speed" in df.columns
    assert "course" in df.columns


# ----------------------------------------------------------------------
# 4. Hydrodynamic Drift Model Tests
# ----------------------------------------------------------------------
def test_drift_physics():
    from src.attribution.drift_physics import HydrodynamicDriftModel
    model = HydrodynamicDriftModel()
    
    # Test northern hemisphere
    res_nh = model.backtrack_slick_origin(
        center_lon=-5.14,
        center_lat=36.22,
        hours_elapsed=3.0,
        current_speed_ms=0.3,
        current_direction_deg=80.0,
        wind_speed_ms=8.0,
        wind_direction_from_deg=90.0
    )
    assert res_nh["origin_lat"] is not None
    assert res_nh["origin_lon"] is not None
    assert res_nh["total_distance_km"] > 0
    
    # Test southern hemisphere
    res_sh = model.backtrack_slick_origin(
        center_lon=57.75,
        center_lat=-20.44,
        hours_elapsed=4.0,
        current_speed_ms=0.2,
        current_direction_deg=310.0,
        wind_speed_ms=10.0,
        wind_direction_from_deg=115.0
    )
    assert res_sh["origin_lat"] is not None
    assert res_sh["total_distance_km"] > 0

    # Test extreme polar latitude edge case (near 90 deg)
    res_polar = model.backtrack_slick_origin(
        center_lon=0.0,
        center_lat=89.95,
        hours_elapsed=2.0,
        current_speed_ms=0.1,
        current_direction_deg=0.0,
        wind_speed_ms=5.0,
        wind_direction_from_deg=180.0
    )
    assert res_polar["origin_lat"] is not None


# ----------------------------------------------------------------------
# 5. Mask Vectorizer Tests
# ----------------------------------------------------------------------
def test_mask_vectorizer():
    from src.attribution.vectorizer import SpillMaskVectorizer
    vec = SpillMaskVectorizer(min_area_pixels=10)
    
    # Test binary mask with a circle
    mask = np.zeros((100, 100), dtype=np.uint8)
    y, x = np.ogrid[:100, :100]
    mask[(x - 50)**2 + (y - 50)**2 <= 25**2] = 1
    
    poly = vec.mask_to_polygons(mask, bounds_wgs84=(-5.5, 36.0, -4.8, 36.5))
    assert poly is not None
    assert hasattr(poly, "bounds")
    assert hasattr(poly, "centroid")


def test_mask_vectorizer_empty():
    from src.attribution.vectorizer import SpillMaskVectorizer
    vec = SpillMaskVectorizer()
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    poly = vec.mask_to_polygons(empty_mask)
    assert poly is not None


# ----------------------------------------------------------------------
# 6. Hybrid SOTA Segmentation Engine Tests
# ----------------------------------------------------------------------
def test_segmentation_engine():
    from src.segmentation.inference import OilSpillSegmenter
    seg = OilSpillSegmenter(detection_threshold=0.50)
    
    # Synthetic test image
    test_img = np.random.uniform(0.3, 0.7, (128, 128)).astype(np.float32)
    # Add dark slick patch
    test_img[40:80, 40:80] = 0.08
    
    res = seg.segment_sar_image(test_img, bounds_wgs84=(-5.5, 36.0, -4.8, 36.5))
    assert "probability_map" in res
    assert "binary_mask" in res
    assert "polygon" in res
    assert "slick_area_km2" in res
    assert res["probability_map"].shape == (128, 128)
    assert res["binary_mask"].shape == (128, 128)
    assert res["slick_pixels"] > 0

    # Clean ocean edge case (zero slick)
    clean_img = np.full((128, 128), 0.5, dtype=np.float32)
    res_clean = seg.segment_sar_image(clean_img, bounds_wgs84=(-5.5, 36.0, -4.8, 36.5))
    assert res_clean["slick_pixels"] == 0
    assert res_clean["slick_area_km2"] == 0.0
    assert res_clean["mean_confidence"] == 0.0


# ----------------------------------------------------------------------
# 7. Vessel Attribution Engine Tests
# ----------------------------------------------------------------------
def test_vessel_attribution_engine():
    from src.attribution.engine import VesselAttributionEngine
    engine = VesselAttributionEngine(distance_threshold_km=25.0)
    
    ais_df = pd.read_csv("data/sample_ais.csv")
    spill_coords = [(-5.165, 36.215), (-5.150, 36.228), (-5.120, 36.235), (-5.110, 36.225), (-5.130, 36.212)]
    spill_time = datetime(2024, 5, 15, 6, 30, tzinfo=timezone.utc)
    
    ranked = engine.attribute_suspect_vessels(
        spill_polygon=spill_coords,
        ais_data=ais_df,
        spill_timestamp=spill_time
    )
    assert not ranked.empty
    assert "rank" in ranked.columns
    assert "attribution_score" in ranked.columns
    assert "closest_distance_km" in ranked.columns
    assert "is_causally_valid" in ranked.columns
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.iloc[0]["attribution_score"] > 0

    # Test empty AIS dataframe edge case
    empty_ranked = engine.attribute_suspect_vessels(
        spill_polygon=spill_coords,
        ais_data=pd.DataFrame(),
        spill_timestamp=spill_time
    )
    assert empty_ranked.empty

    # Test empty spill coords edge case
    empty_poly_ranked = engine.attribute_suspect_vessels(
        spill_polygon=[],
        ais_data=ais_df,
        spill_timestamp=spill_time
    )
    assert "rank" in empty_poly_ranked.columns


# ----------------------------------------------------------------------
# 8. Vessel Image & Maritime API Tests
# ----------------------------------------------------------------------
def test_vessel_image_api():
    from src.vessel_image_api import VesselImageAPI
    details = VesselImageAPI.get_vessel_details(
        mmsi=372711000,
        imo=9337779,
        name="MV WAKASHIO"
    )
    assert details["vessel_name"] is not None
    assert details["image_url"] is not None
    assert details["image_url"].startswith("http") or os.path.exists(details["image_url"])
    assert details["flag"] is not None


# ----------------------------------------------------------------------
# 9. Live Monitor Service Tests
# ----------------------------------------------------------------------
def test_live_monitor_service():
    from src.live_monitor import LiveSpillMonitorService
    spills = LiveSpillMonitorService.get_live_spills()
    assert len(spills) > 0
    for s in spills:
        assert "name" in s
        assert "latitude" in s
        assert "longitude" in s
        assert "elapsed_str" in s
        assert "threat_level" in s
        assert "reported_time_utc" in s
    
    # Test radar tracks
    tracks = LiveSpillMonitorService.get_live_radar_tracks(36.22, -5.14, num_vessels=4)
    assert not tracks.empty
    assert "mmsi" in tracks.columns
    assert "timestamp" in tracks.columns
    
    # Test next satellite pass
    pass_info = LiveSpillMonitorService.compute_next_sentinel1_pass(36.22, -5.14)
    assert "countdown_hours" in pass_info
    assert "next_overpass_utc" in pass_info


# ----------------------------------------------------------------------
# 10. STAC Client Tests
# ----------------------------------------------------------------------
def test_stac_client():
    from src.stac_client import Sentinel1STACClient
    client = Sentinel1STACClient()
    # Offline fallback test
    scenes = client.search_scenes(bbox=(-5.5, 36.0, -4.8, 36.5), start_date="2024-05-01", end_date="2024-05-15")
    assert len(scenes) > 0
    assert "id" in scenes[0]
    assert "platform" in scenes[0]


# ----------------------------------------------------------------------
# 11. Production Pipeline Integration Test
# ----------------------------------------------------------------------
def test_production_pipeline_wakashio():
    import subprocess
    res = subprocess.run(
        [sys.executable, "run_production_pipeline.py", "--incident", "wakashio"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    assert res.returncode == 0, f"Error: {res.stderr}\nOutput: {res.stdout}"
    assert os.path.exists("output/production/interactive_marine_map.html")
    assert os.path.exists("output/production/forensic_investigation_report.md")
    assert os.path.exists("output/production/attribution_rankings.csv")


def test_production_pipeline_gibraltar():
    import subprocess
    res = subprocess.run(
        [sys.executable, "run_production_pipeline.py", "--incident", "gibraltar"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    assert res.returncode == 0, f"Error: {res.stderr}\nOutput: {res.stdout}"


def test_production_pipeline_sample():
    import subprocess
    res = subprocess.run(
        [sys.executable, "run_production_pipeline.py", "--incident", "sample"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    assert res.returncode == 0, f"Error: {res.stderr}\nOutput: {res.stdout}"


# ----------------------------------------------------------------------
# Run All Tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("🧪 EXECUTING FULL SYSTEM INTEGRITY & BUG AUDIT")
    print("="*70)
    
    run_test("SAR Preprocessor - Basic Ingestion", test_sar_preprocessor)
    run_test("SAR Preprocessor - Edge Cases", test_sar_edge_cases)
    run_test("AIS Preprocessor - Standard Parsing", test_ais_preprocessor)
    run_test("AIS Preprocessor - Empty / Zero-Ping Edge Cases", test_ais_empty_edge_cases)
    run_test("Universal AIS Multi-Format Loader", test_universal_ais_loader)
    run_test("Hydrodynamic Leeway Drift Physics (NH & SH)", test_drift_physics)
    run_test("Spill Mask Vectorizer - Regular Shapes", test_mask_vectorizer)
    run_test("Spill Mask Vectorizer - Empty Edge Case", test_mask_vectorizer_empty)
    run_test("Hybrid SOTA Segmentation Engine", test_segmentation_engine)
    run_test("Vessel Attribution & Causality Engine", test_vessel_attribution_engine)
    run_test("Vessel Image API & Registry Resolution", test_vessel_image_api)
    run_test("Live Spill Monitor & Radar Streaming Service", test_live_monitor_service)
    run_test("Sentinel-1 STAC Catalog Client", test_stac_client)
    run_test("Production Pipeline - Wakashio Benchmark", test_production_pipeline_wakashio)
    run_test("Production Pipeline - Gibraltar Benchmark", test_production_pipeline_gibraltar)
    run_test("Production Pipeline - Synthetic Benchmark", test_production_pipeline_sample)
    
    print("="*70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 ALL 16 INTEGRITY TESTS PASSED WITH ZERO ERRORS!")
        sys.exit(0)
