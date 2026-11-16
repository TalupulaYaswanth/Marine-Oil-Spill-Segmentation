# NaviSpill AI - Dataset Manifest & Provenance

This document tracks all datasets, satellite imagery sources, AIS records, benchmark samples, licenses, and preprocessing operations used in the NaviSpill AI project.

---

## 1. Candidate SAR Oil Spill Datasets

### A. M4D Marine Oil Spill Dataset (Sentinel-1)
- **Source**: Marine Disaster Database (M4D) / National and Kapodistrian University of Athens.
- **Sensor**: Sentinel-1 Synthetic Aperture Radar (C-band SAR, IW mode, VV/VH polarizations).
- **Format**: 1-channel grayscale GeoTIFFs / PNG tiles with paired semantic segmentation masks.
- **Classes**:
  - `0`: Sea Surface (Background)
  - `1`: Oil Spill
  - `2`: Look-alike (e.g., biogenic slicks, low-wind areas, internal waves)
  - `3`: Ship
  - `4`: Land
- **Licensing**: Open academic research / Creative Commons.
- **Status**: Checked for availability.

### B. Hugging Face / Zenodo Sentinel-1 SAR Oil Spill Benchmarks
- **Repository**: `chri/oil-spill-dataset` / Zenodo records.
- **Format**: Paired image and mask GeoTIFF/PNGs.
- **License**: CC BY 4.0.

### C. Curated Benchmark & Realistic Synthetic Verification Set
- **Role**: Deterministic regression and pipeline verification set.
- **Format**: Single-channel GeoTIFFs (16-bit float / 8-bit normalized backscatter) with georeferencing (WGS84 EPSG:4326 affine transform) and paired ground truth binary masks.
- **Label**: `SYNTHETIC_BENCHMARK` / `SAMPLE_S1_SAR`.
- **Purpose**: Ensures 100% reproducible testing of the full end-to-end pipeline (calibration, inference, vectorization, AIS spatial joining) regardless of external download limits or server downtime.

---

## 2. AIS (Automatic Identification System) Dataset

### A. Real AIS Historical Trajectory Sample
- **Source**: US Coast Guard (MarineCadastre.gov) / MarineTraffic open samples.
- **Schema**:
  - `MMSI`: Maritime Mobile Service Identity (unique 9-digit vessel identifier)
  - `BaseDateTime`: UTC timestamp (`YYYY-MM-DDTHH:MM:SS`)
  - `LAT`: Latitude in decimal degrees (WGS84)
  - `LON`: Longitude in decimal degrees (WGS84)
  - `SOG`: Speed Over Ground in knots
  - `COG`: Course Over Ground in degrees (0–360)
  - `Heading`: Vessel true heading
  - `VesselName`: Name of vessel
  - `VesselType`: Cargo, Tanker, Fishing, Passenger, Tug, Other
  - `Draft`: Vessel draft in meters
- **Licensing**: Public Domain (US Government work).

### B. Controlled Benchmark AIS Dataset
- **Role**: Synchronized spatial-temporal test scenario matching the SAR scene footprint.
- **Vessels included**:
  - Candidate 1 (Spill Proximity): Tanker crossing the oil slick within ±2 hours of satellite overpass.
  - Candidate 2 (Passing Vessel): Cargo ship passing 8 km away at normal cruising speed.
  - Candidate 3 (Stationary Vessel): Anchor/fishing vessel outside the spill corridor.
- **Label**: Clearly marked as `SAMPLE_AIS_BENCHMARK`.

---

## 3. Data Storage Structure
```
datasets/
├── raw/
│   ├── sar/
│   └── ais/
├── processed/
│   ├── sar_tiles/
│   ├── masks/
│   └── ais_cleaned/
└── sample/
    ├── sample_s1_scene.tif
    ├── sample_s1_mask.png
    └── sample_ais_trajectories.csv
```
