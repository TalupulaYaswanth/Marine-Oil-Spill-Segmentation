# Marine Oil Spill Segmentation & Vessel Attribution Pipeline

**Project Title:** Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution Using Satellite Imagery and AIS Data  
**Course/Project:** CSE471

---

## Architecture Overview

```
                   ┌──────────────────────────────────────────────┐
                   │    Phase 1: Geospatial Data Preprocessing    │
                   │    • SAR Radiometric Calibration & dB Scaling│
                   │    • AIS Spatiotemporal Bounding Box Match   │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    Phase 2: Deep Learning Spill Segmentation │
                   │    • Custom SAR Dataset & Augmentations      │
                   │    • 1-Channel U-Net Architecture            │
                   │    • Compound BCE + Soft Dice Loss           │
                   │    • Metric Tracking (IoU, F1, Prec, Rec)    │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    Phase 3: Vessel Attribution Engine        │
                   │    • Spill Polygon Metric Distance Engine    │
                   │    • Automatic Local UTM Zone Reprojection   │
                   │    • Spatiotemporal Attribution Scoring      │
                   │    • Suspect Ranking & GeoJSON/CSV Reporting │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    Phase 4: Interactive Dashboard (Streamlit)│
                   │    • Interactive Folium/Leaflet Marine Map   │
                   │    • Overlay SAR Bounds, Spill Mask & Tracks │
                   │    • Real-Time Slider Attribution Filtering  │
                   │    • Forensic Suspect Reports & Analytics    │
                   └──────────────────────────────────────────────┘
```

---

## Project Structure

```
CSE471/
├── requirements.txt            # Dependency specifications
├── app.py                      # Phase 4: Streamlit interactive visualization dashboard
├── run_phase1.py               # Phase 1: SAR & AIS preprocessing runner
├── generate_sample_data.py     # Generates synthetic SAR GeoTIFF & sample AIS CSV
├── train_segmentation.py       # Phase 2: U-Net model training & validation runner
├── generate_sample_patches.py  # Generates 256x256 SAR patches & binary masks
├── run_phase3.py               # Phase 3: Vessel detection & attribution engine runner
├── README.md                   # Complete system documentation
├── data/
│   ├── sample_ais.csv          # Sample AIS tracking data
│   └── patches/                # Generated synthetic SAR training/val patches
├── output/                     # Generated GeoJSONs, CSV reports, and GeoTIFFs
└── src/
    ├── __init__.py
    ├── sar_preprocessor.py     # SAR image loading, NoData masking, dB scaling
    ├── ais_preprocessor.py     # AIS parsing, GeoPandas conversion, spatiotemporal filter
    ├── pipeline.py             # Phase 1 end-to-end orchestration
    ├── segmentation/
    │   ├── dataset.py          # Custom PyTorch Dataset with radar augmentations
    │   ├── unet.py             # 1-channel U-Net architecture
    │   ├── losses.py           # Combined BCE + Dice Loss & MetricTracker
    │   └── trainer.py          # Training and validation loop
    └── attribution/
        ├── vectorizer.py       # Raster mask to Shapely Polygon vectorizer
        └── engine.py           # Spatiotemporal attribution scoring engine
```

---

## Phase 1: Geospatial Data Preprocessing

```bash
# 1. Generate sample AIS and SAR files
python generate_sample_data.py

# 2. Run Preprocessing Pipeline
python run_phase1.py \
  --sar-path data/S1A_IW_GRDH_1SDV_20240515T063000_sample.tif \
  --ais-path data/sample_ais.csv \
  --output-dir output/phase1 \
  --time-window-hours 3.0 \
  --spatial-buffer-deg 0.05
```

---

## Phase 2: Oil Spill Semantic Segmentation

```bash
# 1. Generate synthetic SAR patches and binary masks
python generate_sample_patches.py

# 2. Train U-Net model
python train_segmentation.py \
  --train-images data/patches/train/images \
  --train-masks data/patches/train/masks \
  --val-images data/patches/val/images \
  --val-masks data/patches/val/masks \
  --epochs 10 \
  --batch-size 4 \
  --lr 0.001
```

---

## Phase 3: Vessel Detection & Attribution Engine

```bash
python run_phase3.py \
  --ais-path data/sample_ais.csv \
  --output-dir output/phase3 \
  --threshold-km 25.0 \
  --spill-timestamp 2024-05-15T06:30:00Z
```

---

## Phase 4: Interactive Visualization Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

### Dashboard Features
- **Interactive Maritime Map:** Pan, zoom, and toggle base layers (OpenStreetMap, Esri Satellite Imagery).
- **Overlays:**
  - Amber dashed bounding box representing the Sentinel-1 SAR acquisition footprint.
  - Deep crimson polygon representing the deep-learning-segmented marine oil slick.
  - Full AIS trajectory paths and individual pings color-coded by suspect ranking (Red = Rank 1, Orange = Rank 2, Gray = Ambient).
- **Interactive Telemetry Popups:** Click on any ping to view MMSI, timestamp, speed, heading, and distance to the spill.
- **Dynamic Investigation Controls:** Real-time sliders in the sidebar for candidate search radius (1–50 km) and spatial/temporal weights.
- **Forensic Attribution Report:** Tabular breakdown of suspect rankings, confidence scores, and physical causality validation, complete with CSV export and vessel speed analytics.
