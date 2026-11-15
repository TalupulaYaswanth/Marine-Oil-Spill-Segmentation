# 🛰️ Marine Oil Spill Segmentation & Vessel Attribution Pipeline

**Project Title:** Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution Using Satellite Imagery and AIS Data  
**Standard Profile:** SkyTruth Cerulean & EMSA CleanSeaNet Operational Benchmark  
**Course/Project:** CSE471  

---

## 🌟 Executive Overview & Best Production Profile

This system implements the state-of-the-art (SOTA) operational architecture modeled after **SkyTruth Cerulean** (the global gold standard for satellite oil spill surveillance) and the **European Maritime Safety Agency (EMSA CleanSeaNet)**.

It delivers a fully functional, end-to-end pipeline connecting:
1. **Satellite Earth Observation (SAR):** Automated ingestion and decibel backscatter calibration ($\sigma^0$) of Sentinel-1 C-band Synthetic Aperture Radar imagery.
2. **Deep Learning & Adaptive CFAR Segmentation:** Dual-mode semantic segmentation engine (PyTorch U-Net with BCE-Dice compound loss + EMSA CleanSeaNet Constant False Alarm Rate dark-spot detector) that isolates capillary wave damping anomalies.
3. **Hydrodynamic Leeway Drift Backtracking:** Physics-based **NOAA ADIOS-2 / French CEDRE** surface leeway transport modeling with windage ($3.2\% U_{10}$), Coriolis deflection ($\pm 15^\circ$), and ambient tidal currents to backtrack slicks to the precise discharge origin point $T_0$.
4. **AIS Spatiotemporal Forensic Attribution:** Sub-meter metric UTM distance calculation, trajectory reconstruction, and multi-criteria Bayesian attribution scoring (CPA, temporal offset, heading alignment, ship type risk) with hard physical causality enforcement.
5. **Maritime Intelligence & Vessel Photo API:** High-resolution live vessel photo retrieval and vessel registration lookup (IMO, MMSI, Flag state, dimensions, deadweight tonnage) via MarineTraffic, VesselFinder, and Wikimedia Commons.
6. **Interactive Dashboard & Automated Reporting:** Streamlit web console and automated generation of forensic casualty dossiers (Markdown, CSV, JSON, GeoJSON, and Leaflet HTML).

```
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 1: Geospatial Remote Sensing Ingestion    │
                   │        • Sentinel-1 SAR Calibration & Decibel Scaling  │
                   │        • Universal Multi-Format AIS Ingestion (NOAA/DMA│
                   └───────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 2: Deep Learning Spill Segmentation       │
                   │        • Single-Channel SAR U-Net / EMSA CFAR Hybrid   │
                   │        • Compound BCE + Soft Dice Loss Formulation     │
                   │        • Jaccard Index (IoU) & F1-Score Evaluation     │
                   └───────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 3: Vessel Detection & Attribution Engine  │
                   │        • Local Metric UTM Projection (Sub-meter Acc.)  │
                   │        • NOAA ADIOS Hydrodynamic Leeway Drift Physics  │
                   │        • Spatiotemporal Attribution & Causality Scoring │
                   └───────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 4: SOTA Operational Console & Reports     │
                   │        • 1-Click Master Production Execution Engine    │
                   │        • Interactive Folium / Leaflet Marine Map       │
                   │        • Verified Vessel Photo Intelligence Dossier    │
                   │        • Open STAC Catalog Search (Planetary Computer) │
                   └────────────────────────────────────────────────────────┘
```

---

## ⚙️ Operational Best Profile Configuration (`configs/best_production_profile.yaml`)

| System Component | Production Specification | Operational Standard |
| :--- | :--- | :--- |
| **SAR Constellation** | Sentinel-1A / Sentinel-1B (C-Band, 5.405 GHz) | Copernicus Open Access Hub |
| **Acquisition Mode** | Interferometric Wide Swath (IW), GRD High Resolution | 10m pixel spacing, 250 km swath |
| **Polarization** | Co-Polarization VV (primary) + Cross-Polarization VH | Dual-Pol SAR |
| **Radiometric Scaling** | $\sigma^0 = 10 \log_{10}(I + \epsilon)$ (decibels) | Dynamic range: $[-30\text{ dB}, 0\text{ dB}]$ |
| **Speckle Filter** | Enhanced Lee Filter (5x5 kernel) | Percentile clipping: 1st–99th |
| **Segmentation Engine** | U-Net 5-Stage Convolutional Encoder-Decoder / EMSA CFAR | Balanced BCE + Soft Dice (50/50) |
| **Detection Threshold** | $P \ge 0.50$, Minimum slick area threshold: $0.15\text{ km}^2$ | Filters biogenic look-alikes |
| **Hydrodynamic Drift** | NOAA ADIOS-2 / CEDRE Leeway Drift Model | $100\%\text{ current} + 3.2\%\text{ wind} \pm 15^\circ\text{ Coriolis}$ |
| **Search Radius** | 25.0 km candidate buffer around slick boundary | Temporal window: $\pm 6.0\text{ hours}$ |
| **Attribution Weights** | Spatial CPA: 40%, Temporal: 30%, Heading: 15%, Ship Risk: 15% | Physical causality ($\Delta t < -30\text{m}$ penalized) |
| **Vessel Photo APIs** | MarineTraffic, VesselFinder, Wikimedia Commons, EQUASIS | Verified vessel photo & registry |

---

## 🌊 Benchmark Real-World Incidents Included

### 1. 🌊 The MV Wakashio Coral Reef Disaster (Mauritius, Aug 2020)
- **Copernicus EMS Rapid Mapping Activation:** `EMSR449`
- **Primary Casualty:** *MV WAKASHIO* (MMSI: `372711000`, IMO: `9337779`, Flag: Panama, 300m Capesize Bulk Carrier, 203,130 DWT).
- **Casualty Location:** Pointe d'Esny coral reef (`-20.4430°S, 57.7470°E`), discharging ~1,000 tonnes of heavy bunker fuel into a pristine marine park.
- **Satellite Observation:** Sentinel-1A IW GRD (`S1A_IW_GRDH_1SDV_20200810T015311_...`).
- **Telemetry Monitored:** MV Wakashio, Mauritian National Coast Guard (*CGS Barracuda*), salvage tugs (*Sir Gaetan Duval*), bunker tanker (*Tresta Star*), and transiting cargo vessels.
- **Environmental Forcing:** Southeast Trade Winds (115° at 18.5 knots), South Equatorial Current (0.32 m/s flowing Northwest directly into the lagoon).

### 2. 🚢 Strait of Gibraltar Marine Highway Corridor (Mediterranean Gateway)
- **Maritime Setting:** High-density Traffic Separation Scheme (TSS) linking the Atlantic Ocean and Mediterranean Sea.
- **Vessel Traffic:** Real IMO-registered Aframax Crude Oil Tankers (*Alexandra I*), Ultra Large Container Ships (*CMA CGM Montmartre*), and regional escort tugs (*VB Gibraltar*).
- **Satellite Observation:** Sentinel-1A IW GRD (`S1A_IW_GRDH_1SDV_20240515T063012_...`).

---

## 🚀 Execution Guide: Making the Project Work

### 1. Run Master Production Pipeline (One Command)
Execute the complete SkyTruth Cerulean end-to-end pipeline:
```bash
# Execute on MV Wakashio Coral Reef Disaster
python run_production_pipeline.py --incident wakashio

# Execute on Strait of Gibraltar Marine Highway Corridor
python run_production_pipeline.py --incident gibraltar

# Execute with custom candidate search radius (e.g. 30 km)
python run_production_pipeline.py --incident wakashio --threshold-km 30.0
```
**Artifacts Generated in `output/production/`:**
- `interactive_marine_map.html`: Standalone interactive Leaflet maritime situation map.
- `forensic_investigation_report.md`: Complete maritime pollution investigation report.
- `attribution_rankings.csv`: Ranked candidate polluter vessels with attribution scores.
- `spill_footprint.geojson`: Vectorized oil slick boundary polygon.
- `primary_suspect_dossier.json`: Comprehensive JSON telemetry and identity dossier.
- `calibrated_sar.png`: Radiometrically calibrated radar backscatter preview.
- `segmentation_overlay.png`: Dual-panel SAR scene vs slick probability heatmap.

---

### 2. Run Individual Architecture Phases

```bash
# Phase 1: SAR & AIS Geospatial Preprocessing
python run_phase1.py

# Phase 2: Segmentation Model Training & CFAR Validation
python train_segmentation.py

# Phase 3: Vessel Attribution & Causality Scoring
python run_phase3.py
```

---

### 3. Launch Interactive Web Dashboard

Launch the full-featured Streamlit console:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

**Dashboard Features:**
- **🚀 SOTA Production Pipeline Tab:** 1-click execution cockpit with side-by-side radar calibration and probability heatmaps, suspect vessel photo card, and 1-click artifact downloads.
- **🗺️ Maritime Situation Map:** Interactive Folium/Leaflet layer switching between Esri World Satellite Imagery, SAR footprints, oil spill polygons, and colored vessel tracks.
- **📋 Forensic Attribution Report:** Suspect rankings, CPA distance, time offset, and real ship photo resolution.
- **🌊 Hydrodynamic Drift Model:** Live NOAA ADIOS leeway vectors and reverse backtrack origin point.
- **📊 Vessel Telemetry Profiles:** Interactive Matplotlib speed (SOG) and heading (COG) time series.
- **🛰️ STAC Satellite Catalog Search:** Direct queries to Microsoft Planetary Computer and AWS Earth Search for Sentinel-1 scenes worldwide.

---

## 📂 Repository Structure

```
├── configs/
│   └── best_production_profile.yaml   # SkyTruth Cerulean & EMSA CleanSeaNet Profile
├── data/
│   ├── sample_ais.csv                 # Benchmark multi-vessel AIS telemetry
│   ├── S1A_IW_GRDH_...sample.tif      # Synthetic calibrated Sentinel-1 SAR scene
│   └── real_world/                    # Real Copernicus EMS incidents (Wakashio & Gibraltar)
├── output/
│   ├── phase1/                        # Preprocessed SAR rasters & filtered pings
│   ├── phase3/                        # Attribution rankings CSV
│   └── production/                    # Production dossier, reports, maps & heatmaps
├── src/
│   ├── sar_preprocessor.py            # Radiometric dB calibration & multi-backend loader
│   ├── ais_preprocessor.py            # AIS ingestion & spatiotemporal filtering
│   ├── pipeline.py                    # Phase 1 orchestration pipeline
│   ├── ais_loader.py                  # Universal NOAA/DMA/MarineTraffic loader
│   ├── stac_client.py                 # Open STAC client (Planetary Computer / AWS)
│   ├── live_monitor.py                # Live NOAA emergency spills & radar feed
│   ├── vessel_image_api.py            # MarineTraffic / VesselFinder photo resolver
│   ├── segmentation/
│   │   ├── inference.py               # Production Hybrid Segmenter (PyTorch + EMSA CFAR)
│   │   ├── unet.py                    # PyTorch Single-Channel SAR U-Net
│   │   ├── losses.py                  # BCEDiceLoss & IoU/F1 MetricTracker
│   │   ├── dataset.py                 # PyTorch SAR Dataset with augmentations
│   │   └── trainer.py                 # Checkpointing & validation loop
│   └── attribution/
│       ├── engine.py                  # Multi-criteria Bayesian attribution engine
│       ├── drift_physics.py           # NOAA ADIOS-2 leeway drift & backtracking
│       └── vectorizer.py              # Mask to Shapely Polygon georeferenced vectorizer
├── app.py                             # Full-featured Streamlit interactive console
├── run_production_pipeline.py         # Master production pipeline CLI
├── run_phase1.py                      # Phase 1 runner
├── train_segmentation.py              # Phase 2 runner
└── run_phase3.py                      # Phase 3 runner
```

---

## 📜 License & Compliance
This software adheres to open geospatial data standards (OGC GeoTIFF, GeoJSON, EPSG:4326) and maritime safety conventions (IMO SOLAS Regulation V/19).
