# Marine Oil Spill Segmentation & Vessel Attribution Pipeline

**Project Title:** Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution Using Satellite Imagery and AIS Data  
**Course/Project:** CSE471

---

## Architecture Overview

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
                   │        • Single-Channel SAR U-Net Architecture         │
                   │        • Compound BCE + Soft Dice Loss Formulation     │
                   │        • Jaccard Index (IoU) & F1-Score Evaluation     │
                   └───────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 3: Vessel Detection & Attribution Engine  │
                   │        • Local Metric UTM Projection (Sub-meter Accuracy│
                   │        • NOAA ADIOS Hydrodynamic Leeway Drift Physics  │
                   │        • Spatiotemporal Attribution & Causality Scoring │
                   └───────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                   ┌────────────────────────────────────────────────────────┐
                   │        Phase 4: Real-World Operational Dashboard       │
                   │        • Copernicus EMS EMSR449 Real Incident Benchmark│
                   │        • Open STAC Catalog Search (Planetary Computer) │
                   │        • Interactive Marine Navigation Map (Folium)    │
                   └────────────────────────────────────────────────────────┘
```

---

## Real-World Case Studies Included

### 1. 🌊 The MV Wakashio Coral Reef Disaster (Mauritius, Aug 2020)
- **Copernicus EMS Rapid Mapping Activation:** `EMSR449`
- **Primary Suspect:** *MV WAKASHIO* (MMSI: `372711000`, IMO: `9337779`, Flag: Panama, 300m Capesize Bulk Carrier, 203,130 DWT)
- **Grounded Location:** Pointe d'Esny coral reef (`-20.4430°S, 57.7470°E`), spilling ~1,000 tonnes of Very Low Sulphur Fuel Oil (VLSFO) into the marine reserve.
- **Satellite Scene:** Sentinel-1A IW GRD (`S1A_IW_GRDH_1SDV_20200810T015311_...`)
- **Telemetry Monitored:** MV Wakashio, Mauritian National Coast Guard (*CGS Barracuda*), salvage tugs (*Sir Gaetan Duval*), bunker tanker (*Tresta Star*), and passing container liners (*MSC Shannon*).
- **Physical Environmental Forcing:** Southeast Trade Winds (115° at 18.5 knots), South Equatorial Current (0.32 m/s flowing Northwest directly into the lagoon).

### 2. 🚢 Strait of Gibraltar Marine Highway Corridor
- **Maritime Setting:** High-density Traffic Separation Scheme (TSS) linking the Atlantic Ocean and Mediterranean Sea.
- **Vessel Traffic:** Real IMO-registered Crude Oil Tankers (*Alexandra I*), Ultra Large Container Ships (*CMA CGM Montmartre*), and regional pilot tugs (*VB Gibraltar*).
- **Satellite Scene:** Sentinel-1A IW GRD (`S1A_IW_GRDH_1SDV_20240515T063012_...`)

---

## Key Modules & Capabilities

1. **Universal Real-World AIS Loader ([`src/ais_loader.py`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/src/ais_loader.py)):**
   - Ingests and auto-standardizes NOAA Marine Cadastre, Danish Maritime Authority (DMA), Spire, and MarineTraffic CSV files.
   - Automatically maps `MMSI`, `BaseDateTime`, `SOG` (Speed), `COG` (Course), `VesselName`, `IMO`, and `ShipType`.

2. **Hydrodynamic Leeway Drift Physics ([`src/attribution/drift_physics.py`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/src/attribution/drift_physics.py)):**
   - Implements the **NOAA ADIOS / French CEDRE** maritime leeway formulation:
     $$\vec{U}_{\text{drift}} = \vec{U}_{\text{current}} + \alpha \cdot \vec{U}_{\text{wind}} + \vec{U}_{\text{stokes}}$$
     with windage leeway $\alpha = 3.2\%$ and Coriolis deflection angle ($\pm 15^\circ$ to the right in NH, left in SH).
   - Backtracks the satellite-observed slick into the past to pinpoint the precise discharge origin coordinates!

3. **Open STAC Satellite Catalog Client ([`src/stac_client.py`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/src/stac_client.py)):**
   - Connects directly to **Microsoft Planetary Computer** and **Earth Search (AWS)** STAC APIs.
   - Allows users to search Sentinel-1 SAR GRD scenes for any bounding box and date range worldwide.

4. **Interactive Dashboard ([`app.py`](file:///c:/Users/V/OneDrive/ドキュメント/Rockstar%20Games/CSE471/app.py)):**
   - Full-featured Streamlit application with interactive Folium marine maps, real vessel names, speed/heading telemetry profiles, and downloadable forensic attribution reports.

---

## Quickstart

### 1. Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate oilspill-env

# Or using pip
pip install -r requirements.txt
```

### 2. Populate Real-World Datasets
```bash
python src/setup_real_world_data.py
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.
