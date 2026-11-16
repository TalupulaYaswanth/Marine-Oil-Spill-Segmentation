# NaviSpill AI: Marine Oil Spill Segmentation and Vessel Attribution

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A student-level deep learning and geospatial intelligence system designed for automated marine oil spill detection from **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery and multi-factor **Automatic Identification System (AIS)** candidate vessel attribution. Developed for the **CSE471 Capstone Project**.

---

## 🌟 Key Features

1. **Radiometric SAR Preprocessing (`src/preprocessing/sar_preprocessor.py`)**:
   - Supports GeoTIFFs and standard raster formats (PNG, JPG).
   - Rejects radar speckle via 1st-to-99th percentile contrast clipping.
   - Robust NoData handling and min-max normalization to $[0.0, 1.0]$.
   - Preserves affine geotransforms and Coordinate Reference Systems (CRS).
   - Sliding-window tiling for large satellite footprints.

2. **Pretrained U-Net Deep Learning Segmentation (`src/models/`, `src/training/`)**:
   - Pretrained **ResNet-34** encoder adapted for single-channel grayscale SAR backscatter.
   - Initialized with ImageNet weights via channel-averaged conv1 kernels.
   - Hybrid loss function: $\mathcal{L}_{total} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$ to conquer class imbalance.
   - Produces probability maps, binary masks, and semi-transparent alert overlays.
   - **Measured Test Set Performance**: IoU: **0.8272** | Dice (F1): **0.9054** | Precision: **0.8642** | Recall: **0.9509**.

3. **Geospatial Vectorization (`src/attribution/vectorizer.py`)**:
   - Extracts georeferenced Shapely polygons directly from binary masks using `rasterio.features.shapes`.
   - Reprojects to metric Coordinate Reference Systems (EPSG:3857 Web Mercator) for exact physical area estimation in square kilometers ($km^2$).

4. **Multi-Factor AIS Vessel Attribution (`src/attribution/ranker.py`)**:
   - Ingests and cleans maritime AIS telemetry CSV files (UTC timestamps, SOG/COG sanity checks, coordinate validation).
   - Constructs vessel trajectories (`LineString`) grouped by MMSI.
   - Evaluates multi-factor score:
     - Spatial distance to spill polygon (in metric km)
     - Temporal delta to satellite overpass time
     - Direct trajectory-spill intersection
     - Cruising speed plausibility ($6-22$ knots) and vessel type risk weighting
   - Prioritizes candidate vessels into risk categories: `HIGH`, `MEDIUM`, `LOW`, `NEGLIGIBLE`.
   - One-click export to CSV and GeoJSON.

5. **Interactive Tactical Streamlit Dashboard (`app.py`)**:
   - Modern 4-tab user interface.
   - Ingestion of SAR imagery (GeoTIFF/PNG) and AIS CSV telemetry.
   - Dual view: Original SAR, Predicted Mask, and Composite Alert Overlay.
   - Full-featured **Folium tactical map** displaying georeferenced spill clusters and vessel trajectory paths colored by risk tier.
   - Prioritized candidate ranking table and configurable threshold sliders.
   - Clear synthetic/benchmark dataset notifications and investigative disclaimers.

---

## 📁 Project Structure

```
NaviSpill AI
├── checkpoints/              # Saved model weights
│   ├── best_model.pth        # Best trained checkpoint (Val IoU: 0.8371)
│   └── latest_model.pth      # Epoch 2 checkpoint
├── datasets/
│   ├── raw/                  # Downloaded raw archives (Images.zip, Masked.zip)
│   ├── processed/            # Processed 256x256 tiles and binary masks
│   │   ├── sar_tiles/        # train (120), val (30), test (30)
│   │   └── masks/            # train (120), val (30), test (30)
│   └── sample/               # Calibrated benchmark demonstration data
│       ├── sample_s1_scene.tif           # 512x512 GeoTIFF (EPSG:4326, Strait of Gibraltar)
│       ├── sample_s1_mask.png            # Ground-truth spill mask
│       └── sample_ais_trajectories.csv   # Synchronized AIS telemetry (5 vessels)
├── docs/                     # Detailed architectural documentation
│   ├── ARCHITECTURE.md       # Architectural diagrams & mathematical formulas
│   ├── MODEL_CARD.md         # Architecture, weights, hyperparameters, loss
│   └── LIMITATIONS.md        # Natural look-alikes, hardware, & ethical caveats
├── experiments/              # Training curve logs & metrics JSON
│   ├── test_metrics.json     # Measured test set metrics
│   ├── training_history.json # Epoch losses and IoU progression
│   └── training_curves.png   # Convergence plots
├── reports/                  # Generated incident attribution reports (CSV/GeoJSON)
├── src/                      # Modular Python package
│   ├── preprocessing/        # SAR calibration & AIS cleaning
│   │   ├── sar_preprocessor.py
│   │   └── ais_preprocessor.py
│   ├── models/               # U-Net architecture & hybrid loss
│   │   ├── unet_resnet.py
│   │   └── losses.py
│   ├── training/             # PyTorch Dataset, training loop, inference
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── inference.py
│   └── attribution/          # GIS vectorization & vessel ranker
│       ├── vectorizer.py
│       └── ranker.py
├── tests/                    # Automated pytest suite (14 unit & integration tests)
├── .streamlit/               # Streamlit configuration
│   └── config.toml
├── app.py                    # Streamlit web application
├── requirements.txt          # Core dependencies
├── PROJECT_PLAN.md           # Master plan & milestones
├── PROGRESS.md               # Live status checklist
├── CHANGELOG.md              # Detailed chronological changelog
├── DATASET_MANIFEST.md       # Provenance, licenses, formats
├── TEST_REPORT.md            # Complete test execution logs
└── DECISIONS.md              # Engineering trade-offs & rationale
```

---

## 🚀 Installation & Execution

### 1. Environment Setup
Clone the repository and install required dependencies:
```bash
cd "C:\Users\Admin\Desktop\NaviSpill AI"
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
Verify that all unit and integration tests pass:
```bash
python -m pytest tests/ -v
```
*(All 14 tests pass covering preprocessing, model forward pass, checkpoint saving, loss functions, polygon vectorization, attribution scoring, and end-to-end integration).*

### 3. Launch the Interactive Dashboard
Launch the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Training & Evaluation

To re-train or fine-tune the model on custom SAR tiles:
```bash
python -m src.training.train
```
- By default, trains with batch size $4$ using AdamW and Cosine Annealing learning rate.
- Automatically saves the best checkpoint to `checkpoints/best_model.pth`.
- Evaluates the model on the held-out test set and saves convergence curves to `experiments/training_curves.png`.

---

## 📊 Measured Model Performance

Evaluated on the held-out test set (`datasets/processed/sar_tiles/test`):

| Metric | Score | Description |
|---|---|---|
| **Intersection over Union (IoU)** | **0.8272** | Overlap accuracy on oil spill pixels |
| **Dice Coefficient (F1 Score)** | **0.9054** | Harmonic mean of precision and recall |
| **Precision** | **0.8642** | Low rate of false positives on sea clutter |
| **Recall (Sensitivity)** | **0.9509** | High capture rate of dark spill formations |
| **Evaluation Loss** | **0.1706** | Hybrid BCE + Dice Loss |

---

## ⚖️ Legal & Investigative Disclaimer

> **ATTENTION**: The computed attribution score and candidate vessel ranking are designed strictly as **investigative prioritization aids** based on spatio-temporal telemetry, kinematics, and geometric proximity. They **do NOT constitute legal or judicial proof of liability**. Innocent vessels may cross pre-existing slicks discharged by earlier non-transmitting vessels ("dark vessels"). Ground-truth verification requires physical oceanic chemical sampling or aerial surveillance.

---

## 📜 Dataset Provenance & Licenses

- **Satellite SAR Imagery**: Derived from the public MIT-licensed satellite dataset (`Thadzy/Oilspill`, Hugging Face).
- **Pretrained Encoder Weights**: ResNet-34 ImageNet weights under the BSD 3-Clause License.
- **AIS Telemetry**: Structured following US Coast Guard (MarineCadastre.gov) public-domain standards.
- **Project License**: MIT License.
