# NaviSpill AI - Project Progress Tracker
**Project**: Marine Oil Spill Segmentation and Vessel Attribution  
**Course**: CSE471 Capstone Project  
**Current Status**: **COMPLETED & VERIFIED (100%)**  
**Last Updated**: 2026-11-09T19:18:00+05:30  

---

## Live Checklist

### Phase 0: Inspection & Planning
- [x] Inspect existing workspace and adjacent folders (`NaviSpill AI` confirmed target).
- [x] Check Python version (Python 3.10.11), free disk space (~61.5 GB), CUDA status (CPU mode detected).
- [x] Initialize Git repository within `NaviSpill AI`.
- [x] Create project scaffolding (`checkpoints/`, `datasets/`, `docs/`, `experiments/`, `reports/`, `src/`, `tests/`).
- [x] Create `PROJECT_PLAN.md`, `PROGRESS.md`, `CHANGELOG.md`, `DECISIONS.md`, `DATASET_MANIFEST.md`, `TEST_REPORT.md`.
- [x] Complete core dependencies installation via pip (`torch`, `torchvision`, `segmentation_models_pytorch`, `rasterio`, `geopandas`, `shapely`, `pyproj`, `streamlit`, `folium`, `streamlit_folium`, `albumentations`, `opencv-python-headless`).

### Phase 1: Data Acquisition & Preprocessing
- [x] Source / investigate Sentinel-1 SAR oil spill datasets (M4D, Zenodo, HuggingFace).
- [x] Download manageable subset (`Images.zip` 69 MB, `Masked.zip` 271 MB from MIT-licensed `Thadzy/Oilspill`).
- [x] Provide a verified benchmark sample scene with prominent labeling (`datasets/sample/sample_s1_scene.tif` + `sample_s1_mask.png`).
- [x] Generate synchronized benchmark AIS dataset with 5 vessels (`datasets/sample/sample_ais_trajectories.csv`).
- [x] Implement `SARPreprocessor` (NoData handling, contrast percentile clipping, min-max normalization, georeference preservation, sliding window tiling).
- [x] Implement `AISPreprocessor` (UTC datetime parsing, bounding box filter, lat/lon bounds validation, SOG/COG sanity checks, trajectory LineStrings).
- [x] Write and pass unit tests for SAR and AIS preprocessors (`test_sar_preprocessing.py`, `test_ais_preprocessing.py`).

### Phase 2: Deep Learning Segmentation Model
- [x] Implement U-Net with ResNet34 encoder adapted for 1-channel SAR input (validated channel-averaged conv1 kernel initialization).
- [x] Implement combined `BCEWithLogitsLoss` + `DiceLoss`.
- [x] Implement PyTorch Dataset and DataLoader with synchronized geometric augmentations (Albumentations).
- [x] Execute training experiment (CPU-tuned, 2 epochs, batch size 4, validation loop, early stopping, best checkpoint saved).
- [x] Implement inference pipeline (output probability map, binary thresholded mask, visual alert overlay).
- [x] Save loss/metric curves (`experiments/training_curves.png`) and log real evaluation metrics on held-out test set (IoU: 0.8272, Dice: 0.9054, Precision: 0.8642, Recall: 0.9509).

### Phase 3: Geospatial Vectorization & Vessel Attribution
- [x] Implement `MaskVectorizer` to extract GeoPandas polygons from binary masks using affine transforms and CRS (`rasterio.features.shapes`).
- [x] Implement local metric CRS reprojection (Web Mercator EPSG:3857) for physical distance calculations in kilometers.
- [x] Implement `VesselAttributor` with configurable weights (distance to spill, temporal alignment, trajectory intersection, speed plausibility).
- [x] Implement candidate ranking engine with confidence scores and investigative caveats.
- [x] Export results to GeoJSON and CSV.
- [x] Write and pass unit tests (`test_vectorizer.py`, `test_attribution.py`).

### Phase 4: Streamlit Interactive Dashboard
- [x] Build interactive multi-tab Streamlit interface (`app.py`).
- [x] Provide SAR raster upload and AIS CSV upload.
- [x] Real-time inference triggering and mask visualization (Calibrated SAR, Binary Mask, Alert Overlay).
- [x] Interactive Folium map displaying spill polygons and AIS vessel routes colored by risk level.
- [x] Interactive candidate vessel attribution table and configurable thresholds (distance cutoff, time window, factor weights).
- [x] One-click GeoJSON and CSV export for incident response.
- [x] Prominent synthetic/sample dataset disclaimers and legal investigative notice.

### Phase 5: Verification, Testing & Final Documentation
- [x] Run full automated test suite: 14 / 14 tests PASSED.
- [x] Verify Streamlit starts cleanly (`python app.py` exit code 0).
- [x] Update `TEST_REPORT.md` with real execution outputs.
- [x] Complete `README.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/LIMITATIONS.md`.
- [x] Final comprehensive delivery report.
