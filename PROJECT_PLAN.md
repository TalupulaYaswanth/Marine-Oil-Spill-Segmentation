# NaviSpill AI - Project Execution Plan
**Title**: Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution Using Satellite Imagery and AIS Data  
**Student Course Level**: CSE471 Capstone Project  
**Author/Engineer**: Autonomous AI/ML Geospatial Lead  
**Created Date**: 2026-11-09  
**Execution Window**: 2–3 Hours Target  

---

## 1. Project Objectives
1. **Satellite SAR Segmentation**: Preprocess Synthetic Aperture Radar (SAR) imagery (Sentinel-1 SAR format / GeoTIFF / PNG) and perform pixel-level semantic segmentation of marine oil spills using a deep learning architecture (U-Net with ImageNet-pretrained ResNet34 backbone).
2. **Geospatial Vectorization**: Convert detected raster oil spill masks into georeferenced polygons (GeoJSON/Shapefile/GeoPandas) preserving spatial CRS (Coordinate Reference System) and affine transforms.
3. **AIS Spatio-Temporal Attribution**: Correlate marine vessel AIS trajectories with spill polygons across space, time, course over ground (COG), speed over ground (SOG), and proximity to compute an objective, multi-factor attribution score for candidate vessels.
4. **Interactive Streamlit Dashboard**: Provide a modern student-level web interface for SAR image ingestion, mask generation, GIS map visualization (Folium), AIS track overlay, candidate ranking table, and report/data export (CSV/GeoJSON).

---

## 2. System Architecture & Component Mapping
```
NaviSpill AI
├── checkpoints/              # Saved model weights (.pth) & training checkpoints
├── datasets/
│   ├── raw/                  # Downloaded raw SAR & AIS datasets
│   ├── sample/               # Verified sample & benchmark datasets
│   └── processed/            # Preprocessed 256x256 / 512x512 tiles & cleaned AIS CSVs
├── docs/                     # Technical documentation & project cards
│   ├── ARCHITECTURE.md       # Pipeline architecture & math formulas
│   ├── MODEL_CARD.md         # Architecture, weights, hyperparameters, loss
│   └── LIMITATIONS.md        # Real-world limitations & synthetic notices
├── experiments/              # Training curve logs & metrics JSON
├── reports/                  # Generated incident attribution reports (CSV/GeoJSON)
├── src/                      # Modular Python package
│   ├── __init__.py
│   ├── preprocessing/        # SAR calibration, normalization, tiling, AIS cleaning
│   │   ├── __init__.py
│   │   ├── sar_preprocessor.py
│   │   └── ais_preprocessor.py
│   ├── models/               # U-Net ResNet34 model definition & loss functions
│   │   ├── __init__.py
│   │   ├── unet_resnet.py
│   │   └── losses.py
│   ├── training/             # PyTorch dataset, dataloader, trainer, evaluation
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── train.py
│   ├── attribution/          # Vectorization, spatial join, scoring logic
│   │   ├── __init__.py
│   │   ├── vectorizer.py
│   │   └── ranker.py
│   └── utils/                # Geo helpers, visualizers, CRS reprojection
│       ├── __init__.py
│       └── geo_utils.py
├── tests/                    # Automated unit & integration tests
│   ├── test_sar_preprocessing.py
│   ├── test_model_forward.py
│   ├── test_attribution.py
│   └── test_pipeline_integration.py
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── PROJECT_PLAN.md           # Master plan & milestones
├── PROGRESS.md               # Live status checklist & updates
├── CHANGELOG.md              # Granular commit & change log
├── DATASET_MANIFEST.md       # Provenance, licenses, formats
├── TEST_REPORT.md            # Test executions, pass/fail status
└── DECISIONS.md              # Engineering trade-offs & rationale
```

---

## 3. Execution Milestones & Time Allocation

| Phase | Milestone | Estimated Duration | Target Deliverable |
|---|---|---|---|
| **Phase 0** | Inspection & Setup | 10 mins | Repo init, requirements, docs skeleton, hardware check |
| **Phase 1** | Data Acquisition & Preprocessing | 25 mins | Sentinel-1 SAR subset + AIS dataset, GeoTIFF preprocessor, AIS cleaner |
| **Phase 2** | Model Development & Training | 35 mins | 1-channel ResNet34 U-Net, BCE+Dice loss, CPU-tuned train/eval loop, weights |
| **Phase 3** | Vectorization & Attribution | 25 mins | Mask-to-polygon, metric CRS reprojection, trajectory distance & ranker |
| **Phase 4** | Streamlit Interactive Dashboard | 30 mins | Ingestion UI, model inference runner, Folium GIS map, ranker table |
| **Phase 5** | Testing, Verification & Final Report | 20 mins | Pytest suite, pipeline verification, final documentation update |

---

## 4. Key Constraints & Safety Rules
- **No Hallucinated Data**: Clearly distinguish between real Sentinel-1 SAR data, real AIS data, and synthetically benchmarked test data.
- **CPU / Resource Efficiency**: Since CUDA is unavailable (`torch: cpu`), batch sizes must be small ($B=4$), images sized to $256 \times 256$ or $512 \times 512$, and training limited to short epochs (1-3 epochs for demonstration) with early stopping.
- **Investigative Attribution Disclaimer**: Attribution scores are probabilistic investigative assistance tools, never judicial proof of liability.
- **Real-Time Testing**: Every component will be executed and verified with unit tests.
