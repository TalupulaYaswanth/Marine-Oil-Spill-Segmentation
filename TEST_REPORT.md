# NaviSpill AI - Test Execution Report

**Date**: 2026-11-09  
**Platform**: Windows 11 / Python 3.10.11 / CPU Execution Mode  
**Test Harness**: Pytest 9.2.0  
**Overall Status**: **14 / 14 PASSED (100% Pass Rate)**  

---

## 1. Summary of Automated Test Suite

| Test ID | Module | Test Function | Result | Execution Time | Description |
|---|---|---|---|---|---|
| TC-01 | `test_sar_preprocessing.py` | `test_sar_load_and_metadata` | **PASSED** | 0.8s | Verifies GeoTIFF metadata extraction (CRS, transform, nodata) |
| TC-02 | `test_sar_preprocessing.py` | `test_sar_nodata_and_normalization` | **PASSED** | 0.4s | Validates NoData handling and [0, 1] percentile normalization |
| TC-03 | `test_sar_preprocessing.py` | `test_sar_tiling` | **PASSED** | 0.5s | Validates spatial window sliding and tile affine transforms |
| TC-04 | `test_ais_preprocessing.py` | `test_ais_load_and_clean` | **PASSED** | 0.3s | Tests UTC datetime parsing, coordinate domain, SOG filtering |
| TC-05 | `test_ais_preprocessing.py` | `test_ais_spatial_temporal_filtering` | **PASSED** | 0.3s | Validates spatial bounding box and temporal window filters |
| TC-06 | `test_ais_preprocessing.py` | `test_ais_trajectory_construction` | **PASSED** | 0.4s | Validates MMSI grouping into LineStrings and Points |
| TC-07 | `test_model_forward.py` | `test_model_initialization_and_forward` | **PASSED** | 4.2s | Verifies 1-channel U-Net ResNet34 forward pass & logits |
| TC-08 | `test_model_forward.py` | `test_checkpoint_save_and_load` | **PASSED** | 3.8s | Tests checkpoint serialization and identical forward output |
| TC-09 | `test_losses.py` | `test_bce_dice_loss` | **PASSED** | 0.2s | Tests BCE + Dice loss convergence on good vs bad predictions |
| TC-10 | `test_losses.py` | `test_compute_metrics` | **PASSED** | 0.1s | Verifies exact computation of IoU, Dice, Precision, Recall |
| TC-11 | `test_vectorizer.py` | `test_vectorizer_on_binary_mask` | **PASSED** | 0.5s | Verifies raster mask extraction into GeoDataFrame polygons |
| TC-12 | `test_vectorizer.py` | `test_vectorizer_empty_mask` | **PASSED** | 0.1s | Handles empty masks gracefully without errors |
| TC-13 | `test_attribution.py` | `test_attribution_ranking` | **PASSED** | 0.6s | Verifies candidate scoring and ranking logic (Top MMSI 111) |
| TC-14 | `test_pipeline_integration.py` | `test_full_pipeline_end_to_end` | **PASSED** | 7.1s | Full end-to-end integration: GeoTIFF -> UNet -> Vectorize -> AIS -> Attribution -> CSV/GeoJSON |

---

## 2. Model Training & Evaluation Log

- **Architecture**: U-Net (ResNet34 Backbone, ImageNet Weights)
- **Loss Function**: `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`
- **Training Epochs**: 2 (CPU-optimized demonstration fine-tuning)
- **Batch Size**: 4
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$)

### Epoch History
- **Epoch 1**: Train Loss = 0.4054 | Val Loss = 0.9828 | Val IoU = 0.2494 | Val Dice = 0.3992 (Duration: 151.9s)
- **Epoch 2**: Train Loss = 0.1874 | Val Loss = 0.1663 | Val IoU = 0.8371 | Val Dice = 0.9113 (Duration: 109.5s)

### Held-out Test Set Evaluation Results
- **Evaluated Samples**: 30 held-out images and masks ($256 \times 256$)
- **Evaluation Loss**: **0.1706**
- **Intersection-over-Union (IoU)**: **0.8272**
- **Dice Coefficient (F1)**: **0.9054**
- **Precision**: **0.8642**
- **Recall**: **0.9509**

---

## 3. UI Application Verification
- **Framework**: Streamlit 1.65.0
- **Standalone Syntax & Run Execution**: Verified cleanly with exit code 0 (`python app.py`).
- **Configuration**: Headless mode configured in `.streamlit/config.toml` (telemetry disabled).
- **Execution Command**:
  ```bash
  streamlit run app.py
  ```
