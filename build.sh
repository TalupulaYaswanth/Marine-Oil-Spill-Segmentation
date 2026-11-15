#!/usr/bin/env bash
# ==============================================================================
# Render Build Script: Marine Oil Spill Sentinel & Vessel Attribution
# ==============================================================================
set -e

echo "=== [1/4] Upgrading Pip and Build Tools ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/4] Installing Python Dependencies ==="
pip install -r requirements.txt

echo "=== [3/4] Pre-Generating Synthetic Benchmark SAR & AIS Data ==="
python generate_sample_data.py
python src/setup_real_world_data.py || true

echo "=== [4/4] Verifying Pipeline Integrity ==="
python test_system_integrity.py

echo "=== Build Completed Successfully! Ready for Deployment ==="
