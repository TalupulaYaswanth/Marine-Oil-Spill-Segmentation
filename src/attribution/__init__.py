"""
Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution.
Phase 3: Vessel Detection and Attribution Engine Module.
"""

from .engine import VesselAttributionEngine

try:
    from .vectorizer import SpillMaskVectorizer
    __all__ = ["VesselAttributionEngine", "SpillMaskVectorizer"]
except ImportError:
    __all__ = ["VesselAttributionEngine"]
