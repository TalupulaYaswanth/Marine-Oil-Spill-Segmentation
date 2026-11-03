"""
Module: vectorizer.py
Description: Converts binary deep-learning segmentation masks into georeferenced 
             Shapely Polygons / MultiPolygons using rasterio affine transformations.
"""

from typing import Tuple, List, Optional
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import rasterio.features
from rasterio.transform import Affine


class SpillMaskVectorizer:
    """
    Transforms pixel-space binary segmentation masks into geospatial Shapely polygons
    aligned with native or geographic coordinate systems.
    """

    def __init__(self, min_area_pixels: int = 25):
        """
        Args:
            min_area_pixels: Minimum contour area threshold in pixels to filter small noise speckles.
        """
        self.min_area_pixels = min_area_pixels

    def mask_to_polygons(
        self,
        binary_mask: np.ndarray,
        transform: Affine,
        simplify_tolerance: Optional[float] = None
    ) -> Polygon:
        """
        Vectorizes a 2D binary numpy mask into a unified Shapely Polygon.

        Args:
            binary_mask: 2D binary numpy array (H, W) where 1 indicates oil spill.
            transform: Affine transform mapping pixel (row, col) to geographic/projected coordinates.
            simplify_tolerance: Optional Douglas-Peucker simplification tolerance in coordinate units.

        Returns:
            Shapely Polygon (or MultiPolygon) representing the oil spill boundary.
        """
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.squeeze()

        mask_uint8 = (binary_mask > 0.5).astype(np.uint8)

        # Extract vector polygon geometries from raster contours
        shapes_gen = rasterio.features.shapes(
            mask_uint8,
            mask=(mask_uint8 == 1),
            transform=transform,
            connectivity=8
        )

        polygons: List[Polygon] = []
        for geom_dict, val in shapes_gen:
            if val == 1:
                poly = shape(geom_dict)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)

        if not polygons:
            # Fallback for empty prediction
            return Polygon()

        # Merge contiguous/overlapping spill components into a unified geometry
        merged_spill = unary_union(polygons)

        if simplify_tolerance and simplify_tolerance > 0:
            merged_spill = merged_spill.simplify(simplify_tolerance, preserve_topology=True)

        return merged_spill

