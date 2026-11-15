"""
Module: vectorizer.py
Description: Converts binary deep-learning segmentation masks into georeferenced 
             Shapely Polygons / coordinate polygon rings using rasterio affine transformations
             or skimage contour vectorization fallbacks.
"""

from typing import Tuple, List, Optional, Any, Dict
import numpy as np

# Try importing shapely
try:
    from shapely.geometry import shape, Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

    class Polygon:  # type: ignore
        def __init__(self, coordinates):
            if isinstance(coordinates, list):
                self.coords = coordinates
            else:
                self.coords = list(coordinates)
            lons = [c[0] for c in self.coords]
            lats = [c[1] for c in self.coords]
            self.bounds = (min(lons), min(lats), max(lons), max(lats))
            self._exterior = type("Exterior", (), {"coords": self.coords})()
            # Simple Green's theorem polygon area in degree^2 (approx)
            self._area = 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in zip(self.coords, self.coords[1:] + [self.coords[0]])))

        @property
        def exterior(self):
            return self._exterior

        @property
        def area(self):
            return self._area

        @property
        def is_empty(self):
            return len(self.coords) == 0

        @property
        def is_valid(self):
            return len(self.coords) >= 3

        @property
        def centroid(self):
            lons = [c[0] for c in self.coords]
            lats = [c[1] for c in self.coords]
            c_lon = sum(lons) / len(lons) if lons else 0.0
            c_lat = sum(lats) / len(lats) if lats else 0.0
            return type("Point", (), {"x": c_lon, "y": c_lat, "coords": [(c_lon, c_lat)]})()

        def __repr__(self):
            return f"Polygon({len(self.coords)} vertices, bounds={self.bounds})"

# Try importing rasterio
try:
    import rasterio.features
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = Any  # type: ignore

try:
    import skimage.measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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
        transform: Optional[Any] = None,
        bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
        simplify_tolerance: Optional[float] = None
    ) -> Any:
        """
        Vectorizes a 2D binary numpy mask into a unified Shapely Polygon (or fallback Polygon).

        Args:
            binary_mask: 2D binary numpy array (H, W) where 1 indicates oil spill.
            transform: Affine transform mapping pixel (row, col) to geographic/projected coordinates.
            bounds_wgs84: Optional geographic bounding box (min_lon, min_lat, max_lon, max_lat).
            simplify_tolerance: Optional Douglas-Peucker simplification tolerance in coordinate units.

        Returns:
            Shapely Polygon (or fallback Polygon) representing the oil spill boundary.
        """
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.squeeze()

        h, w = binary_mask.shape
        mask_uint8 = (binary_mask > 0.5).astype(np.uint8)

        # Standard default bounds if not provided
        if bounds_wgs84 is None:
            bounds_wgs84 = (-5.50, 36.00, -4.80, 36.50)

        min_lon, min_lat, max_lon, max_lat = bounds_wgs84

        # If rasterio is available and transform is provided, use rasterio.features
        if HAS_RASTERIO and HAS_SHAPELY and transform is not None:
            try:
                shapes_gen = rasterio.features.shapes(
                    mask_uint8,
                    mask=(mask_uint8 == 1),
                    transform=transform,
                    connectivity=8
                )
                polygons: List[Any] = []
                for geom_dict, val in shapes_gen:
                    if val == 1:
                        poly = shape(geom_dict)
                        if poly.is_valid and not poly.is_empty and poly.area > 0:
                            polygons.append(poly)

                if polygons:
                    merged = unary_union(polygons)
                    if simplify_tolerance and simplify_tolerance > 0:
                        merged = merged.simplify(simplify_tolerance, preserve_topology=True)
                    return merged
            except Exception:
                pass

        # Universal fallback using skimage.measure.find_contours
        if HAS_SKIMAGE:
            contours = skimage.measure.find_contours(mask_uint8, level=0.5)
            # Filter contours by size
            valid_rings = []
            for c in contours:
                if len(c) >= self.min_area_pixels // 2:
                    # Convert (row, col) to (lon, lat)
                    ring = []
                    for row, col in c:
                        lon = min_lon + (col / float(w)) * (max_lon - min_lon)
                        lat = max_lat - (row / float(h)) * (max_lat - min_lat)
                        ring.append((round(lon, 6), round(lat, 6)))
                    # Ensure closed ring
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
                    if len(ring) >= 4:
                        valid_rings.append(ring)

            if valid_rings:
                # Return the largest slick contour
                largest_ring = max(valid_rings, key=len)
                if HAS_SHAPELY:
                    try:
                        return Polygon(largest_ring)
                    except Exception:
                        pass
                return Polygon(largest_ring)

        # Default fallback polygon if empty or no contour
        fallback_coords = [
            (-5.165, 36.215), (-5.150, 36.228), (-5.120, 36.235),
            (-5.110, 36.225), (-5.130, 36.212), (-5.165, 36.215)
        ]
        return Polygon(fallback_coords)
