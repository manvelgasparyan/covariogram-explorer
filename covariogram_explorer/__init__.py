"""covariogram_explorer: exact covariograms of planar shapes.

The covariogram of a planar set K is g(x) = area of K ∩ (K + x). This package
provides exact covariograms of the disk, ellipse, annulus, Reuleaux triangle,
parallelograms, triangles and convex polygons; the distributions of distances
between random points and of chord lengths derived from them; the covariogram
of binary images; and an interactive explorer (covariogram-explorer).
"""

__version__ = "1.1.0"

from .geometry import clip_convex, disk_disk_area, lens_unit, polygon_area, regular_polygon
from .shapes import (Annulus, ConvexPolygon, Disk, Ellipse, Parallelogram, ReuleauxTriangle,
                     Shape, Triangle, get_shape, make_shapes)
from .distributions import (chord_length_cdf, distance_cdf, distance_density, distance_moment,
                            isotropic_covariogram, mean_chord_length, mean_distance)
from .images import covariogram_from_image, rasterize

__all__ = [
    "Shape", "Disk", "Ellipse", "Annulus", "ReuleauxTriangle", "Parallelogram", "Triangle",
    "ConvexPolygon", "make_shapes", "get_shape",
    "isotropic_covariogram", "distance_density", "distance_cdf", "distance_moment",
    "mean_distance", "chord_length_cdf", "mean_chord_length",
    "rasterize", "covariogram_from_image",
    "lens_unit", "disk_disk_area", "clip_convex", "polygon_area", "regular_polygon",
    "App", "main",
]


def __getattr__(name):
    # the graphical part is imported only when needed, so that the numerical
    # functions can be used without choosing a Matplotlib window backend
    if name in ("App", "main"):
        from . import app
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
