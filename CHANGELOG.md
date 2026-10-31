# Changelog

All notable changes to this project are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] - 2026-11-01

### Added
- Random points: `Shape.sample(n, rng)` and the exact membership test `Shape.contains(P)`.
- Geometric quantities: `Shape.perimeter` and `Shape.width(u)`.
- Distances between two random points: `isotropic_covariogram`, `distance_density`,
  `distance_cdf`, `distance_moment`, `mean_distance`.
- Chord lengths of convex shapes: `chord_length_cdf` (a fixed direction or isotropic random
  lines) and `mean_chord_length`.
- Covariograms of binary images: `rasterize`, `covariogram_from_image`.
- `get_shape(name)`; `python -m covariogram_explorer`.
- Replication script for the accompanying paper; tests for all new functions.

### Changed
- The single module became the package `covariogram_explorer` (`geometry`, `shapes`,
  `distributions`, `images`, `app`). Existing imports such as
  `from covariogram_explorer import make_shapes` keep working; run the program with
  `covariogram-explorer` or `python -m covariogram_explorer`.

## [1.0.0] - 2026-11-01

First public release.

### Added
- Gallery of twelve shapes: disk, ellipse, annulus, Reuleaux triangle, square, rectangle,
  parallelogram, equilateral triangle, right triangle, regular pentagon, hexagon and octagon.
- Explorer with a colour map of the covariogram over its support, the shape, its translate
  and their overlap, and read-outs of g(x), its share of the area, |x| and the shape of the
  overlap.
- Closed-form covariograms for nine shapes and exact polygon clipping for convex polygons.
- Mouse, keyboard and command-line control (`--list`, `--shape`, `--x`, `--save`, `--version`).
- Library interface, vectorised over NumPy arrays, and one-line addition of convex polygons.
- Examples, including distance distributions and mean distances computed from the covariogram.
- Test suite and continuous integration on Linux, Windows and macOS.
- Automatic build of a Windows executable for tagged releases.
