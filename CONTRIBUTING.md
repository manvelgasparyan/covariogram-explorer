# Contributing

Contributions are welcome: bug reports, ideas, new shapes and improvements.

## Reporting a problem or an idea

Open an [issue](https://github.com/manvelgasparyan/covariogram-explorer/issues). For a bug, please
include your operating system, your Python and Matplotlib versions
(`python -c "import sys, matplotlib; print(sys.version, matplotlib.__version__)"`) and the
steps that lead to the problem.

## Development setup

```bash
git clone https://github.com/manvelgasparyan/covariogram-explorer
cd covariogram-explorer
pip install -e ".[test]"
pytest
```

All tests must pass before a pull request is merged. They also run automatically on every
push and pull request.

## Adding a shape

1. **Convex polygons** need no code beyond one line in `make_shapes()`:
   `ConvexPolygon(vertices, "Name", "slug")`. Their covariogram is computed exactly by clipping.
2. **Shapes with a closed-form covariogram** need a subclass of `Shape` (in
   `covariogram_explorer/shapes.py`) that implements:
   - `rings()`: the boundary as closed polylines (outer ring counterclockwise, holes clockwise);
   - `covariogram(P)`: the covariogram for an array `P` of shape `(..., 2)`, written with NumPy
     array operations;
   - `contains(P)`: an exact membership test (used for sampling and images);
   - for curved shapes, exact `perimeter` and `width(u)` if they are known;
   - optionally, `describe(x)` (a short description of the overlap) and `overlay()` (extra
     curves on the map).
3. **Tests.** Add the new shape to `make_shapes()` and give the tests an independent,
   high-resolution reference for it in `reference_polygon()` in
   `tests/test_covariogram_explorer.py`. This is needed only for curved shapes; polygons are
   their own exact reference. Also add its exact area to `exact_area()`. Then run `pytest`: the
   new shape is checked against brute-force clipping, against its area, for the symmetry
   g(-x) = g(x), and for the identity "integral of g = area squared".
4. Please cite the source of the formula in the docstring of the class.

## Style

- Keep the runtime dependencies to NumPy and Matplotlib.
- Keep the numerical modules independent of the graphical one (`app.py`), so that the
  library can be used without a window backend.
