"""Evaluating covariograms."""
import numpy as np
import covariogram_explorer as ce

x = np.array([0.2, 0.1])
for slug in ["disk", "annulus", "reuleaux-triangle", "right-triangle", "hexagon"]:
    s = ce.get_shape(slug)
    print(f"{s.name:18s} g(x) = {s.value(x):.6f}  ({s.describe(x)})")

# vectorised: g along the ray at 30 degrees, in one call
r = np.linspace(0, 1, 6)
ray = r[:, None] * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
print(np.round(ce.get_shape("reuleaux-triangle").covariogram(ray), 4))
