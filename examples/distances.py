"""Distance between two independent uniform random points, from the covariogram."""
import numpy as np
import covariogram_explorer as ce

print(f"{'shape':22s} {'E|X-Y|':>9s} {'E|X-Y|/sqrt(area)':>18s}")
for shape in ce.make_shapes():
    if shape.closed_form:
        m = ce.mean_distance(shape)
        print(f"{shape.name:22s} {m:9.6f} {m / np.sqrt(shape.area):18.6f}")

# the whole distribution, checked against a simulation
R = ce.get_shape("reuleaux-triangle")
rng = np.random.default_rng(1)
d = np.hypot(*(R.sample(100_000, rng) - R.sample(100_000, rng)).T)
for s in [0.2, 0.4, 0.6, 0.8]:
    print(f"P(|X-Y| <= {s}): exact {ce.distance_cdf(R, s):.4f}, simulated {np.mean(d <= s):.4f}")
