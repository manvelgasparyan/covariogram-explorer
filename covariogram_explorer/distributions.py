"""Distributions derived from the covariogram.

* Distances: X and Y independent and uniform in K. The difference X - Y has the
  density g(x) / area(K)^2, so the distribution of |X - Y| follows from the
  rotational average of g.
* Chords (convex K): -d/dr g(r u) = width(u') (1 - F(u, r)), where u' is
  perpendicular to u (Matheron, 1975). Averaging over directions gives the
  chord length distribution of isotropic uniform random lines,
  F(r) = 1 + (pi / perimeter) d/dr gbar(r).
"""

import numpy as np

from .geometry import PI
from .shapes import ConvexPolygon, Parallelogram, Triangle, _unit


def _directions(n_theta):
    theta = 2 * PI * np.arange(n_theta) / n_theta
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _radial_rule(a, b, n, breaks=()):
    """Gauss-Legendre nodes and weights on [a, b], split at the given breaks."""
    edges = [a] + [t for t in sorted(breaks) if a < t < b] + [b]
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = np.concatenate([lo + (hi - lo) * (x + 1) / 2 for lo, hi in zip(edges[:-1], edges[1:])])
    weights = np.concatenate([(hi - lo) * w / 2 for lo, hi in zip(edges[:-1], edges[1:])])
    return nodes, weights


def _as_output(values, like):
    return float(values) if np.ndim(like) == 0 else values


def isotropic_covariogram(shape, r, n_theta=720):
    """Rotational average of g: (1 / 2 pi) times the integral of g(r e(theta))."""
    r = np.asarray(r, dtype=float)
    P = r[..., None, None] * _directions(n_theta)
    return _as_output(shape.covariogram(P).mean(axis=-1), r)


def distance_density(shape, s, n_theta=720):
    """Density at s of the distance |X - Y| between independent uniform points of K."""
    s = np.asarray(s, dtype=float)
    dens = 2 * PI * s * np.asarray(isotropic_covariogram(shape, s, n_theta)) / shape.area ** 2
    return _as_output(dens, s)


def distance_cdf(shape, s, n_r=200, n_theta=720):
    """P(|X - Y| <= s) for independent uniform points X, Y of K."""
    s = np.asarray(s, dtype=float)
    out = np.zeros(s.shape)
    S = shape.support_radius()
    for idx, si in np.ndenumerate(s):
        if si > 0:
            rho, w = _radial_rule(0.0, min(si, S), n_r, shape.radial_kinks())
            gbar = np.asarray(isotropic_covariogram(shape, rho, n_theta))
            out[idx] = 2 * PI * np.sum(w * rho * gbar) / shape.area ** 2
    return _as_output(np.clip(out, 0.0, 1.0), s)


def distance_moment(shape, k=1, n_r=400, n_theta=720):
    """E|X - Y|^k for independent uniform points X, Y of K."""
    rho, w = _radial_rule(0.0, shape.support_radius(), n_r, shape.radial_kinks())
    gbar = np.asarray(isotropic_covariogram(shape, rho, n_theta))
    return float(2 * PI * np.sum(w * rho ** (k + 1) * gbar) / shape.area ** 2)


def mean_distance(shape):
    """E|X - Y|, the mean distance between two independent uniform points of K.

    For polygonal shapes, g has kinks along every ray, so the quadrature error
    decreases like the square of the step size; two resolutions are then
    combined by Richardson extrapolation."""
    if isinstance(shape, (Parallelogram, Triangle)):
        coarse, fine = distance_moment(shape, 1, 800, 1440), distance_moment(shape, 1, 1600, 2880)
    elif isinstance(shape, ConvexPolygon):            # clipping is slower: fewer points
        coarse, fine = distance_moment(shape, 1, 200, 360), distance_moment(shape, 1, 400, 720)
    else:
        return distance_moment(shape, 1)
    return (4 * fine - coarse) / 3


def _derivative(f, r, h):
    lo = np.maximum(r - h, 0.0)
    hi = r + h
    return (np.asarray(f(hi)) - np.asarray(f(lo))) / (hi - lo)


def chord_length_cdf(shape, r, direction=None, n_theta=720, h=None):
    """Distribution function of the length of a random chord of a convex shape.

    direction=None: the chord cut out by an isotropic uniform random line that
        hits K, F(r) = 1 + (pi / perimeter) d/dr gbar(r).
    direction=u (angle in radians or a vector): the chord on a random line
        parallel to u, whose offset is uniform among the lines that hit K,
        F(u, r) = 1 + d/dr g(r u) / width(u'), with u' perpendicular to u.
    The derivatives are central differences with step h."""
    if not shape.convex:
        raise ValueError("chord length distributions are provided for convex shapes only")
    r = np.asarray(r, dtype=float)
    h = 1e-6 * shape.support_radius() if h is None else h
    if direction is None:
        deriv = _derivative(lambda t: isotropic_covariogram(shape, t, n_theta), r, h)
        F = 1 + PI / shape.perimeter * deriv
    else:
        u = _unit(direction)
        deriv = _derivative(lambda t: shape.covariogram(np.asarray(t)[..., None] * u), r, h)
        F = 1 + deriv / shape.width((-u[1], u[0]))
    F = np.where(r <= 0, 0.0, np.clip(F, 0.0, 1.0))
    return _as_output(F, r)


def mean_chord_length(shape):
    """Mean chord length of an isotropic uniform random line hitting a convex
    shape: pi * area / perimeter (Cauchy-Crofton)."""
    if not shape.convex:
        raise ValueError("chord length distributions are provided for convex shapes only")
    return PI * shape.area / shape.perimeter
