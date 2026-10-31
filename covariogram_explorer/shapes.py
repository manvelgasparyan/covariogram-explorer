"""Planar shapes with exact covariograms.

Every shape is centred at the origin and provides

* ``covariogram(P)``: the covariogram g(x) = area of K ∩ (K + x), vectorised over
  an array ``P`` of shape ``(..., 2)``;
* ``contains(P)``: exact membership test, used for random sampling and images;
* geometric quantities: ``area``, ``perimeter``, ``width(u)``, ``support_radius()``.
"""

import numpy as np
from matplotlib.path import Path

from .geometry import (PI, S3, arc, centred_ccw, clip_convex, disk_disk_area,
                       lens_unit, polygon_area, regular_polygon)


def _unit(u):
    """A unit vector from an angle (in radians) or from a vector."""
    u = np.asarray(u, dtype=float)
    if u.ndim == 0:
        return np.array([np.cos(u), np.sin(u)])
    return u / np.hypot(u[0], u[1])


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
class Shape:
    """A planar shape centred at the origin.

    Subclasses provide rings() (the boundary: outer ring counterclockwise,
    holes clockwise) and covariogram(P) for an array P of shape (..., 2).
    They should also override contains(P) with an exact test; the default
    uses the polygonal boundary."""

    name = slug = method = ""
    closed_form = True
    convex = True

    def rings(self):
        raise NotImplementedError

    def covariogram(self, P):
        raise NotImplementedError

    def describe(self, x):
        """Short description of the overlap K ∩ (K + x)."""
        return ""

    def overlay(self):
        """Extra curves to draw on the map of g (list of (n, 2) arrays)."""
        return []

    def contains(self, P):
        """Boolean array telling which points of P (shape (..., 2)) lie in K."""
        P = np.asarray(P, dtype=float)
        inside = self.path().contains_points(P.reshape(-1, 2))
        return inside.reshape(P.shape[:-1])

    def radial_kinks(self):
        """Radii at which g has a kink in every direction (used by quadrature)."""
        return ()

    @property
    def noun(self):
        """Name as used inside a sentence."""
        return self.name[0].lower() + self.name[1:]

    # derived quantities ------------------------------------------------------
    def path(self):
        verts, codes = [], []
        for ring in self.rings():
            ring = np.vstack([ring, ring[:1]])
            c = np.full(len(ring), Path.LINETO)
            c[0], c[-1] = Path.MOVETO, Path.CLOSEPOLY
            verts.append(ring)
            codes.append(c)
        return Path(np.vstack(verts), np.concatenate(codes))

    def value(self, x):
        return float(self.covariogram(np.asarray(x, dtype=float).reshape(1, 2))[0])

    @property
    def area(self):
        if not hasattr(self, "_area"):
            self._area = self.value((0.0, 0.0))
        return self._area

    def extent(self):
        return float(np.max(np.hypot(*self.rings()[0].T)))

    def support_outline(self, n=720):
        """Boundary of the difference body K - K, the support of g."""
        pts = self.rings()[0]
        t = np.linspace(0, 2 * PI, n, endpoint=False)
        proj = pts @ np.vstack([np.cos(t), np.sin(t)])
        return pts[np.argmax(proj, axis=0)] - pts[np.argmin(proj, axis=0)]

    def support_radius(self):
        """Largest |x| with g(x) > 0, i.e. the diameter of K."""
        if not hasattr(self, "_support_radius"):
            self._support_radius = float(np.max(np.hypot(*self.support_outline().T)))
        return self._support_radius

    def width(self, u):
        """Length of the projection of K onto the line spanned by u
        (u is an angle in radians or a vector)."""
        proj = self.rings()[0] @ _unit(u)
        return float(proj.max() - proj.min())

    @property
    def perimeter(self):
        """Total length of the boundary."""
        total = 0.0
        for ring in self.rings():
            closed = np.vstack([ring, ring[:1]])
            total += np.sum(np.hypot(*np.diff(closed, axis=0).T))
        return float(total)

    def bounding_box(self):
        """Lower-left and upper-right corners of a box containing K."""
        pts = self.rings()[0]
        return pts.min(axis=0), pts.max(axis=0)

    def sample(self, n, rng=None):
        """n independent random points, uniformly distributed in K.

        `rng` is a numpy Generator or a seed. Rejection sampling from the
        bounding box with the exact test contains()."""
        rng = np.random.default_rng(rng)
        lo, hi = self.bounding_box()
        pad = 1e-9 * float(np.max(hi - lo))
        lo, hi = lo - pad, hi + pad
        accepted = []
        count = 0
        while count < n:
            m = max(256, int(1.2 * (n - count) * np.prod(hi - lo) / self.area))
            P = lo + (hi - lo) * rng.random((m, 2))
            P = P[self.contains(P)]
            accepted.append(P)
            count += len(P)
        return np.vstack(accepted)[:n]


class Disk(Shape):
    def __init__(self, radius):
        self.r = radius
        self.name, self.slug = "Disk", "disk"
        self.method = (r"closed form:  $g(x) = r^2 L(|x|/r)$,   "
                       r"$L(d) = 2\arccos(d/2) - (d/2)\sqrt{4-d^2}$  (area of a lens)")

    def rings(self):
        return [arc((0, 0), self.r, 0, 2 * PI, 256)]

    def covariogram(self, P):
        P = np.asarray(P, dtype=float)
        return self.r**2 * lens_unit(np.hypot(P[..., 0], P[..., 1]) / self.r)

    def contains(self, P):
        P = np.asarray(P, dtype=float)
        return np.hypot(P[..., 0], P[..., 1]) <= self.r

    def width(self, u):
        return 2.0 * self.r

    @property
    def perimeter(self):
        return 2 * PI * self.r

    def bounding_box(self):
        return np.array([-self.r, -self.r]), np.array([self.r, self.r])

    def describe(self, x):
        return "a lens (2 arcs)"


class Ellipse(Shape):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.name, self.slug = "Ellipse", "ellipse"
        self.method = (r"closed form:  $g(x) = ab\,L\left(\sqrt{(x_1/a)^2 + (x_2/b)^2}\right)$"
                       r"   (affine image of the unit disk)")

    def rings(self):
        t = np.linspace(0, 2 * PI, 256, endpoint=False)
        return [np.column_stack([self.a * np.cos(t), self.b * np.sin(t)])]

    def covariogram(self, P):
        P = np.asarray(P, dtype=float)
        return self.a * self.b * lens_unit(np.hypot(P[..., 0] / self.a, P[..., 1] / self.b))

    def contains(self, P):
        P = np.asarray(P, dtype=float)
        return (P[..., 0] / self.a) ** 2 + (P[..., 1] / self.b) ** 2 <= 1

    def width(self, u):
        u = _unit(u)
        return float(2 * np.hypot(self.a * u[0], self.b * u[1]))

    @property
    def perimeter(self):
        # trapezoidal rule for a smooth periodic integrand: exact to rounding
        t = np.linspace(0, 2 * PI, 4096, endpoint=False)
        return float(2 * PI * np.mean(np.hypot(self.a * np.sin(t), self.b * np.cos(t))))

    def bounding_box(self):
        return np.array([-self.a, -self.b]), np.array([self.a, self.b])

    def describe(self, x):
        return "lens-shaped (2 arcs)"


class Annulus(Shape):
    def __init__(self, outer, inner):
        self.R, self.r = outer, inner
        self.convex = False
        self.name, self.slug = "Annulus", "annulus"
        self.method = (r"closed form:  $g = A_{RR} - 2A_{Rr} + A_{rr}$,   "
                       r"$A_{ab}$ = area of two overlapping disks with radii $a$, $b$")

    def rings(self):
        return [arc((0, 0), self.R, 0, 2 * PI, 256),
                arc((0, 0), self.r, 0, 2 * PI, 256)[::-1]]

    def covariogram(self, P):
        P = np.asarray(P, dtype=float)
        d = np.hypot(P[..., 0], P[..., 1])
        return (disk_disk_area(self.R, self.R, d) - 2 * disk_disk_area(self.R, self.r, d)
                + disk_disk_area(self.r, self.r, d))

    def contains(self, P):
        P = np.asarray(P, dtype=float)
        d = np.hypot(P[..., 0], P[..., 1])
        return (d >= self.r) & (d <= self.R)

    def width(self, u):
        return 2.0 * self.R

    @property
    def perimeter(self):
        return 2 * PI * (self.R + self.r)

    def bounding_box(self):
        return np.array([-self.R, -self.R]), np.array([self.R, self.R])

    def radial_kinks(self):
        return (self.R - self.r, 2 * self.r, self.R + self.r)

    def describe(self, x):
        return "holes overlap" if np.hypot(*x) < 2 * self.r else "holes apart"


class ReuleauxTriangle(Shape):
    """Reuleaux triangle of width h, closed form of Gasparyan (2026)."""

    A = np.array([0.0, 1 / S3])
    B = np.array([-0.5, -0.5 / S3])
    C = np.array([0.5, -0.5 / S3])
    W = np.array([[np.cos(k * PI / 3), np.sin(k * PI / 3)] for k in range(6)])
    noun = "Reuleaux triangle"

    def __init__(self, width):
        self.h = width
        self.name, self.slug = "Reuleaux triangle", "reuleaux-triangle"
        self.method = ("closed form of Gasparyan (2026): on each petal and curvy triangle, "
                       "half the sum of two lens areas plus a linear term")

    def rings(self):
        pts = np.vstack([arc(self.A, 1, 4 * PI / 3, 5 * PI / 3),
                         arc(self.B, 1, 0, PI / 3),
                         arc(self.C, 1, 2 * PI / 3, PI)])
        return [self.h * pts]

    def _petal(self, q):
        """Index k of the petal containing q (width-1 coordinates), or None."""
        for k in range(6):
            a, b = self.W[(k - 1) % 6], self.W[(k + 1) % 6]
            if np.hypot(*(q - a)) <= 1 and np.hypot(*(q - b)) <= 1:
                return k
        return None

    def covariogram(self, P):
        Q = np.asarray(P, dtype=float) / self.h
        x, y = Q[..., 0], Q[..., 1]
        g = np.zeros(x.shape)
        todo = np.hypot(x, y) < 1
        W = self.W
        for k in range(6):                                   # petals P_k
            a, b = W[(k - 1) % 6], W[(k + 1) % 6]
            m = (todo & (np.hypot(x - a[0], y - a[1]) <= 1)
                 & (np.hypot(x - b[0], y - b[1]) <= 1))
            if m.any():
                w, xm, ym = W[k], x[m], y[m]
                g[m] = ((lens_unit(np.hypot(xm, ym)) + lens_unit(np.hypot(xm + w[0], ym + w[1]))) / 2
                        + S3 / 2 * (xm * w[0] + ym * w[1]) - PI / 3 - S3 / 4)
                todo &= ~m
        if todo.any():                                       # curvy triangles T_k
            xr, yr = x[todo], y[todo]
            k = np.floor(np.mod(np.arctan2(yr, xr), 2 * PI) / (PI / 3)).astype(int) % 6
            w, v = W[k], W[(k + 1) % 6]
            g[todo] = ((lens_unit(np.hypot(xr + w[:, 0], yr + w[:, 1]))
                        + lens_unit(np.hypot(xr + v[:, 0], yr + v[:, 1]))) / 2
                       + (xr * (w[:, 0] + v[:, 0]) + yr * (w[:, 1] + v[:, 1])) / (2 * S3) - PI / 6)
        return self.h**2 * np.maximum(g, 0.0)

    def contains(self, P):
        P = np.asarray(P, dtype=float)
        x, y = P[..., 0], P[..., 1]
        inside = np.ones(x.shape, dtype=bool)
        for V in (self.A, self.B, self.C):
            inside &= np.hypot(x - self.h * V[0], y - self.h * V[1]) <= self.h
        return inside

    def width(self, u):
        return float(self.h)               # constant width

    @property
    def perimeter(self):
        return PI * self.h                 # Barbier's theorem

    def describe(self, x):
        return "petal: 4 arcs" if self._petal(np.asarray(x) / self.h) is not None \
            else "curvy triangle: 3 arcs"

    def overlay(self):
        curves = []
        for k in range(6):
            p = np.vstack([arc(self.W[(k + 1) % 6], 1, (k - 2) * PI / 3, (k - 1) * PI / 3, 40, True),
                           arc(self.W[(k - 1) % 6], 1, (k + 1) * PI / 3, (k + 2) * PI / 3, 40, True)])
            curves.append(self.h * p)
        return curves


class Parallelogram(Shape):
    """Parallelogram spanned by u and v (square and rectangle included)."""

    def __init__(self, u, v, name, slug, overlap_name):
        self.u, self.v = np.asarray(u, float), np.asarray(v, float)
        M = np.column_stack([self.u, self.v])
        self.Minv, self.det = np.linalg.inv(M), abs(np.linalg.det(M))
        self.offset = (self.u + self.v) / 2         # the centre of M [0, 1]^2
        self.name, self.slug, self.overlap_name = name, slug, overlap_name
        if abs(self.u[1]) < 1e-12 and abs(self.v[0]) < 1e-12:
            self.method = r"closed form:  $g(x) = (a - |x_1|)_+\,(b - |x_2|)_+$   (side lengths $a$, $b$)"
        else:
            self.method = (r"closed form:  $g(x) = |\det M|\,(1-|y_1|)_+\,(1-|y_2|)_+$,   "
                           r"$y = M^{-1}x$,  $M = (u\ v)$   (affine image of the square)")

    def rings(self):
        return [centred_ccw([(0, 0), self.u, self.u + self.v, self.v])]

    def covariogram(self, P):
        Y = np.asarray(P, dtype=float) @ self.Minv.T
        return (self.det * np.clip(1 - np.abs(Y[..., 0]), 0, None)
                * np.clip(1 - np.abs(Y[..., 1]), 0, None))

    def contains(self, P):
        Y = (np.asarray(P, dtype=float) + self.offset) @ self.Minv.T
        tol = 1e-12
        return np.all((Y >= -tol) & (Y <= 1 + tol), axis=-1)

    def describe(self, x):
        return self.overlap_name


class Triangle(Shape):
    """Any triangle, as an affine image of the standard triangle."""

    def __init__(self, p0, p1, p2, name, slug):
        self.pts = centred_ccw([p0, p1, p2])
        M = np.column_stack([self.pts[1] - self.pts[0], self.pts[2] - self.pts[0]])
        self.Minv, self.det = np.linalg.inv(M), abs(np.linalg.det(M))
        self.name, self.slug = name, slug
        self.method = (r"closed form:  $g(x) = \frac{|\det M|}{2}\,\left(1 + \min(0, y_1 + y_2)"
                       r" - \max(0, y_1) - \max(0, y_2)\right)_+^2$,   $y = M^{-1}x$")

    def rings(self):
        return [self.pts]

    def covariogram(self, P):
        Y = np.asarray(P, dtype=float) @ self.Minv.T
        y1, y2 = Y[..., 0], Y[..., 1]
        s = 1 + np.minimum(0, y1 + y2) - np.maximum(0, y1) - np.maximum(0, y2)
        return self.det / 2 * np.clip(s, 0, None) ** 2

    def contains(self, P):
        Y = (np.asarray(P, dtype=float) - self.pts[0]) @ self.Minv.T
        tol = 1e-12
        return (Y[..., 0] >= -tol) & (Y[..., 1] >= -tol) & (Y.sum(axis=-1) <= 1 + tol)

    def describe(self, x):
        return "a smaller copy"


class ConvexPolygon(Shape):
    """Any convex polygon; g is computed exactly by polygon clipping."""

    closed_form = False
    NAMES = {3: "a triangle", 4: "a quadrilateral", 5: "a pentagon", 6: "a hexagon",
             7: "a heptagon", 8: "an octagon", 9: "a nonagon", 10: "a decagon"}

    def __init__(self, vertices, name, slug):
        self.pts = centred_ccw(vertices)
        self._v = [tuple(p) for p in self.pts]
        self.name, self.slug = name, slug
        self.method = "exact polygon clipping   (the covariogram of a convex polygon is piecewise quadratic)"

    def rings(self):
        return [self.pts]

    def overlap(self, x):
        """Vertices of the polygon K ∩ (K + x)."""
        clip = [(px + x[0], py + x[1]) for px, py in self._v]
        return clip_convex(self._v, clip)

    def contains(self, P):
        P = np.asarray(P, dtype=float)
        inside = np.ones(P.shape[:-1], dtype=bool)
        q = np.roll(self.pts, -1, axis=0)
        for a, b in zip(self.pts, q):
            e = b - a
            inside &= e[0] * (P[..., 1] - a[1]) - e[1] * (P[..., 0] - a[0]) >= -1e-12
        return inside

    def covariogram(self, P):
        P = np.asarray(P, dtype=float)
        flat = P.reshape(-1, 2)
        out = np.array([polygon_area(self.overlap(p)) for p in flat])
        return out.reshape(P.shape[:-1])

    def describe(self, x):
        pts = self.overlap(x)
        kept = [p for i, p in enumerate(pts)
                if np.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-9]
        n = len(kept)
        return self.NAMES.get(n, f"a {n}-gon")


def make_shapes():
    """The twelve shapes of the gallery (new instances on every call)."""
    return [
        Disk(0.5),
        Ellipse(0.65, 0.35),
        Annulus(0.5, 0.25),
        ReuleauxTriangle(1.0),
        Parallelogram((0.75, 0), (0, 0.75), "Square", "square", "a rectangle"),
        Parallelogram((1.0, 0), (0, 0.5), "Rectangle", "rectangle", "a rectangle"),
        Parallelogram((0.8, 0), (0.35, 0.6), "Parallelogram", "parallelogram", "a parallelogram"),
        Triangle((0, 0), (1, 0), (0.5, S3 / 2), "Equilateral triangle", "equilateral-triangle"),
        Triangle((0, 0), (0.9, 0), (0, 0.9), "Right triangle", "right-triangle"),
        ConvexPolygon(regular_polygon(5, 0.55, PI / 2), "Regular pentagon", "pentagon"),
        ConvexPolygon(regular_polygon(6, 0.5, 0), "Regular hexagon", "hexagon"),
        ConvexPolygon(regular_polygon(8, 0.5, PI / 8), "Regular octagon", "octagon"),
    ]


def get_shape(slug):
    """The gallery shape with the given command-line name, e.g. "reuleaux-triangle"."""
    for shape in make_shapes():
        if shape.slug == slug:
            return shape
    names = ", ".join(shape.slug for shape in make_shapes())
    raise KeyError(f"unknown shape {slug!r}; available: {names}")
