"""
Tests of covariogram_explorer.

Run all checks:        python -m pytest
Print a summary table: python tests/test_covariogram_explorer.py

The comparison with brute-force clipping needs Shapely (pip install shapely);
it is skipped automatically when Shapely is not installed.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import covariogram_explorer as ce  # noqa: E402
from covariogram_explorer.distributions import _radial_rule  # noqa: E402
from covariogram_explorer.geometry import arc  # noqa: E402

SEED = 2026
SHAPES = ce.make_shapes()


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------
def exact_area(shape):
    """Area from elementary formulas (independent of g)."""
    if isinstance(shape, ce.Disk):
        return np.pi * shape.r**2
    if isinstance(shape, ce.Ellipse):
        return np.pi * shape.a * shape.b
    if isinstance(shape, ce.Annulus):
        return np.pi * (shape.R**2 - shape.r**2)
    if isinstance(shape, ce.ReuleauxTriangle):
        return (np.pi - np.sqrt(3)) / 2 * shape.h**2
    p = shape.rings()[0]
    q = np.roll(p, -1, axis=0)
    return abs(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2


def reference_polygon(shape):
    """Independent high-resolution polygon (with hole) for brute-force areas."""
    from shapely.geometry import Polygon
    t = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    if isinstance(shape, ce.Disk):
        return Polygon(shape.r * circle)
    if isinstance(shape, ce.Ellipse):
        return Polygon(circle * [shape.a, shape.b])
    if isinstance(shape, ce.Annulus):
        return Polygon(shape.R * circle, [shape.r * circle])
    if isinstance(shape, ce.ReuleauxTriangle):
        R = ce.ReuleauxTriangle
        pts = np.vstack([arc(R.A, 1, 4 * np.pi / 3, 5 * np.pi / 3, 7000),
                         arc(R.B, 1, 0, np.pi / 3, 7000),
                         arc(R.C, 1, 2 * np.pi / 3, np.pi, 7000)])
        return Polygon(shape.h * pts)
    return Polygon(shape.rings()[0])          # polygons are represented exactly


def reuleaux_chord_cdf(r, psi):
    """Chord length distribution of the Reuleaux triangle of width 1 in the direction
    at angle psi from the nearest side direction (Gasparyan 2026, Corollary 3)."""
    c1, c2 = np.cos(psi), np.cos(np.pi / 3 - psi)
    d1, d2 = np.sqrt(1 + 2 * c1 * r + r**2), np.sqrt(1 + 2 * c2 * r + r**2)
    q1 = (r + c1) * np.sqrt(4 - d1**2) / (2 * d1)
    q2 = (r + c2) * np.sqrt(4 - d2**2) / (2 * d2)
    tail = np.where(r <= 2 * np.cos(np.pi / 3 + psi),
                    0.5 * np.sqrt(4 - r**2) + q1 - np.sqrt(3) / 2 * c1,
                    q1 + q2 - 0.5 * np.sin(np.pi / 3 + psi))
    return 1 - tail


def known_mean_distances():
    """Classical closed forms for the mean distance between two random points."""
    sq = (2 + np.sqrt(2) + 5 * np.log(1 + np.sqrt(2))) / 15          # unit square
    a, b = 1.0, 0.5                                                   # rectangle a x b
    d = np.hypot(a, b)
    rect = (a**3 / b**2 + b**3 / a**2 + d * (3 - a**2 / b**2 - b**2 / a**2)
            + 2.5 * (b**2 / a * np.log((a + d) / b) + a**2 / b * np.log((b + d) / a))) / 15
    return {"disk": 128 * 0.5 / (45 * np.pi),                        # radius 0.5
            "square": 0.75 * sq,                                      # side 0.75
            "rectangle": rect,
            "equilateral-triangle": (4 + 3 * np.log(3)) / 20}         # side 1


# ---------------------------------------------------------------------------
# checks (each returns numbers that the tests compare with bounds)
# ---------------------------------------------------------------------------
def check_brute_force(n=500):
    from shapely.affinity import translate
    rng = np.random.default_rng(SEED)
    result = {}
    for shape in SHAPES:
        poly = reference_polygon(shape)
        S = shape.support_radius()
        pts = rng.uniform(-1.05 * S, 1.05 * S, size=(n, 2))
        g = shape.covariogram(pts)
        ref = np.array([poly.intersection(translate(poly, *p)).area for p in pts])
        result[shape.slug] = float(np.max(np.abs(g - ref)))
    return result


def check_areas():
    return {s.slug: abs(s.area - exact_area(s)) for s in SHAPES}


def check_symmetry(n=500):
    rng = np.random.default_rng(SEED + 1)
    result = {}
    for shape in SHAPES:
        S = shape.support_radius()
        pts = rng.uniform(-1.05 * S, 1.05 * S, size=(n, 2))
        result[shape.slug] = float(np.max(np.abs(shape.covariogram(pts) - shape.covariogram(-pts))))
    return result


def check_integral():
    """|integral of g / area^2 - 1| (exact value 0)."""
    result = {}
    for shape in SHAPES:
        n_r, n_t = (400, 720) if shape.closed_form else (200, 360)
        result[shape.slug] = abs(ce.distance_moment(shape, 0, n_r, n_t) - 1)
    return result


def check_mean_distances():
    return {slug: (ce.mean_distance(ce.get_shape(slug)), value)
            for slug, value in known_mean_distances().items()}


def check_sampling(n=200_000):
    """Points outside K, and the largest |z| of the coordinate means (exact mean 0)."""
    rng = np.random.default_rng(SEED + 2)
    worst, outside = 0.0, 0
    for shape in SHAPES:
        P = shape.sample(n, rng)
        outside += int(np.sum(~shape.contains(P)))
        z = P.mean(axis=0) / (P.std(axis=0) / np.sqrt(n))
        worst = max(worst, float(np.max(np.abs(z))))
    return worst, outside


def check_distance_cdf(n=50_000):
    """Largest difference between the computed distribution of |X - Y| and the
    empirical distribution of simulated distances, on a grid of 41 points."""
    rng = np.random.default_rng(SEED + 3)
    result = {}
    for slug in ["disk", "annulus", "reuleaux-triangle", "right-triangle", "hexagon"]:
        shape = ce.get_shape(slug)
        d = np.sort(np.hypot(*(shape.sample(n, rng) - shape.sample(n, rng)).T))
        grid = np.linspace(0, shape.support_radius(), 41)
        F = ce.distance_cdf(shape, grid, n_r=100, n_theta=720 if shape.closed_form else 360)
        result[slug] = float(np.max(np.abs(F - np.searchsorted(d, grid, side="right") / n)))
    return result


def check_chords():
    r = np.linspace(0.01, 0.99, 50)
    disk = ce.get_shape("disk")
    exact = 1 - np.sqrt(1 - r**2)
    out = {"disk, direction 0.3": np.max(np.abs(ce.chord_length_cdf(disk, r, direction=0.3) - exact)),
           "disk, isotropic": np.max(np.abs(ce.chord_length_cdf(disk, r) - exact))}
    R = ce.get_shape("reuleaux-triangle")
    r = np.linspace(0.005, 0.995, 199)
    out["Reuleaux, 4 directions"] = max(
        np.max(np.abs(ce.chord_length_cdf(R, r, direction=np.pi / 3 + psi) - reuleaux_chord_cdf(r, psi)))
        for psi in np.radians([0, 10, 20, 30]))
    return {k: float(v) for k, v in out.items()}


def check_mean_chords():
    """|integral of (1 - F_iso) - pi area / perimeter| (Cauchy-Crofton)."""
    result = {}
    for slug, kinks, n_theta in [("disk", (), 720), ("ellipse", (), 720),
                                 ("reuleaux-triangle", (), 720),
                                 ("equilateral-triangle", (), 720), ("square", (0.75,), 11520)]:
        shape = ce.get_shape(slug)
        r, w = _radial_rule(0.0, shape.support_radius(), 200, kinks)
        F = ce.chord_length_cdf(shape, r, n_theta=n_theta)
        result[slug] = abs(np.sum(w * (1 - F)) - ce.mean_chord_length(shape))
    return result


def check_perimeters():
    a, b = 0.65, 0.35
    h = ((a - b) / (a + b)) ** 2
    ramanujan = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    R = ce.get_shape("reuleaux-triangle")
    return {"disk": abs(ce.get_shape("disk").perimeter - np.pi),
            "ellipse (Ramanujan)": abs(ce.get_shape("ellipse").perimeter - ramanujan),
            "Reuleaux (Barbier)": abs(R.perimeter - np.pi),
            "Reuleaux constant width": max(abs(R.width(t) - 1) for t in np.linspace(0, np.pi, 50))}


def check_images():
    rng = np.random.default_rng(SEED + 4)
    img = rng.random((23, 17)) < 0.4
    C = ce.covariogram_from_image(img, 1.0)[2]
    ny, nx = img.shape
    worst = 0.0
    for ky in range(-ny + 1, ny):
        for kx in range(-nx + 1, nx):
            P = np.zeros((3 * ny, 3 * nx), bool)
            Q = np.zeros((3 * ny, 3 * nx), bool)
            P[ny:2 * ny, nx:2 * nx] = img
            Q[ny + ky:2 * ny + ky, nx + kx:2 * nx + kx] = img
            worst = max(worst, abs(C[ny + ky, nx + kx] - np.sum(P & Q)))
    errors = {}
    disk = ce.get_shape("disk")
    for px in (0.02, 0.01, 0.005):
        image = ce.rasterize(disk, px)[0]
        lx, ly, Cd = ce.covariogram_from_image(image, px)
        LX, LY = np.meshgrid(lx, ly)
        errors[px] = float(np.max(np.abs(Cd - disk.covariogram(np.stack([LX, LY], -1)))))
    return worst, errors


def check_gui():
    """Drive the window with synthetic events; return a list of failures."""
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    import matplotlib.pyplot as plt

    failures = []
    extra = ce.ConvexPolygon([(0, -0.5), (0.35, 0), (0, 0.6), (-0.35, 0)], "Kite", "kite")
    app = ce.App(SHAPES + [extra])
    fig = app.fig

    def mouse(name, xy):
        fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, *xy, button=1))

    def key(k):
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, k))

    app.show_gallery()
    fig.canvas.draw()
    if len(app.thumbs) != len(SHAPES) + 1:
        failures.append("gallery does not show every shape")
    tiles = {s.slug: ax for ax, s in app.thumbs.items()}
    centre = tiles["annulus"].transAxes.transform((0.5, 0.5))
    mouse("motion_notify_event", centre)
    if app.hover is not tiles["annulus"]:
        failures.append("hover")
    mouse("button_press_event", centre)
    mouse("button_release_event", centre)
    if app.screen != "explorer" or app.shape.slug != "annulus":
        failures.append("opening a shape from the gallery")
    fig.canvas.draw()
    mouse("button_press_event", app.axL.transData.transform((0.3, -0.2)))
    if np.hypot(*(app.x - (0.3, -0.2))) > 0.01:
        failures.append("click on the map")
    mouse("motion_notify_event", app.axL.transData.transform((5.0, 0.0)))
    if abs(app.x[0] - 1.05 * app.shape.support_radius()) > 0.01:
        failures.append("clamping of the shift")
    mouse("button_release_event", app.axL.transData.transform((5.0, 0.0)))
    app.set_x((0.1, 0.1))
    mouse("button_press_event", app.axR.transData.transform((0.0, 0.0)))
    mouse("motion_notify_event", app.axR.transData.transform((0.2, -0.1)))
    mouse("button_release_event", app.axR.transData.transform((0.2, -0.1)))
    if np.hypot(*(app.x - (0.3, 0.0))) > 0.01:
        failures.append("dragging the copy")
    for k in ["0", "right", "shift+up"]:
        key(k)
    S = app.shape.support_radius()
    if np.hypot(*(app.x - (0.01 * S, 0.05 * S))) > 1e-9:
        failures.append("keyboard")
    key("b")
    if app.screen != "gallery":
        failures.append("key B")
    for shape in SHAPES + [extra]:
        app.show_explorer(shape)
        fig.canvas.draw()
        for x in [(0.0, 0.0), (0.2, 0.1), (-0.4, 0.3), (3.0, 3.0)]:
            app.set_x(x)
        centre = app.back_ax.transAxes.transform((0.5, 0.5))
        mouse("button_press_event", centre)
        mouse("button_release_event", centre)
        if app.screen != "gallery":
            failures.append(f"back button ({shape.slug})")
    plt.close(fig)
    return failures


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def test_brute_force():
    import pytest
    pytest.importorskip("shapely")
    for slug, err in check_brute_force().items():
        assert err < 1e-6, slug


def test_areas():
    for slug, err in check_areas().items():
        assert err < 1e-12, slug


def test_symmetry():
    for slug, err in check_symmetry().items():
        assert err < 1e-12, slug


def test_integral_of_g():
    for slug, err in check_integral().items():
        assert err < 1e-4, slug


def test_mean_distances():
    for slug, (computed, known) in check_mean_distances().items():
        assert abs(computed - known) < 1e-7, slug


def test_sampling():
    worst_z, outside = check_sampling()
    assert outside == 0
    assert worst_z < 5


def test_distance_cdf_against_simulation():
    for slug, dist in check_distance_cdf().items():
        assert dist < 1.63 / np.sqrt(50_000), slug          # 1% level of the KS test


def test_chord_lengths():
    for name, err in check_chords().items():
        assert err < 1e-8, name


def test_mean_chord_length():
    for slug, err in check_mean_chords().items():
        assert err < 1e-5, slug


def test_perimeters_and_widths():
    for name, err in check_perimeters().items():
        assert err < 1e-8, name


def test_image_covariogram():
    worst, errors = check_images()
    assert worst == 0
    assert errors[0.005] < errors[0.01] < errors[0.02] < 1e-2


def test_gui():
    assert check_gui() == []


def test_command_line(tmp_path):
    out = tmp_path / "fig.png"
    run = subprocess.run([sys.executable, "-m", "covariogram_explorer", "--shape", "hexagon",
                          "--save", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Validation summary (seed {SEED})\n")
    try:
        brute = check_brute_force()
    except ImportError:
        brute = None
    areas, sym, integral = check_areas(), check_symmetry(), check_integral()
    print(f"{'shape':22s} {'vs clipping':>12s} {'g(0)-area':>10s} {'g(-x)-g(x)':>11s} {'int g/A^2-1':>12s}")
    for shape in SHAPES:
        b = f"{brute[shape.slug]:.1e}" if brute else "skipped"
        print(f"{shape.slug:22s} {b:>12s} {areas[shape.slug]:10.1e} {sym[shape.slug]:11.1e} "
              f"{integral[shape.slug]:12.1e}")
    print("\nmean distance between two random points: computed vs closed form")
    for slug, (computed, known) in check_mean_distances().items():
        print(f"  {slug:22s} {computed:.10f}  {known:.10f}  diff {computed - known:+.1e}")
    z, outside = check_sampling()
    print(f"\nsampling: points outside K = {outside}, largest |z| of the coordinate means = {z:.2f}")
    print("distance distribution vs simulation (max |F - ECDF|, n = 50000; 1% bound 0.0073):")
    for slug, dist in check_distance_cdf().items():
        print(f"  {slug:22s} {dist:.4f}")
    print("chord lengths vs exact formulas (max abs difference):")
    for name, err in check_chords().items():
        print(f"  {name:22s} {err:.1e}")
    print("mean chord length vs pi area / perimeter:")
    for slug, err in check_mean_chords().items():
        print(f"  {slug:22s} {err:.1e}")
    print("perimeters and widths:")
    for name, err in check_perimeters().items():
        print(f"  {name:24s} {err:.1e}")
    worst, errors = check_images()
    print(f"image covariogram: FFT vs pixel counting, max difference = {worst:.0f}")
    print("  digitised disk, max |C - g|: " + ", ".join(f"pixel {k}: {v:.1e}" for k, v in errors.items()))
    failures = check_gui()
    print("\nGUI checks:", "all passed" if not failures else failures)
