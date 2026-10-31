"""Replication script for

    Gasparyan, M. "covariogram-explorer: Exact Covariograms, Random Distances, and
    Chord Lengths of Planar Shapes in Python."

It reproduces all code examples (in order), all figures and the table of mean
distances. Run it from this directory:

    python replication.py

Requirements: covariogram-explorer 1.1.0, NumPy and Matplotlib. Install the package with
    pip install git+https://github.com/manvelgasparyan/covariogram-explorer
or with "pip install ." in the repository. Figures are written to ./figures.
"""
import os
import matplotlib
matplotlib.use("Agg")
os.makedirs("figures", exist_ok=True)

# --- Code chunk 1 ------------------------------------------------------------
import numpy as np
import covariogram_explorer as ce
np.set_printoptions(precision=4, suppress=True)
for s in ce.make_shapes():
    how = "closed form" if s.closed_form else "polygon clipping"
    print(f"{s.slug:21s} area {s.area:.4f}  {how}")

# --- Code chunk 2 ------------------------------------------------------------
R = ce.get_shape("reuleaux-triangle")
print(R.area, R.perimeter, R.width(0.3))
x = np.array([0.2, 0.45])
print(R.value(x), R.describe(x))
P = np.array([[0, 0], [0.5, 0], [0, 0.5], [1, 0]])
print(R.covariogram(P))

# --- Code chunk 3 ------------------------------------------------------------
R2 = ce.ReuleauxTriangle(width=2.0)
print(R2.value(2 * x), 4 * R.value(x))
E, D = ce.Ellipse(a=2.0, b=1.0), ce.Disk(radius=1.0)
y = np.array([0.6, 0.3])
print(E.value(y), 2 * D.value(y / [2, 1]))
T = ce.Triangle((0, 0), (1, 0), (0.3, 0.8), "Scalene triangle", "scalene")
print(T.area, T.value([0.1, 0.1]), T.describe([0.1, 0.1]))

# --- Code chunk 4 ------------------------------------------------------------
rng = np.random.default_rng(1)
X = R.sample(100_000, rng)
print(X.shape, R.contains(X).all())
print(R.area * R.contains(X + x).mean(), R.value(x))

# --- Code chunk 5 ------------------------------------------------------------
Y = R.sample(100_000, rng)
d = np.hypot(*(X - Y).T)
s = np.array([0.2, 0.4, 0.6, 0.8])
print(ce.distance_cdf(R, s))
print(np.array([np.mean(d <= t) for t in s]))
print(ce.distance_density(R, s))
print(ce.mean_distance(R), d.mean())

# --- Code chunk 6 ------------------------------------------------------------
disk = ce.get_shape("disk")
print(ce.mean_distance(disk), 64 / (45 * np.pi))
tri = ce.get_shape("equilateral-triangle")
print(ce.mean_distance(tri), (4 + 3 * np.log(3)) / 20)

# --- Code chunk 7 ------------------------------------------------------------
r = np.array([0.2, 0.5, 0.8, 0.95])
print(ce.chord_length_cdf(R, r, direction=np.pi / 3))
print(ce.chord_length_cdf(R, r, direction=np.pi / 2))
print(ce.chord_length_cdf(R, r))
print(ce.mean_chord_length(R), R.area / R.width(0))

# --- Code chunk 8 ------------------------------------------------------------
image, xs, ys = ce.rasterize(R, 0.01)
print(image.shape, image.sum() * 0.01**2)
lags_x, lags_y, C = ce.covariogram_from_image(image, 0.01)
LX, LY = np.meshgrid(lags_x, lags_y)
G = R.covariogram(np.stack([LX, LY], axis=-1))
print(np.abs(C - G).max())

# --- Code chunk 9 ------------------------------------------------------------
kite = ce.ConvexPolygon([(0, -0.5), (0.35, 0), (0, 0.6), (-0.35, 0)],
                        "Kite", "kite")
print(round(kite.area, 6), kite.describe([0.1, 0.2]))
print(kite.value([0.1, 0.2]))
class UnitSquare(ce.Shape):
    name, slug, method = "Unit square", "unit-square", "closed form"
    def rings(self):
        return [np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])]
    def covariogram(self, P):
        P = np.asarray(P, dtype=float)
        return np.prod(np.clip(1 - np.abs(P), 0, None), axis=-1)
    def contains(self, P):
        return np.all(np.abs(np.asarray(P, dtype=float)) <= 0.5, axis=-1)

sq = UnitSquare()
print(round(sq.value([0.3, 0.2]), 12), sq.perimeter)
print(ce.distance_cdf(sq, 1.0))

# --- Code chunk 10 ------------------------------------------------------------
app = ce.App(ce.make_shapes() + [kite])
app.show_explorer(kite, (0.15, 0.1))
app.fig.savefig("figures/explorer-kite.pdf")

# ==============================================================================
# Figures and tables of the paper
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from covariogram_explorer.app import PURPLE

shapes = {s.slug: s for s in ce.make_shapes()}

# Figures 1 and 2: the two screens of the application
app = ce.App(ce.make_shapes())
app.show_gallery()
app.fig.savefig("figures/fig-gallery.pdf")
app.show_explorer(shapes["annulus"], (0.3, 0.25))
app.fig.savefig("figures/fig-explorer.pdf")
plt.close(app.fig)

# Figure 3: covariogram maps of six shapes
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, slug in zip(axes.flat, ["disk", "annulus", "reuleaux-triangle",
                                "rectangle", "right-triangle", "hexagon"]):
    shape = shapes[slug]
    S = 1.02 * shape.support_radius()
    grid = np.linspace(-S, S, 241 if shape.closed_form else 161)
    Gmap = shape.covariogram(np.stack(np.meshgrid(grid, grid), axis=-1)) / shape.area
    outline = shape.support_outline()
    outline = np.vstack([outline, outline[:1]])
    im = ax.imshow(Gmap, extent=(-S, S, -S, S), origin="lower", cmap="viridis",
                   vmin=0, vmax=1, interpolation="bilinear")
    im.set_clip_path(PathPatch(MplPath(outline), transform=ax.transData))
    ax.contour(grid, grid, Gmap, levels=np.linspace(0, 1, 9)[1:-1], colors="white",
               linewidths=0.5, alpha=0.6)
    ax.plot(outline[:, 0], outline[:, 1], color="0.2", lw=0.8)
    ax.add_patch(PathPatch(shape.path(), fc="none", ec="white", lw=1.0, ls=(0, (3, 2))))
    ax.set_xlim(-S, S)
    ax.set_ylim(-S, S)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(shape.name, fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.8, label=r"$g(x)\,/\,\lambda_2(K)$")
fig.savefig("figures/fig-maps.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 4: distances between two random points
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for slug, ls in [("disk", "-"), ("reuleaux-triangle", "-"), ("square", "--"),
                 ("equilateral-triangle", "--"), ("rectangle", "-."), ("annulus", ":")]:
    shape = shapes[slug]
    A = shape.area
    t = np.linspace(0, shape.support_radius() / np.sqrt(A), 400)
    ax1.plot(t, np.sqrt(A) * ce.distance_density(shape, np.sqrt(A) * t), ls, lw=1.6,
             label=shape.name)
ax1.set_xlabel(r"distance (shapes scaled to area 1)")
ax1.set_ylabel("probability density")
ax1.set_title("(a) density of $|X-Y|$", fontsize=11)
ax1.set_xlim(left=0)
ax1.set_ylim(0, 1.9)
ax1.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
t = np.linspace(0, 1, 300)
ax2.hist(d, bins=60, density=True, color=PURPLE, alpha=0.45, label="simulation, $10^5$ pairs")
ax2.plot(t, ce.distance_density(R, t), "k", lw=1.5, label="from the covariogram")
ax2.set_xlabel("distance")
ax2.set_title("(b) Reuleaux triangle of width 1", fontsize=11)
ax2.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig-distances.pdf")
plt.close(fig)

# Figure 5: chord length distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
t = np.linspace(0.001, 0.999, 300)
for deg, ls in [(0, "-"), (10, "--"), (20, "-."), (30, ":")]:
    ax1.plot(t, ce.chord_length_cdf(R, t, direction=np.pi / 3 + np.radians(deg)), ls,
             lw=1.6, label=rf"$\psi = {deg}^\circ$")
ax1.set_xlabel("chord length $r$")
ax1.set_ylabel(r"$F_R(u, r)$")
ax1.set_title("(a) Reuleaux triangle, chords parallel to $u$", fontsize=11)
ax1.legend(fontsize=9, frameon=False)
for slug, ls in [("disk", "-"), ("reuleaux-triangle", "-"), ("square", "--"),
                 ("equilateral-triangle", "--"), ("hexagon", "-.")]:
    shape = shapes[slug]
    m = ce.mean_chord_length(shape)
    rr = np.linspace(0.001, shape.support_radius(), 300)
    ax2.plot(rr / m, ce.chord_length_cdf(shape, rr, n_theta=720 if shape.closed_form else 360),
             ls, lw=1.6, label=shape.name)
ax2.set_xlabel("chord length / mean chord length")
ax2.set_ylabel("distribution function")
ax2.set_title("(b) isotropic random lines", fontsize=11)
ax2.legend(fontsize=9, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/fig-chords.pdf")
plt.close(fig)

# Figure 6: covariograms of digitised shapes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
pixels = np.array([0.04, 0.02, 0.01, 0.005, 0.0025])
for slug, marker in [("disk", "o"), ("reuleaux-triangle", "s"), ("hexagon", "^"),
                     ("annulus", "D")]:
    shape = shapes[slug]
    errors = []
    for px in pixels:
        img = ce.rasterize(shape, px)[0]
        lx, ly, Cimg = ce.covariogram_from_image(img, px)
        GX, GY = np.meshgrid(lx, ly)
        errors.append(np.abs(Cimg - shape.covariogram(np.stack([GX, GY], -1))).max() / shape.area)
    ax1.loglog(pixels, errors, marker + "-", mfc="none", label=shape.name)
ax1.loglog(pixels, 0.5 * pixels, "k:", lw=1, label="slope 1")
ax1.set_xlabel("pixel size")
ax1.set_ylabel(r"$\max_x |\hat g(x) - g(x)|\,/\,\lambda_2(K)$")
ax1.set_title("(a) error of the digitised covariogram", fontsize=11)
ax1.legend(fontsize=9, frameon=False)
img = ce.rasterize(R, 0.02)[0]
lx, ly, Cimg = ce.covariogram_from_image(img, 0.02)
GX, GY = np.meshgrid(lx, ly)
err = Cimg - R.covariogram(np.stack([GX, GY], -1))
lim = np.abs(err).max()
im = ax2.imshow(err, extent=(lx[0], lx[-1], ly[0], ly[-1]), origin="lower", cmap="RdBu_r",
                vmin=-lim, vmax=lim)
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(-1.05, 1.05)
ax2.set_aspect("equal")
ax2.set_title(r"(b) $\hat g - g$, Reuleaux triangle, pixel 0.02", fontsize=11)
fig.colorbar(im, ax=ax2, shrink=0.85)
fig.tight_layout()
fig.savefig("figures/fig-images.pdf")
plt.close(fig)

# Table: mean distances and mean chord lengths
print("\nTable: mean distance and mean chord length (sizes of the gallery)")
print(f"{'shape':22s} {'area':>8s} {'perimeter':>9s} {'E|X-Y|':>9s} {'/sqrt(A)':>9s} {'chord':>8s}")
for shape in ce.make_shapes():
    if shape.closed_form:
        m = ce.mean_distance(shape)
        chord = f"{ce.mean_chord_length(shape):8.6f}" if shape.convex else "       -"
        print(f"{shape.name:22s} {shape.area:8.6f} {shape.perimeter:9.6f} {m:9.6f} "
              f"{m / np.sqrt(shape.area):9.6f} {chord}")
