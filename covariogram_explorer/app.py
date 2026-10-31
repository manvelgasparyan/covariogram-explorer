"""Interactive application: a gallery of shapes and the covariogram explorer."""

import argparse
import sys

import numpy as np
import matplotlib
if getattr(sys, "frozen", False):      # packaged as an .exe: use the Tk window
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from . import __version__
from .shapes import make_shapes

PURPLE, TEAL, GRAY = "#7F77DD", "#1D9E75", "#888780"
CARD, TILE, TILE_HOVER = "#F1EFE8", "#F7F6F2", "#E9E6DC"


# ---------------------------------------------------------------------------
# The application: start screen (gallery) and explorer in one window
# ---------------------------------------------------------------------------
class App:
    def __init__(self, shapes):
        self.shapes = shapes
        self.fig = plt.figure(figsize=(12, 7.6))
        try:
            self.fig.canvas.manager.set_window_title("Covariogram explorer")
        except AttributeError:
            pass
        self.screen = None
        self.shape = None
        self.x = np.zeros(2)
        self.drag = None
        self.thumbs = {}
        self.hover = None
        self.back_ax = self.axL = self.axR = None
        self.maps = {}
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_move)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("key_press_event", self.on_key)
        self.fig._covariogram_app = self      # keeps the callbacks alive (e.g. in Jupyter)

    # ---- screens --------------------------------------------------------------
    def _new_screen(self, screen):
        self.screen, self.drag, self.hover = screen, None, None
        self.fig.clear()
        toolbar = getattr(getattr(self.fig.canvas, "manager", None), "toolbar", None)
        if toolbar is not None and hasattr(toolbar, "update"):
            toolbar.update()                  # forget zoom history of the old screen

    def show_gallery(self):
        self._new_screen("gallery")
        fig = self.fig
        fig.suptitle("Choose a shape", fontsize=18, y=0.965)
        fig.text(0.5, 0.905, "Click a shape to explore its covariogram   "
                 r"$g(x) = \mathrm{area}\,(K \cap (K + x))$",
                 ha="center", fontsize=12, color="0.35")
        self.thumbs = {}
        n = len(self.shapes)
        cols = 4 if n <= 16 else 5
        rows = -(-n // cols)
        left, right, top, bottom = 0.05, 0.95, 0.86, 0.11
        cw, rh = (right - left) / cols, (top - bottom) / rows
        for i, shape in enumerate(self.shapes):
            r, c = divmod(i, cols)
            ax = fig.add_axes([left + c * cw + 0.012, top - (r + 1) * rh + 0.012,
                               cw - 0.024, rh - 0.055])
            ax.set_facecolor(TILE)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            color = PURPLE if shape.closed_form else TEAL
            ax.add_patch(PathPatch(shape.path(), fc=to_rgba(color, 0.55), ec=color, lw=1.5))
            ring = shape.rings()[0]
            (x0, y0), (x1, y1) = ring.min(axis=0), ring.max(axis=0)
            size = max(x1 - x0, y1 - y0)
            self._fit_limits(ax, x0 - 0.18 * size, x1 + 0.18 * size,
                             y0 - 0.36 * size, y1 + 0.12 * size)
            ax.set_title(shape.name, fontsize=12.5, pad=5)
            ax.text(0.5, 0.05, "closed form" if shape.closed_form else "exact clipping",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="0.45")
            self.thumbs[ax] = shape
        fig.legend(handles=[Patch(fc=PURPLE, label="closed-form formula"),
                            Patch(fc=TEAL, label="exact polygon clipping")],
                   loc="center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, fontsize=11)
        fig.canvas.draw_idle()

    @staticmethod
    def _fit_limits(ax, x0, x1, y0, y1):
        """Set limits containing the box [x0, x1] x [y0, y1] with equal scales."""
        fig_w, fig_h = ax.figure.get_size_inches()
        pos = ax.get_position()
        ratio = (pos.width * fig_w) / (pos.height * fig_h)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if w / h < ratio:
            w = h * ratio
        else:
            h = w / ratio
        ax.set_xlim(cx - w / 2, cx + w / 2)
        ax.set_ylim(cy - h / 2, cy + h / 2)
        ax.set_aspect("equal", adjustable="box")

    def covariogram_map(self, shape):
        """g on a grid over the support (cached; uses g(-x) = g(x))."""
        if shape.slug not in self.maps:
            S = 1.02 * shape.support_radius()
            n = 161 if shape.closed_form else 121
            s = np.linspace(-S, S, n)
            P = np.stack(np.meshgrid(s, s), axis=-1)
            if shape.closed_form:
                G = shape.covariogram(P)
            else:
                half = n // 2 + 1
                G = np.zeros((n, n))
                G[:half] = shape.covariogram(P[:half])
                G[half:] = G[:n - half][::-1, ::-1]
            self.maps[shape.slug] = (s, np.maximum(G, 0.0))
        return self.maps[shape.slug]

    def show_explorer(self, shape, x0=None):
        self._new_screen("explorer")
        self.shape = shape
        S = shape.support_radius()
        fig = self.fig
        fig.suptitle(f"Covariogram of the {shape.noun}", fontsize=17, y=0.965)
        fig.text(0.5, 0.912, shape.method, ha="center", fontsize=10.5, color="0.35")

        self.back_ax = fig.add_axes([0.012, 0.935, 0.12, 0.045])
        self.back_ax.set_facecolor(CARD)
        self.back_ax.set_xticks([])
        self.back_ax.set_yticks([])
        for spine in self.back_ax.spines.values():
            spine.set_visible(False)
        self.back_ax.text(0.5, 0.5, "\u2190  All shapes", ha="center", va="center", fontsize=11)
        self.back_ax.set_navigate(False)

        # left: the map of g ------------------------------------------------------
        self.axL = axL = fig.add_axes([0.03, 0.2, 0.45, 0.66])
        s, G = self.covariogram_map(shape)
        step = s[1] - s[0]
        ext = (s[0] - step / 2, s[-1] + step / 2, s[0] - step / 2, s[-1] + step / 2)
        outline = shape.support_outline()
        outline = np.vstack([outline, outline[:1]])
        im = axL.imshow(G, extent=ext, origin="lower", cmap="viridis",
                        interpolation="bilinear", vmin=0, vmax=shape.area)
        im.set_clip_path(PathPatch(Path(outline), transform=axL.transData))
        axL.contour(s, s, G, levels=np.linspace(0, shape.area, 9)[1:-1],
                    colors="white", linewidths=0.6, alpha=0.55)
        axL.plot(outline[:, 0], outline[:, 1], color="0.25", lw=1)
        for curve in shape.overlay():
            axL.plot(curve[:, 0], curve[:, 1], color="white", lw=0.9, alpha=0.9)
        cbar = fig.colorbar(im, ax=axL, fraction=0.035, pad=0.02)
        cbar.set_label("g(x)")
        lim = 1.05 * S
        axL.set_xlim(-lim, lim)
        axL.set_ylim(-lim, lim)
        axL.set_aspect("equal")
        axL.axis("off")
        axL.set_title("Map of g over its support: click or drag to choose x",
                      fontsize=11, color="0.35")
        self.armL, = axL.plot([0, 0], [0, 0], color="white", lw=1.5, zorder=4)
        axL.plot([0], [0], "o", color="white", ms=3.5, zorder=4)
        self.dot, = axL.plot([0], [0], "o", ms=13, mfc="white", mec="0.1", mew=2, zorder=5)

        # right: K, K + x and the overlap ------------------------------------------
        self.axR = axR = fig.add_axes([0.53, 0.2, 0.44, 0.66])
        path = shape.path()
        self.K_fill = PathPatch(path, fc=to_rgba(GRAY, 0.14), ec="none", zorder=1)
        axR.add_patch(self.K_fill)
        self.overlap = PathPatch(path, fc=to_rgba(PURPLE, 0.55), ec="none", zorder=2)
        axR.add_patch(self.overlap)
        self.overlap.set_clip_path(self.K_fill)          # the copy, seen only inside K
        axR.add_patch(PathPatch(path, fc="none", ec="0.1", lw=1.6, zorder=3))
        self.copy = PathPatch(path, fc="none", ec="0.35", lw=1.6, ls=(0, (5, 4)), zorder=3)
        axR.add_patch(self.copy)
        self.armR, = axR.plot([0, 0], [0, 0], color="0.45", lw=1, zorder=4)
        lim = shape.extent() + 0.6 * S
        axR.set_xlim(-lim, lim)
        axR.set_ylim(-lim, lim)
        axR.set_aspect("equal")
        axR.axis("off")
        axR.set_title("K (solid), copy K + x (dashed), overlap shaded", fontsize=11, color="0.35")

        # bottom: read-out cards ------------------------------------------------------
        self.cards = []
        labels = ("Overlap area g(x)", "Share of full area", "Shift length |x|", "Shape of the overlap")
        for i, label in enumerate(labels):
            card = fig.add_axes([0.05 + 0.23 * i, 0.03, 0.205, 0.11])
            card.set_facecolor(CARD)
            card.set_xticks([])
            card.set_yticks([])
            for spine in card.spines.values():
                spine.set_visible(False)
            card.set_navigate(False)
            card.text(0.06, 0.72, label, transform=card.transAxes, fontsize=10,
                      color="0.4", va="center")
            self.cards.append(card.text(0.06, 0.32, "", transform=card.transAxes,
                                        fontsize=17 if i < 3 else 13, va="center"))

        if x0 is None:
            x0 = 0.35 * S * np.array([np.cos(np.radians(55)), np.sin(np.radians(55))])
        self.set_x(x0)

    # ---- interaction -------------------------------------------------------------
    def _toolbar_active(self):
        toolbar = getattr(getattr(self.fig.canvas, "manager", None), "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    @staticmethod
    def _data_xy(ax, event):
        return ax.transData.inverted().transform((event.x, event.y))

    def on_press(self, event):
        if event.button != 1 or self._toolbar_active():
            return
        if self.screen == "gallery":
            shape = self.thumbs.get(event.inaxes)
            if shape is not None:
                self.show_explorer(shape)
            return
        if event.inaxes is self.back_ax:
            self.show_gallery()
        elif event.inaxes is self.axL:
            self.drag = ("L", None, None)
            self.set_x(self._data_xy(self.axL, event))
        elif event.inaxes is self.axR:
            self.drag = ("R", self._data_xy(self.axR, event), self.x.copy())

    def on_move(self, event):
        if self.screen == "gallery":
            over = event.inaxes if event.inaxes in self.thumbs else None
            if over is not self.hover:
                if self.hover is not None:
                    self.hover.set_facecolor(TILE)
                if over is not None:
                    over.set_facecolor(TILE_HOVER)
                self.hover = over
                self.fig.canvas.draw_idle()
            return
        if self.screen != "explorer":
            return
        over_back = event.inaxes is self.back_ax
        if over_back != (self.hover is self.back_ax):
            self.back_ax.set_facecolor(TILE_HOVER if over_back else CARD)
            self.hover = self.back_ax if over_back else None
            self.fig.canvas.draw_idle()
        if self.drag is None:
            return
        mode, start, x_start = self.drag
        if mode == "L":
            self.set_x(self._data_xy(self.axL, event))
        else:
            self.set_x(x_start + self._data_xy(self.axR, event) - start)

    def on_release(self, event):
        self.drag = None

    def on_key(self, event):
        if self.screen != "explorer":
            return
        key = event.key or ""
        if key in ("escape", "b"):
            self.show_gallery()
            return
        step = (0.05 if key.startswith("shift+") else 0.01) * self.shape.support_radius()
        key = key.replace("shift+", "")
        moves = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}
        if key in moves:
            self.set_x(self.x + step * np.array(moves[key]))
        elif key == "0":
            self.set_x((0.0, 0.0))

    # ---- drawing -------------------------------------------------------------------
    def set_x(self, x):
        lim = 1.05 * self.shape.support_radius()
        self.x = np.clip(np.asarray(x, dtype=float), -lim, lim)
        self.redraw()

    def redraw(self):
        x, shape = self.x, self.shape
        g = shape.value(x)
        d = float(np.hypot(*x))
        self.dot.set_data([x[0]], [x[1]])
        self.armL.set_data([0, x[0]], [0, x[1]])
        shift = Affine2D().translate(x[0], x[1]) + self.axR.transData
        self.overlap.set_transform(shift)
        self.copy.set_transform(shift)
        self.armR.set_data([0, x[0]], [0, x[1]])
        if g <= 1e-12:
            info = "no overlap"
        elif d < 1e-12:
            info = "the whole shape"
        else:
            info = shape.describe(x)
        self.cards[0].set_text(f"{g:.4f}")
        self.cards[1].set_text(f"{100 * g / shape.area:.0f}%")
        self.cards[2].set_text(f"{d:.3f}")
        self.cards[3].set_text(info)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Explore the covariogram of planar shapes.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--list", action="store_true", help="list the shapes and exit")
    parser.add_argument("--shape", metavar="NAME", help="open this shape directly (see --list)")
    parser.add_argument("--x", nargs=2, type=float, metavar=("X1", "X2"), help="starting shift")
    parser.add_argument("--save", metavar="FILE",
                        help="save a picture (start screen, or the shape given by --shape)")
    args, _ = parser.parse_known_args()   # tolerant of extra arguments, e.g. in Jupyter

    shapes = make_shapes()
    if args.list:
        for shape in shapes:
            how = "closed form" if shape.closed_form else "exact polygon clipping"
            print(f"{shape.slug:22s} {shape.name} ({how})")
        return
    by_slug = {shape.slug: shape for shape in shapes}
    if args.shape and args.shape not in by_slug:
        parser.error(f"unknown shape {args.shape!r}; use --list to see the names")

    if args.save:
        plt.switch_backend("Agg")
    else:
        # free the arrow keys (view history) and the log-scale toggles
        rc = plt.rcParams
        rc["keymap.back"] = [k for k in rc["keymap.back"] if k != "left"]
        rc["keymap.forward"] = [k for k in rc["keymap.forward"] if k != "right"]
        rc["keymap.xscale"] = []
        rc["keymap.yscale"] = []

    app = App(shapes)
    if args.shape:
        app.show_explorer(by_slug[args.shape], args.x)
    else:
        app.show_gallery()

    if args.save:
        app.fig.savefig(args.save, dpi=150)
        if args.shape:
            print(f"saved {args.save}   g(x) = {app.shape.value(app.x):.6f}")
        else:
            print(f"saved {args.save}")
    else:
        plt.show()
