"""Geometric helpers: lens areas, disk intersections and convex polygon clipping."""

import numpy as np

PI, S3 = np.pi, np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------
def arc(center, radius, t0, t1, n=64, endpoint=False):
    """Points on a circle around `center`, angles t0 -> t1."""
    t = np.linspace(t0, t1, n, endpoint=endpoint)
    return np.column_stack([center[0] + radius * np.cos(t),
                            center[1] + radius * np.sin(t)])


def lens_unit(d):
    """Area of the intersection of two unit disks whose centres are d apart."""
    d = np.asarray(d, dtype=float)
    t = np.clip(d / 2, 0.0, 1.0)
    value = 2 * np.arccos(t) - d / 2 * np.sqrt(np.clip(4 - d * d, 0.0, None))
    return np.where(d < 2, value, 0.0)


def disk_disk_area(r1, r2, d):
    """Area of the intersection of disks with radii r1, r2 and centre distance d."""
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    inner = d <= abs(r1 - r2)
    out[inner] = PI * min(r1, r2) ** 2
    mid = ~inner & (d < r1 + r2)
    dm = d[mid]
    a1 = np.arccos(np.clip((dm**2 + r1**2 - r2**2) / (2 * dm * r1), -1, 1))
    a2 = np.arccos(np.clip((dm**2 + r2**2 - r1**2) / (2 * dm * r2), -1, 1))
    kite = 0.5 * np.sqrt(np.clip((-dm + r1 + r2) * (dm + r1 - r2)
                                 * (dm - r1 + r2) * (dm + r1 + r2), 0.0, None))
    out[mid] = r1**2 * a1 + r2**2 * a2 - kite
    return out


def clip_convex(subject, clip):
    """Sutherland-Hodgman: intersection of a polygon with a convex polygon.
    Both are lists of (x, y) in counterclockwise order."""
    out = list(subject)
    m = len(clip)
    for i in range(m):
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % m]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        if not inp:
            break
        px, py = inp[-1]
        sp = ex * (py - ay) - ey * (px - ax)
        for qx, qy in inp:
            sq = ex * (qy - ay) - ey * (qx - ax)
            if sq >= 0:
                if sp < 0:
                    t = sp / (sp - sq)
                    out.append((px + t * (qx - px), py + t * (qy - py)))
                out.append((qx, qy))
            elif sp >= 0:
                t = sp / (sp - sq)
                out.append((px + t * (qx - px), py + t * (qy - py)))
            px, py, sp = qx, qy, sq
    return out


def polygon_area(pts):
    """Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    s = 0.0
    px, py = pts[-1]
    for qx, qy in pts:
        s += px * qy - qx * py
        px, py = qx, qy
    return abs(s) / 2


def regular_polygon(n, circumradius, rotation):
    t = rotation + 2 * PI * np.arange(n) / n
    return np.column_stack([circumradius * np.cos(t), circumradius * np.sin(t)])


def centred_ccw(vertices):
    """Move a polygon's centroid to the origin and orient it counterclockwise."""
    p = np.asarray(vertices, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2
    centroid = ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)
    p = p - centroid
    return p if area > 0 else p[::-1]
