"""
Core routines for the Sierpinski octahedron (octahedron flake).

The fractal is the attractor of six maps  f_v(p) = (p + v) / 2,  v = ±e_x, ±e_y, ±e_z.
Everything here works in units where the octahedron is |x| + |y| + |z| <= 1.

  chaos_game(...)   fast chaos game (numba), optionally with a restriction rule
                    on consecutive vertices and an arbitrary contraction ratio
  splat(...)        projects points into an image: additive density + z-buffer
  inside(...)       exact membership test for the level-n approximation
                    (used to draw crisp slices at pixel resolution)
  estimate(...)     distance estimate for ray marching the solid fractal
"""
import numpy as np
from numba import njit, prange

VERTS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float64)

# vertex i and i ^ 1 are antipodal; all other pairs are adjacent (share an edge)
RULES = {
    "classic":      (True, True, True),    # (same, adjacent, opposite) allowed
    "no-repeat":    (False, True, True),
    "no-opposite":  (True, True, False),
    "adjacent":     (False, True, False),
}


def rule_matrix(same=True, adjacent=True, opposite=True):
    """6x6 matrix: allowed[i, j] is True if vertex j may follow vertex i."""
    allowed = np.zeros((6, 6), dtype=np.bool_)
    for i in range(6):
        for j in range(6):
            if i == j:
                allowed[i, j] = same
            elif i ^ 1 == j:
                allowed[i, j] = opposite
            else:
                allowed[i, j] = adjacent
    return allowed


CYCLE = [0, 2, 4, 1, 3, 5]      # +x, +y, +z, -x, -y, -z


def cyclic_rule(*forbidden_steps):
    """Chiral rules: forbid jumping k steps forward along the cycle +x,+y,+z,-x,-y,-z."""
    pos = {v: i for i, v in enumerate(CYCLE)}
    allowed = np.ones((6, 6), dtype=np.bool_)
    for i in range(6):
        for k in forbidden_steps:
            allowed[i, CYCLE[(pos[i] + k) % 6]] = False
    return allowed


def rule_dimension(allowed, ratio=0.5):
    """Similarity dimension log(lambda) / log(1/r), lambda = spectral radius of the rule."""
    lam = max(abs(np.linalg.eigvals(allowed.astype(float))))
    return np.log(lam) / np.log(1.0 / ratio)


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _chaos(n, allowed, ratio, warmup):
    succ = np.zeros((6, 6), dtype=np.int64)
    nsucc = np.zeros(6, dtype=np.int64)
    for i in range(6):
        for j in range(6):
            if allowed[i, j]:
                succ[i, nsucc[i]] = j
                nsucc[i] += 1
    pts = np.empty((n, 3), dtype=np.float32)
    code = np.empty(n, dtype=np.uint8)
    x = y = z = 0.0
    prev = np.random.randint(0, 6)
    prev2 = prev
    for k in range(n + warmup):
        v = succ[prev, np.random.randint(0, nsucc[prev])]
        axis = v >> 1
        s = 1.0 - 2.0 * (v & 1)
        # p <- p + (1 - r) * (vertex - p)
        x += (1.0 - ratio) * ((s if axis == 0 else 0.0) - x)
        y += (1.0 - ratio) * ((s if axis == 1 else 0.0) - y)
        z += (1.0 - ratio) * ((s if axis == 2 else 0.0) - z)
        prev2 = prev
        prev = v
        if k >= warmup:
            i = k - warmup
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z
            # last vertex = top-level sub-octahedron, previous one = level 2
            code[i] = 6 * v + prev2
    return pts, code


def chaos_game(n, rule="classic", ratio=0.5, seed=0, warmup=64):
    """Returns (points float32 (n, 3), code uint8 (n,)); code = 6*level1 + level2 address."""
    allowed = rule_matrix(*RULES[rule]) if isinstance(rule, str) else rule
    _seed(seed)
    return _chaos(n, allowed, ratio, warmup)


@njit(cache=True)
def splat(pts, rgb, rot, scale, width, height, dens, col, zbuf, zcol):
    """Orthographic projection after rotation `rot` (3x3).
    dens/col accumulate additively (x-ray view), zbuf/zcol keep the nearest point."""
    cx = width * 0.5
    cy = height * 0.5
    for i in range(pts.shape[0]):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]
        u = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z
        v = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z
        w = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z
        px = int(cx + u * scale)
        py = int(cy - v * scale)
        if px < 0 or py < 0 or px >= width or py >= height:
            continue
        dens[py, px] += 1.0
        for c in range(3):
            col[py, px, c] += rgb[i, c]
        if w > zbuf[py, px]:
            zbuf[py, px] = w
            for c in range(3):
                zcol[py, px, c] = rgb[i, c]


@njit(cache=True)
def inside(x, y, z, levels):
    """Exact test for the level-`levels` approximation.
    Returns (k, top): k = 0 outside the octahedron, k >= 1 removed as a hole of level k,
    k = -1 survives all levels; top = nearest top-level vertex (sub-octahedron)."""
    top = -1
    for lev in range(levels):
        ax = abs(x)
        ay = abs(y)
        az = abs(z)
        if ax + ay + az > 1.0:
            return lev, top
        # the six half-size octahedra are exactly {|p_axis| >= sum of the other two}
        if ax >= ay and ax >= az:
            v = 0 if x >= 0 else 1
            gap = ax - ay - az
        elif ay >= az:
            v = 2 if y >= 0 else 3
            gap = ay - ax - az
        else:
            v = 4 if z >= 0 else 5
            gap = az - ax - ay
        if top < 0:
            top = v
        if gap < 0.0:
            return lev + 1, top
        x *= 2.0
        y *= 2.0
        z *= 2.0
        if v == 0:
            x -= 1.0
        elif v == 1:
            x += 1.0
        elif v == 2:
            y -= 1.0
        elif v == 3:
            y += 1.0
        elif v == 4:
            z -= 1.0
        else:
            z += 1.0
    return -1, top


@njit(parallel=True, cache=True)
def slice_image(origin, du, dv, width, height, levels):
    """Evaluates `inside` on the plane origin + i*du + j*dv."""
    lev = np.empty((height, width), dtype=np.int16)
    top = np.empty((height, width), dtype=np.int8)
    for j in prange(height):
        for i in range(width):
            px = origin[0] + i * du[0] + j * dv[0]
            py = origin[1] + i * du[1] + j * dv[1]
            pz = origin[2] + i * du[2] + j * dv[2]
            a, b = inside(px, py, pz, levels)
            lev[j, i] = a
            top[j, i] = b
    return lev, top


@njit(cache=True)
def estimate(x, y, z, levels):
    """Distance estimate to the level-`levels` solid, plus the level-1 and level-2 address."""
    scale = 1.0
    a1 = -1
    a2 = -1
    for lev in range(levels):
        ax = abs(x)
        ay = abs(y)
        az = abs(z)
        if ax + ay + az > 3.0:
            break
        if ax >= ay and ax >= az:
            v = 0 if x >= 0 else 1
            x = 2.0 * x - (1.0 if x >= 0 else -1.0)
            y *= 2.0
            z *= 2.0
        elif ay >= az:
            v = 2 if y >= 0 else 3
            y = 2.0 * y - (1.0 if y >= 0 else -1.0)
            x *= 2.0
            z *= 2.0
        else:
            v = 4 if z >= 0 else 5
            z = 2.0 * z - (1.0 if z >= 0 else -1.0)
            x *= 2.0
            y *= 2.0
        scale *= 2.0
        if lev == 0:
            a1 = v
        elif lev == 1:
            a2 = v
    d = (abs(x) + abs(y) + abs(z) - 1.0) * 0.57735027
    return d / scale, a1, a2
