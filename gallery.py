"""
Renders a gallery of the Sierpinski octahedron without any GPU / VTK dependency.

    pip install numpy numba pillow
    python gallery.py all            # or: hero cutaway xray slices variants build sweep

Output goes to ./gallery/
"""
import os
import sys
import time

import numpy as np
from numba import njit, prange
from PIL import Image, ImageDraw, ImageFont

import octaflake as of

OUT = "gallery"

# one hue family per axis, lighter for +, deeper for -
PALETTE = np.array([
    [0.96, 0.42, 0.32], [0.98, 0.70, 0.26],   # +x coral, -x amber
    [0.16, 0.70, 0.66], [0.46, 0.80, 0.34],   # +y teal,  -y green
    [0.58, 0.46, 0.96], [0.30, 0.56, 0.98],   # +z violet, -z blue
])
CUT_COLOR = np.array([0.82, 0.72, 0.58])
BG_TOP = np.array([0.075, 0.080, 0.110])
BG_BOTTOM = np.array([0.010, 0.010, 0.016])


def linear(c):
    """sRGB -> linear, the ray tracer shades in linear light."""
    return np.asarray(c, float) ** 2.2


# ---------------------------------------------------------------- ray marching

@njit(cache=True)
def _scene(x, y, z, levels, cut):
    d, a1, a2 = of.estimate(x, y, z, levels)
    # optional cutaway: remove the half-space n.p > c
    dc = cut[0] * x + cut[1] * y + cut[2] * z - cut[3]
    if cut[4] > 0.5 and dc > d:
        return dc, a1, a2, True
    return d, a1, a2, False


@njit(cache=True)
def _dist(x, y, z, levels, cut):
    return _scene(x, y, z, levels, cut)[0]


@njit(cache=True)
def _shade_pixel(ox, oy, oz, dx, dy, dz, levels, cut, light, palette, cut_color,
                 pix_angle):
    t = 0.0
    hit = False
    for _ in range(400):
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        d = _dist(px, py, pz, levels, cut)
        eps = max(2e-5, 0.6 * pix_angle * t)
        if d < eps:
            hit = True
            break
        t += 0.9 * d
        if t > 20.0:
            break
    if not hit:
        return -1.0, 0.0, 0.0
    px = ox + dx * t
    py = oy + dy * t
    pz = oz + dz * t
    d, a1, a2, is_cut = _scene(px, py, pz, levels, cut)

    # normal: tetrahedral gradient
    h = max(1e-5, 0.5 * pix_angle * t)
    k0 = _dist(px + h, py - h, pz - h, levels, cut)
    k1 = _dist(px - h, py - h, pz + h, levels, cut)
    k2 = _dist(px - h, py + h, pz - h, levels, cut)
    k3 = _dist(px + h, py + h, pz + h, levels, cut)
    nx = k0 - k1 - k2 + k3
    ny = -k0 - k1 + k2 + k3
    nz = -k0 + k1 - k2 + k3
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-30
    nx /= nl
    ny /= nl
    nz /= nl

    # soft shadow toward the key light
    sx = px + nx * 4 * h
    sy = py + ny * 4 * h
    sz = pz + nz * 4 * h
    shadow = 1.0
    st = 1e-3
    for _ in range(96):
        ds = _dist(sx + light[0] * st, sy + light[1] * st, sz + light[2] * st, levels, cut)
        shadow = min(shadow, 12.0 * ds / st)
        if shadow < 1e-3:
            shadow = 0.0
            break
        st += max(ds, 2e-3)
        if st > 4.0:
            break
    shadow = max(0.0, shadow)

    # ambient occlusion along the normal
    occ = 0.0
    w = 1.0
    for i in range(1, 6):
        hh = 0.012 * i
        dd = _dist(px + nx * hh, py + ny * hh, pz + nz * hh, levels, cut)
        occ += w * (hh - dd)
        w *= 0.7
    ao = min(1.0, max(0.0, 1.0 - 6.0 * occ))

    if is_cut:
        br = cut_color[0]
        bg = cut_color[1]
        bb = cut_color[2]
    else:
        br = 0.72 * palette[a1, 0] + 0.28 * palette[a2, 0]
        bg = 0.72 * palette[a1, 1] + 0.28 * palette[a2, 1]
        bb = 0.72 * palette[a1, 2] + 0.28 * palette[a2, 2]

    diff = max(0.0, nx * light[0] + ny * light[1] + nz * light[2])
    fill = max(0.0, -nx * light[0] - ny * light[1] + 0.3 * nz)
    # half vector for a small specular highlight
    hx = light[0] - dx
    hy = light[1] - dy
    hz = light[2] - dz
    hl = np.sqrt(hx * hx + hy * hy + hz * hz)
    spec = max(0.0, (nx * hx + ny * hy + nz * hz) / hl) ** 40
    sky = 0.5 + 0.5 * nz
    fres = (1.0 + (nx * dx + ny * dy + nz * dz)) ** 4
    key = 1.6 * diff * shadow
    amb = (0.035 + 0.10 * sky + 0.10 * fill + 0.05 * (1.0 - sky)) * ao
    r = br * (key + amb) + 0.30 * spec * shadow + 0.020 * fres * ao
    g = bg * (key + amb) + 0.30 * spec * shadow + 0.025 * fres * ao
    b = bb * (key + amb) + 0.32 * spec * shadow + 0.040 * fres * ao
    return r, g, b


@njit(parallel=True, cache=True)
def _render(width, height, cam, target, fov, levels, cut, light, palette, cut_color,
            bg_top, bg_bottom):
    img = np.empty((height, width, 3), dtype=np.float64)
    fx = target[0] - cam[0]
    fy = target[1] - cam[1]
    fz = target[2] - cam[2]
    fl = np.sqrt(fx * fx + fy * fy + fz * fz)
    fx /= fl
    fy /= fl
    fz /= fl
    # right = forward x up(z)
    rx = fy
    ry = -fx
    rz = 0.0
    rl = np.sqrt(rx * rx + ry * ry) + 1e-12
    rx /= rl
    ry /= rl
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    tan = np.tan(0.5 * fov)
    pix_angle = 2.0 * tan / height
    for j in prange(height):
        for i in range(width):
            sx = (2.0 * (i + 0.5) / width - 1.0) * tan * width / height
            sy = (1.0 - 2.0 * (j + 0.5) / height) * tan
            dx = fx + sx * rx + sy * ux
            dy = fy + sx * ry + sy * uy
            dz = fz + sx * rz + sy * uz
            dl = np.sqrt(dx * dx + dy * dy + dz * dz)
            r, g, b = _shade_pixel(cam[0], cam[1], cam[2], dx / dl, dy / dl, dz / dl,
                                   levels, cut, light, palette, cut_color, pix_angle)
            if r < 0.0:
                s = j / (height - 1.0)
                img[j, i, 0] = bg_top[0] * (1 - s) + bg_bottom[0] * s
                img[j, i, 1] = bg_top[1] * (1 - s) + bg_bottom[1] * s
                img[j, i, 2] = bg_top[2] * (1 - s) + bg_bottom[2] * s
            else:
                img[j, i, 0] = r
                img[j, i, 1] = g
                img[j, i, 2] = b
    return img


def camera(azimuth, elevation, distance, target=(0.0, 0.0, 0.0)):
    a = np.radians(azimuth)
    e = np.radians(elevation)
    t = np.array(target, float)
    return t + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]), t


def to_image(img, supersample=1):
    img = np.clip(img, 0.0, None)
    img = img / (1.0 + 0.10 * img)                 # soft shoulder
    img = np.clip(img, 0.0, 1.0) ** (1 / 2.2)
    im = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
    if supersample > 1:
        im = im.resize((im.width // supersample, im.height // supersample), Image.LANCZOS)
    return im


def raymarch(size, azimuth=38, elevation=24, distance=4.2, fov_deg=32, levels=9, cut=None,
             light=(0.45, -0.35, 0.82), supersample=2, target=(0, 0, 0)):
    cam, tgt = camera(azimuth, elevation, distance, target)
    lt = np.array(light, float)
    lt /= np.linalg.norm(lt)
    cutv = np.zeros(5) if cut is None else np.array([*cut, 1.0], float)
    s = size * supersample
    img = _render(s, s, cam, tgt, np.radians(fov_deg), levels, cutv, lt, linear(PALETTE),
                  linear(CUT_COLOR), linear(BG_TOP), linear(BG_BOTTOM))
    return to_image(img, supersample)


# ---------------------------------------------------------------- helpers

def font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(im, text, size=22, pos=(18, 14), fill=(235, 235, 240)):
    ImageDraw.Draw(im).text(pos, text, font=font(size), fill=fill)
    return im


def rotation(azimuth, elevation):
    """Rows: screen-right, screen-up, towards-viewer (z up in world)."""
    a = np.radians(azimuth)
    e = np.radians(elevation)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(-view, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -view)
    return np.array([right, up, view])


def save(im, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    im.save(path)
    print(f"saved {path}")


# ---------------------------------------------------------------- scenes

def hero():
    save(raymarch(1100, levels=9), "hero.png")


def cutaway():
    ims = [
        label(raymarch(640, cut=(0, 0, 1, 0.0), elevation=34), "cut z = 0: a solid square"),
        label(raymarch(640, cut=(0, 0, 1, 0.1), elevation=34), "cut z = 0.1"),
        label(raymarch(640, cut=(1 / np.sqrt(3),) * 3 + (0.0,), azimuth=45, elevation=35.26,
                       distance=4.4), "cut x+y+z = 0"),
    ]
    sheet = Image.new("RGB", (640 * 3, 640))
    for k, im in enumerate(ims):
        sheet.paste(im, (640 * k, 0))
    save(sheet, "cutaway.png")


def xray_image(pts, code, rot, size, scale, gain=4.0, palette=PALETTE):
    rgb = palette[code // 6].astype(np.float32)
    dens = np.zeros((size, size))
    col = np.zeros((size, size, 3))
    zbuf = np.full((size, size), -np.inf)
    zcol = np.zeros((size, size, 3))
    of.splat(pts, rgb, rot, scale, size, size, dens, col, zbuf, zcol)
    mean = col / np.maximum(dens, 1)[..., None]
    ref = np.percentile(dens[dens > 0], 99.7)
    inten = np.log1p(gain * dens / ref * 10) / np.log1p(gain * 10)
    img = np.clip(mean * inten[..., None], 0, 1) ** (1 / 1.6)
    return Image.fromarray((img * 255).astype(np.uint8))


MAGMA = np.array([[0, 0, 0], [0.18, 0.05, 0.30], [0.62, 0.12, 0.42],
                  [0.96, 0.45, 0.25], [1.0, 0.85, 0.55], [1, 1, 0.95]])


def density_image(pts, rot, size, scale, gain=3.0):
    """Log-density of the projected points through a magma-like ramp."""
    dens = np.zeros((size, size))
    col = np.zeros((size, size, 3))
    zbuf = np.full((size, size), -np.inf)
    zcol = np.zeros((size, size, 3))
    of.splat(pts, np.zeros((len(pts), 3), np.float32), rot, scale, size, size, dens, col, zbuf,
             zcol)
    ref = np.percentile(dens[dens > 0], 99.5)
    t = np.clip(np.log1p(gain * dens / ref * 10) / np.log1p(gain * 10), 0, 1)
    x = t * (len(MAGMA) - 1)
    i = np.minimum(x.astype(int), len(MAGMA) - 2)
    f = (x - i)[..., None]
    return Image.fromarray(((MAGMA[i] * (1 - f) + MAGMA[i + 1] * f) * 255).astype(np.uint8))


def xray(n=80_000_000):
    """Shadows of the chaos-game measure: where points pile up when projected."""
    pts, code = of.chaos_game(n)
    size = 640
    views = [
        ((0, 90 - 1e-6), "along z: solid square"),
        ((45, 35.264), "along (1,1,1): solid hexagon"),
        ((45, 0), "along (1,1,0): walls edge-on"),
    ]
    sheet = Image.new("RGB", (size * 3, size))
    for k, ((az, el), text) in enumerate(views):
        im = density_image(pts, rotation(az, el), size, size * 0.46)
        sheet.paste(label(im, text, 20), (size * k, 0))
    save(sheet, "shadows.png")
    im = xray_image(pts, code, rotation(0, 90 - 1e-6), 1000, 1000 * 0.47)
    save(im, "xray_top.png")


def slice_rgb(lev, top, levels):
    """Escape-time colouring: a hole is tinted by its top-level piece and gets brighter
    the deeper (the later) it was carved out, i.e. the closer it is to the fractal."""
    h, w = lev.shape
    img = np.zeros((h, w, 3))
    ins = lev < 0
    img[ins] = 0.45 + 0.55 * PALETTE[np.maximum(top[ins], 0)]
    hole = lev > 0
    t = (lev[hole] - 1) / (levels - 1)
    img[hole] = PALETTE[np.maximum(top[hole], 0)] * (0.06 + 0.94 * t[:, None] ** 1.3)
    img[lev == 0] = BG_BOTTOM
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def plane_slice(normal, c, size, levels=16, extent=1.02):
    n = np.array(normal, float)
    n /= np.linalg.norm(n)
    # in-plane basis
    a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    a -= n * (a @ n)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    if abs(n[2]) > 0.9:        # axis slices: diamond in its usual orientation
        a = np.array([1.0, 0, 0])
        b = np.array([0, 1.0, 0])
    step = 2 * extent / size
    origin = n * c - extent * a + extent * b
    lev, top = of.slice_image(origin, a * step, -b * step, size, size, levels)
    return slice_rgb(lev, top, levels)


def slices():
    size = 520
    d = 1 / np.sqrt(3)
    items = [
        ((0, 0, 1), 0.0, "z = 0  (solid square)"),
        ((0, 0, 1), 0.1, "z = 0.1"),
        ((0, 0, 1), 1 / 3, "z = 1/3"),
        ((1, 1, 1), 1e-9, "x+y+z = 0"),
        ((1, 1, 1), 0.5 * d, "x+y+z = 1/2"),
        ((1, 1, 1), 1.0 * d - 1e-9, "x+y+z = 1  (face: Sierpinski triangle)"),
    ]
    sheet = Image.new("RGB", (size * 3, size * 2))
    for k, (nrm, c, text) in enumerate(items):
        im = plane_slice(nrm, c, size)
        sheet.paste(label(im, text, 19), ((k % 3) * size, (k // 3) * size))
    save(sheet, "slices.png")


def variants(n=25_000_000):
    size = 560
    rot = rotation(38, 24)
    items = []
    for rule in ("classic", "no-repeat", "no-opposite", "adjacent"):
        dim = of.rule_dimension(of.rule_matrix(*of.RULES[rule]))
        items.append((dict(rule=rule), f"{rule}   dim = {dim:.3f}"))
    chiral = of.cyclic_rule(1, 2)
    items.append((dict(rule=chiral), f"chiral: no +1,+2 in cycle   dim = {of.rule_dimension(chiral):.3f}"))
    items.append((dict(ratio=1 / 3), f"classic, ratio 1/3   dim = {np.log(6) / np.log(3):.3f}"))
    sheet = Image.new("RGB", (size * 3, size * 2))
    for k, (kw, text) in enumerate(items):
        pts, code = of.chaos_game(n, **kw)
        im = xray_image(pts, code, rot, size, size * 0.43)
        sheet.paste(label(im, text, 19), ((k % 3) * size, (k // 3) * size))
    save(sheet, "variants.png")


def save_gif(frames, name, ms):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print(f"saved {path}")


def build(size=420, frames_per_level=12, max_level=7):
    """Levels 0..max_level while the camera turns: one octahedron -> 6 -> 36 -> ..."""
    frames = []
    total = frames_per_level * (max_level + 1)
    for k in range(total):
        lev = k // frames_per_level
        im = raymarch(size, azimuth=20 + 360 * k / total, levels=lev, supersample=1)
        frames.append(label(im, f"level {lev}:  {6 ** lev:,} octahedra", 17))
    save_gif(frames, "build.gif", 90)


def sweep(size=420, n_frames=90):
    """A plane z = c sweeping through the fractal."""
    frames = []
    for k in range(n_frames):
        c = k / (n_frames - 1)
        frames.append(label(plane_slice((0, 0, 1), c, size, levels=14), f"z = {c:.3f}", 18))
    save_gif(frames + frames[::-1], "sweep.gif", 60)


def zoom(size=300, n_frames=40):
    """Seamless infinite zoom into the centre. Near the origin the fractal is invariant
    under scaling by 2, so flying from distance d to d/2 along (1,1,1) loops perfectly.
    The camera sits inside one of the 8 tetrahedral holes, looking at its apex."""
    frames = []
    for k in range(n_frames):
        c = 0.3 * 2.0 ** (-k / n_frames)
        cam = np.array([c, c, c])
        img = _render(2 * size, 2 * size, cam, np.zeros(3), np.radians(70), 18, np.zeros(5),
                      cam / np.linalg.norm(cam), linear(PALETTE), linear(CUT_COLOR),
                      linear(BG_TOP), linear(BG_BOTTOM))
        frames.append(to_image(img, 2))
    save_gif(frames, "zoom.gif", 70)


SCENES = dict(hero=hero, cutaway=cutaway, xray=xray, slices=slices, variants=variants,
              build=build, sweep=sweep, zoom=zoom)

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(SCENES)
    for name in names:
        t = time.time()
        SCENES[name]()
        print(f"{name}: {time.time() - t:.1f}s")
