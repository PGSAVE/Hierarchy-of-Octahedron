"""
Numerical experiments on the Sierpinski octahedron (octahedron |x|+|y|+|z| <= 1).

    python research.py

1. Pile-up near the walls: mu(|z| < eps) ~ eps^a with a = log2(3/2)
2. Dimension of horizontal slices {z = c}
      the number of level-n pieces meeting the plane is a product of 2x2 matrices
      chosen by the binary digits of c, so
          dim(slice) = Lyapunov exponent / log 2
3. Shadow of the chaos-game measure on the z axis: its dimension (Garsia entropy)
4. Furstenberg's dimension conservation:
      dim(shadow) + dim(typical slice through a typical point) = log2 6
"""
import numpy as np
from numba import njit

import octaflake as of

LOG2_6 = np.log2(6)

# Normalised height t = (c - z0) / rho of the plane inside a sub-octahedron.
# Children: 4 equatorial (t -> 2t if |t| <= 1/2), one polar (t -> 2t -+ 1).
# All heights at level n are  {2^n c} or {2^n c} - 1,  so the counts (a, b) evolve by
M0 = np.array([[4.0, 0.0], [1.0, 1.0]])    # next binary digit of c is 0
M1 = np.array([[1.0, 1.0], [0.0, 4.0]])    # next binary digit of c is 1


@njit(cache=True)
def lyapunov(bits, m0, m1, a, b):
    s = 0.0
    for d in bits:
        if d == 0:
            a, b = m0[0, 0] * a + m0[0, 1] * b, m0[1, 0] * a + m0[1, 1] * b
        else:
            a, b = m1[0, 0] * a + m1[0, 1] * b, m1[1, 0] * a + m1[1, 1] * b
        n = a + b
        s += np.log(n)
        a /= n
        b /= n
    return s / len(bits) / np.log(2)


def slice_dim_periodic(bits):
    """Exact slice dimension for c with a periodic binary expansion."""
    m = np.eye(2)
    for d in bits:
        m = (M1 if d else M0) @ m
    return np.log2(max(abs(np.linalg.eigvals(m)))) / len(bits)


def binary_digits_of_sum(digits):
    """Binary digits of (sum_k digits_k 2^-k) mod 1 for digits in {-1, 0, 1}, exactly."""
    num = 0
    for d in digits:
        num = 2 * num + int(d)
    num %= 2 ** len(digits)
    return np.array([int(ch) for ch in bin(num)[2:].zfill(len(digits))], dtype=np.int64)


def main():
    rng = np.random.default_rng(0)

    print("1. pile-up near the plane z = 0")
    pts, _ = of.chaos_game(20_000_000)
    eps = 0.1 / 2.0 ** np.arange(7)
    mass = np.array([(np.abs(pts[:, 2]) < e).mean() for e in eps])
    slope = np.polyfit(np.log(eps), np.log(mass), 1)[0]
    print(f"   mu(|z|<eps) ~ eps^{slope:.3f}   theory log2(3/2) = {np.log2(1.5):.3f}")
    print(f"   {100 * mass[-1]:.2f}% of all points lie within {eps[-1]:.4f} of the plane "
          f"(volume share {100 * eps[-1] * 3:.2f}%)\n")

    print("2. dimension of horizontal slices z = c")
    print(f"   dyadic c (e.g. 0, 1/2, 3/8): 2  (the slice contains solid squares)")
    print(f"   c = 1/3 (digits 0101...):   {slice_dim_periodic([0, 1]):.4f}")
    print(f"   c = 1/7 (digits 001...):    {slice_dim_periodic([0, 0, 1]):.4f}")
    typical = lyapunov(rng.integers(0, 2, 4_000_000), M0, M1, 1.0, 0.0)
    print(f"   Lebesgue-typical c:         {typical:.4f}")
    print(f"   Marstrand bound dim - 1:    {LOG2_6 - 1:.4f}")
    dims = []
    for _ in range(40):
        digits = rng.choice([-1, 0, 1], size=4000, p=[1 / 6, 2 / 3, 1 / 6])
        bits = binary_digits_of_sum(digits)[:-64]
        a, b = (1.0, 0.0) if digits[np.flatnonzero(digits)[0]] > 0 else (0.0, 1.0)
        dims.append(lyapunov(bits, M0, M1, a, b))
    slice_mu = float(np.mean(dims))
    print(f"   slice through a mu-typical point: {slice_mu:.4f}\n")

    print("3. dimension of the shadow of mu on the z axis (Garsia entropy)")
    p = np.array([1 / 6, 2 / 3, 1 / 6])
    dist = np.array([1.0])
    h_prev = 0.0
    for n in range(1, 21):
        up = np.zeros(2 * len(dist) - 1)
        up[::2] = dist
        dist = np.convolve(up, p)
        nz = dist[dist > 0]
        h = -(nz * np.log2(nz)).sum()
        rate, h_prev = h - h_prev, h
    print(f"   dim(shadow) = {rate:.4f}   (< 1: the shadow is a full segment, "
          f"but the measure on it is singular)\n")

    print("4. dimension conservation")
    print(f"   {rate:.4f} + {slice_mu:.4f} = {rate + slice_mu:.4f}    log2 6 = {LOG2_6:.4f}")


if __name__ == "__main__":
    main()
