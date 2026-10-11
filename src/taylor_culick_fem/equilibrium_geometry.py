"""Geometry of the steady elastocapillary problem: reference grids and pull-back remeshing.

The reference body is the initial half sheet of the dynamic runs: full thickness 1, a hole
of radius ``r0`` on the midplane and a quarter-circle edge cap of radius 1/2. It is meshed
in four structured blocks (a cap split into ``core``, ``lower`` and ``upper``, and the
``sheet`` beyond), optionally with all nine nodes of each quadratic element given.

Large openings crush the edge region radially by orders of magnitude, so a fixed reference
mesh degrades. :func:`design_grids` designs a mesh in the deformed configuration of a
converged state (a cap O-grid fitted to the deformed edge, vertical columns graded outward)
and maps every node back through that solution to its reference position. A problem built
on these grids is the same Lagrangian problem on the exact reference body; only the
discretisation changes, and the old solution serves only as the initial guess.

Nothing here imports pyoomph.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

HALF_THICKNESS = 0.5


def radial_nodes(start: float, first_spacing: float, growth: float, end: float,
                 max_spacing: float = math.inf) -> np.ndarray:
    """Nodes from ``start`` to ``end``: spacing grows geometrically from ``first_spacing``
    and is capped at ``max_spacing``."""
    if not (end > start and first_spacing > 0 and growth >= 1 and max_spacing >= first_spacing):
        raise ValueError("radial nodes need end > start, 0 < first spacing <= max spacing and growth >= 1")
    nodes = [start]
    spacing = first_spacing
    while nodes[-1] + spacing < end:
        nodes.append(nodes[-1] + spacing)
        spacing = min(spacing * growth, max_spacing)
    if end - nodes[-1] < 0.5 * spacing / growth and len(nodes) > 1:
        nodes[-1] = end
    else:
        nodes.append(end)
    return np.asarray(nodes)


def coons_patch(bottom, top, left, right, nu: int, nv: int) -> np.ndarray:
    """Nodes of a Coons patch, shape ``(nu + 1, nv + 1, 2)``: edges exact, interior blended."""
    p00, p10, p01, p11 = bottom(0.0), bottom(1.0), top(0.0), top(1.0)
    grid = np.empty((nu + 1, nv + 1, 2))
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            if j == 0:
                grid[i, j] = bottom(u)
            elif j == nv:
                grid[i, j] = top(u)
            elif i == 0:
                grid[i, j] = left(v)
            elif i == nu:
                grid[i, j] = right(v)
            else:
                grid[i, j] = ((1 - v) * bottom(u) + v * top(u) + (1 - u) * left(v) + u * right(v)
                              - ((1 - u) * (1 - v) * p00 + u * (1 - v) * p10
                                 + (1 - u) * v * p01 + u * v * p11))
    return grid


def _line(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return lambda s: p + s * (q - p)


def reference_grids(r0: float, length: float, elements_per_quarter: int, growth: float,
                    first_spacing: float | None = None, max_spacing: float = math.inf,
                    core_fraction: float = 0.5, quadratic: bool = False) -> dict[str, np.ndarray]:
    """Block grids of the initial half sheet in reference coordinates.

    The quarter-disk cap (centre ``(r0 + h/2, 0)``) is split into a square ``core`` and two
    curved blocks, ``lower`` (from the tip) and ``upper`` (to the shoulder); the flat ``sheet``
    beyond it has ``2 * elements_per_quarter`` elements through the half thickness and
    geometrically growing radial spacing, capped at ``max_spacing`` because the round-off of
    the radial displacement shows up as shear across thin elements in proportion to their
    aspect ratio. With ``quadratic`` every element carries its nine nodes, so each block grid
    has twice as many intervals as elements in each direction.
    """
    a = HALF_THICKNESS
    n = elements_per_quarter
    m = 2 * n if quadratic else n
    cx = r0 + a
    c = core_fraction * a

    def arc(angle: float) -> np.ndarray:
        return np.array([cx + a * math.cos(angle), a * math.sin(angle)])

    tip, middle, shoulder = arc(math.pi), arc(0.75 * math.pi), arc(0.5 * math.pi)
    centre = np.array([cx, 0.0])
    core_a, core_b, core_e = np.array([cx - c, 0.0]), np.array([cx - c, c]), np.array([cx, c])
    grids = {
        "core": coons_patch(_line(core_a, centre), _line(core_b, core_e),
                            _line(core_a, core_b), _line(centre, core_e), m, m),
        "lower": coons_patch(_line(tip, core_a), _line(middle, core_b),
                             lambda v: arc(math.pi - 0.25 * math.pi * v), _line(core_a, core_b), m, m),
        "upper": coons_patch(_line(core_b, core_e), lambda u: arc(0.75 * math.pi - 0.25 * math.pi * u),
                             _line(core_b, middle), _line(core_e, shoulder), m, m),
    }
    first = first_spacing if first_spacing is not None else a / (2 * n)
    radii = radial_nodes(cx, first, growth, length, max_spacing)
    if quadratic:
        radii = np.insert(radii, np.arange(1, len(radii)), 0.5 * (radii[1:] + radii[:-1]))
    heights = np.concatenate([np.linspace(0.0, c, m + 1), np.linspace(c, a, m + 1)[1:]])
    sheet = np.empty((len(radii), 2 * m + 1, 2))
    sheet[:, :, 0] = radii[:, None]
    sheet[:, :, 1] = heights[None, :]
    grids["sheet"] = sheet
    return grids


def _lagrange(x):
    """1D quadratic Lagrange basis at -1, 0, 1 and derivatives."""
    return (np.array([0.5 * x * (x - 1), 1 - x * x, 0.5 * x * (x + 1)]),
            np.array([x - 0.5, -2 * x, x + 0.5]))


class FieldMap:
    """Q2 element data of a solved state: reference nodes, displacement, local node indices."""

    def __init__(self, Xe: np.ndarray, ue: np.ndarray, ij: np.ndarray):
        self.Xe, self.ue, self.ij = Xe, ue, ij  # (nel, 9, 2), (nel, 9, 2), (nel, 9, 2) in {0,1,2}
        self.ye = Xe + ue
        self._trees = {}

    @classmethod
    def from_problem(cls, problem) -> FieldMap:
        """Read every Q2 element of the ``solid`` mesh of a solved problem."""
        mesh = problem.get_mesh("solid")
        if hasattr(mesh, "_interpolate_hanging_values"):
            mesh._interpolate_hanging_values()
        idx = mesh.get_nodal_field_indices()
        ix, iy = idx["displacement_x"], idx["displacement_y"]
        Xe, ue, ij = [], [], []
        for element in mesh.elements():
            if element.nnode() != 9:
                raise RuntimeError("FieldMap expects Q2 quadrilaterals")
            xs, us, ks = [], [], []
            for k in range(9):
                node = element.node_pt(k)
                loc = element.local_coordinate_of_node(k)
                xs.append((node.x(0), node.x(1)))
                us.append((node.value(ix), node.value(iy)))
                ks.append((round(loc[0]) + 1, round(loc[1]) + 1))
            Xe.append(xs)
            ue.append(us)
            ij.append(ks)
        return cls(np.asarray(Xe), np.asarray(ue), np.asarray(ij))

    def save(self, path, **extra):
        np.savez(path, Xe=self.Xe, ue=self.ue, ij=self.ij, **extra)

    @classmethod
    def load(cls, path) -> FieldMap:
        data = np.load(path)
        return cls(data["Xe"], data["ue"], data["ij"])

    def _shape(self, e, s, t):
        ls, dls = _lagrange(s)
        lt, dlt = _lagrange(t)
        i, j = self.ij[e, :, 0], self.ij[e, :, 1]
        return ls[i] * lt[j], dls[i] * lt[j], ls[i] * dlt[j]

    def _tree(self, which):
        if which not in self._trees:
            pts = self.Xe if which == "ref" else self.ye
            self._trees[which] = cKDTree(pts.mean(axis=1))
        return self._trees[which]

    def locate(self, p, which: str, tol: float = 1e-9, k: int = 16):
        """Element and local coordinates of point ``p`` in the ``ref`` or ``def`` configuration."""
        nodes = self.Xe if which == "ref" else self.ye
        p = np.asarray(p, float)
        best = None
        for kk in (k, 4 * k, 16 * k):
            _, cand = self._tree(which).query(p, k=min(kk, len(nodes)))
            for e in np.atleast_1d(cand):
                s = t = 0.0
                ok = False
                for _ in range(40):
                    N, Ns, Nt = self._shape(e, s, t)
                    x = N @ nodes[e]
                    jac = np.array([Ns @ nodes[e], Nt @ nodes[e]]).T
                    r = x - p
                    if np.hypot(*r) < 1e-13 * (1 + np.hypot(*p)):
                        ok = True
                        break
                    try:
                        ds = np.linalg.solve(jac, r)
                    except np.linalg.LinAlgError:
                        break
                    s, t = s - ds[0], t - ds[1]
                    if abs(s) > 3 or abs(t) > 3:
                        break
                    if np.hypot(*ds) < 1e-14:
                        ok = True
                        break
                if ok:
                    excess = max(abs(s), abs(t)) - 1
                    if excess <= tol:
                        return e, s, t
                    if best is None or excess < best[0]:
                        best = (excess, e, s, t)
        if best is not None and best[0] < 1e-3:
            return best[1:]
        raise RuntimeError(f"point {p} not found in the {which} configuration")

    def evaluate(self, e, s, t):
        N, _, _ = self._shape(e, s, t)
        return N @ self.Xe[e], N @ self.ue[e]

    def forward(self, X):
        """Displacement at reference point ``X``."""
        e, s, t = self.locate(X, "ref", tol=1e-6)
        return self.evaluate(e, s, t)[1]

    def pullback(self, y):
        """Reference point mapped to deformed point ``y``."""
        e, s, t = self.locate(y, "def", tol=1e-6)
        return self.evaluate(e, s, t)[0]


def reference_surface(s0, r0=1.0):
    """Reference free surface by arclength from the tip: cap arc, then the top line."""
    cx, arc_len = r0 + HALF_THICKNESS, 0.5 * math.pi * HALF_THICKNESS
    s0 = np.atleast_1d(np.asarray(s0, float))
    out = np.empty((len(s0), 2))
    on_arc = s0 <= arc_len
    phi = math.pi - s0[on_arc] / HALF_THICKNESS
    out[on_arc] = np.c_[cx + HALF_THICKNESS * np.cos(phi), HALF_THICKNESS * np.sin(phi)]
    out[~on_arc] = np.c_[cx + (s0[~on_arc] - arc_len), np.full((~on_arc).sum(), HALF_THICKNESS)]
    return out


def deformed_surface(fmap: FieldMap, length: float, r0=1.0, n_arc=400, n_line=1500):
    """Dense samples (s0, y_r, y_z, deformed arclength) of the deformed free surface."""
    arc_len = 0.5 * math.pi * HALF_THICKNESS
    s_line_end = arc_len + (length - (r0 + HALF_THICKNESS))
    s_arc = np.linspace(0.0, arc_len, n_arc)
    s_line = arc_len + (s_line_end - arc_len) * (np.geomspace(1e-6, 1.0, n_line) - 1e-6) / (1 - 1e-6)
    s0 = np.unique(np.concatenate([s_arc, s_line]))
    X = reference_surface(s0, r0)
    y = np.array([x + fmap.forward(x) for x in X])
    ell = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(y, axis=0).T))])
    return s0, y, ell


def design_grids(fmap: FieldMap, n: int, length: float, growth: float = 1.08,
                 max_spacing: float = 300.0, core_fraction: float = 0.5, r0: float = 1.0,
                 first_aspect: float = 1.0) -> dict[str, np.ndarray]:
    """Nine-node reference grids whose deformed images are a well-shaped mesh of the current
    state: every node, mid-side and centre nodes included, is pulled back from its designed
    deformed position, so the deformed elements are the designed ones."""
    f = core_fraction
    m = 2 * n
    s0, ysurf, ell = deformed_surface(fmap, length, r0)
    arc_len = 0.5 * math.pi * HALF_THICKNESS

    def surf_at_ell(target):
        return float(np.interp(target, ell, s0))

    def surf_point(s):
        return np.array([np.interp(s, s0, ysurf[:, 0]), np.interp(s, s0, ysurf[:, 1])])

    tip = surf_point(0.0)
    shoulder = surf_point(arc_len)
    ell_shoulder = float(np.interp(arc_len, s0, ell))
    s_mid = surf_at_ell(0.5 * ell_shoulder)
    middle = surf_point(s_mid)
    centre = np.array([shoulder[0], 0.0])
    core_a = np.array([tip[0] + (1 - f) * (centre[0] - tip[0]), 0.0])
    core_b = np.array([core_a[0], f * shoulder[1]])
    core_e = np.array([centre[0], f * shoulder[1]])

    # Surface parameters along the curved edges, equally spaced in deformed arclength.
    def curve(s_from, s_to):
        e_from, e_to = np.interp(s_from, s0, ell), np.interp(s_to, s0, ell)
        return lambda v: surf_at_ell(e_from + v * (e_to - e_from))

    s_lower = curve(0.0, s_mid)
    s_upper = curve(s_mid, arc_len)
    line = _line

    blocks = {
        "core": coons_patch(line(core_a, centre), line(core_b, core_e),
                            line(core_a, core_b), line(centre, core_e), m, m),
        "lower": coons_patch(line(tip, core_a), line(middle, core_b),
                             lambda v: surf_point(s_lower(v)), line(core_a, core_b), m, m),
        "upper": coons_patch(line(core_b, core_e), lambda u: surf_point(s_upper(u)),
                             line(core_b, middle), line(core_e, shoulder), m, m),
    }
    # Surface parameters of the designed surface nodes (exact pull-back onto the reference surface).
    surface_s = {("lower", 0, j): s_lower(j / m) for j in range(m + 1)}
    surface_s.update({("upper", i, m): s_upper(i / m) for i in range(m + 1)})

    # Sheet: vertical columns from the cap outward, graded in deformed radius.
    far_r = length + fmap.forward(np.array([length, 0.0]))[0]
    first = first_aspect * shoulder[1] / (2 * n)
    radii = radial_nodes(centre[0], first, growth, far_r, max_spacing)
    radii = np.insert(radii, np.arange(1, len(radii)), 0.5 * (radii[1:] + radii[:-1]))
    fractions = np.concatenate([np.linspace(0.0, f, m + 1), np.linspace(f, 1.0, m + 1)[1:]])
    sheet = np.empty((len(radii), 2 * m + 1, 2))
    col_s = np.empty(len(radii))
    beyond = s0 >= arc_len
    if not np.all(np.diff(ysurf[beyond, 0]) > 0):
        raise RuntimeError("deformed free surface is not a graph of r beyond the shoulder")
    for i, r in enumerate(radii):
        if i == 0:
            s_top = arc_len
        else:
            s_top = max(float(np.interp(r, ysurf[beyond, 0], s0[beyond])), arc_len)
        top = surf_point(s_top)
        col_s[i] = s_top
        sheet[i, :, 0] = r
        sheet[i, :, 1] = fractions * top[1]
        sheet[i, -1] = top
    blocks["sheet"] = sheet

    # Pull back: free-surface nodes exactly by arclength, far column exactly at X = length,
    # midplane nodes by a 1D inverse of the monotone r(X, 0), everything else through the
    # inverse map; shared nodes are pulled back once.
    cache: dict[tuple[float, float], np.ndarray] = {}
    xb = np.unique(np.concatenate([np.linspace(r0, r0 + 4 * HALF_THICKNESS, 800),
                                   np.geomspace(r0 + 4 * HALF_THICKNESS, length, 1200)]))
    rb = np.array([x + fmap.forward(np.array([x, 0.0]))[0] for x in xb])
    if not np.all(np.diff(rb) > 0):
        raise RuntimeError("deformed midplane radius is not monotone")

    def pull_bottom(r):
        x = float(np.interp(r, rb, xb))
        for _ in range(30):
            res = x + fmap.forward(np.array([x, 0.0]))[0] - r
            h = 1e-7 * max(1.0, x)
            slope = (h + fmap.forward(np.array([x + h, 0.0]))[0] - fmap.forward(np.array([x, 0.0]))[0]) / h
            step = res / slope
            x -= step
            if abs(step) < 1e-13 * max(1.0, x):
                break
        return np.array([x, 0.0])

    def pull(y):
        key = (round(float(y[0]), 12), round(float(y[1]), 12))
        if key not in cache:
            X = fmap.pullback(y)
            cache[key] = X
        return cache[key]

    grids = {}
    for name, grid in blocks.items():
        out = np.empty_like(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                y = grid[i, j]
                if name == "sheet" and i == len(radii) - 1:
                    X = np.array([length, fractions[j] * HALF_THICKNESS])
                elif (name, i, j) in surface_s:
                    X = reference_surface(surface_s[(name, i, j)], r0)[0]
                elif name == "sheet" and j == grid.shape[1] - 1:
                    X = reference_surface(col_s[i], r0)[0]
                elif abs(y[1]) < 1e-12:
                    X = pull_bottom(float(y[0]))
                else:
                    X = pull(y).copy()
                key = (round(float(y[0]), 12), round(float(y[1]), 12))
                cache[key] = X
                out[i, j] = X
        grids[name] = out
    # Shared edges: make coincident nodes bitwise identical.
    for name, grid in grids.items():
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                y = blocks[name][i, j]
                grid[i, j] = cache[(round(float(y[0]), 12), round(float(y[1]), 12))]
    return grids


def deformed_quality(fmap: FieldMap) -> np.ndarray:
    """Per element: ratio of the singular values of d(y)/d(s,t) at the centre (1 = square)."""
    out = np.empty(len(fmap.Xe))
    for e in range(len(fmap.Xe)):
        _, Ns, Nt = fmap._shape(e, 0.0, 0.0)
        jac = np.array([Ns @ fmap.ye[e], Nt @ fmap.ye[e]]).T
        sv = np.linalg.svd(jac, compute_uv=False)
        out[e] = sv[0] / max(sv[1], 1e-300)
    return out


def incompressibility_error(fmap: FieldMap, x_max: float = 5.0) -> float:
    """Largest |J - 1| (J = det F including the hoop stretch) at sample points of elements
    whose reference centre lies below ``x_max``; a resolution monitor for the edge region."""
    worst = 0.0
    for e in np.where(fmap.Xe.mean(axis=1)[:, 0] < x_max)[0]:
        for s, t in ((0, 0), (-0.77, -0.77), (0.77, -0.77), (-0.77, 0.77), (0.77, 0.77)):
            N, Ns, Nt = fmap._shape(e, s, t)
            dX = np.array([Ns @ fmap.Xe[e], Nt @ fmap.Xe[e]]).T
            dy = np.array([Ns @ fmap.ye[e], Nt @ fmap.ye[e]]).T
            X, y = N @ fmap.Xe[e], N @ fmap.ye[e]
            worst = max(worst, abs(np.linalg.det(dy @ np.linalg.inv(dX)) * y[0] / X[0] - 1.0))
    return worst
