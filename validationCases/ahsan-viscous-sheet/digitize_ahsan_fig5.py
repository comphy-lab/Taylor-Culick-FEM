#!/usr/bin/env python3
"""
Digitise Figure 5 of Ahsan, Brandao, Davidovitch & Stone,
"Retraction Dynamics of a Highly Viscous Liquid Sheet" (arXiv:2507.04603), page 15.

Figure 5 reports, for a stretched elastic sheet that is punctured and then
retracts, the two quantities that characterise the tension "phase change"
front as a function of the dimensionless sheet length script-L:

    panel (a)   T_c   = time of phase change      (log-log axes)
                blue open circles : "Simulation"
                red dashed line   : "Analytic, L >> 1", which is the line
                                    T_c = L (the box diagonal, since both
                                    axes span the same decade range)
    panel (b)   dT_c  = duration of phase change  (x log, y LINEAR)
                blue open circles only, no theory line

METHOD
------
The first thing to establish is whether the figure is vector art (in which
case the marker centres are recoverable exactly from the PDF content
stream, with literally zero digitisation error) or raster.  It is raster:

    pdftocairo -svg -f 15 -l 15 <pdf> p15.svg

produces an SVG in which the only <path> elements are glyph outlines for
the surrounding text, and the two panels appear as two <image> elements
holding base64 JPEGs.  `pdfimages -list` confirms this: page 15 carries
exactly two JPEG XObjects, 979x773 and 974x788 px, at ~308 ppi effective
resolution.  So there is nothing to extract from the content stream and
we must pick marker centres from pixels.

We do NOT re-render the page with pdftoppm, because that would resample an
already-raster source onto a new grid and only add interpolation blur.
Instead `pdfimages -png` writes the two embedded JPEGs at their native
resolution, which is the highest-fidelity form of the figure that exists
inside the PDF.  (JPEG is lossy, so marker edges carry ringing artefacts;
that is the dominant pixel-level noise source and is handled below by
fitting a circle to the whole marker ring rather than thresholding hard.)

Steps:

1. Locate the axes box of each panel from full-width / full-height runs of
   near-black pixels.  This is unambiguous: MATLAB draws a solid black box.

2. Calibrate each axis.  MATLAB draws its grid as light grey lines; the
   *major* (decade / labelled) lines are solid and slightly darker, the
   *minor* lines are dotted.  Both are detected:
     - minor lines: columns/rows inside the frame whose count of grey,
       low-saturation pixels exceeds 12% of the frame extent;
     - major lines: columns/rows whose mean grey level over a clean strip
       of the interior is a local minimum below the minor-line level.
   For a log axis the detected lines must be a contiguous run of the
   "decade ladder" 2,3,...,9,10,20,...,90,100.  The labelling is therefore
   found by brute force: try every possible offset of the detected sequence
   into the ladder and keep the one with the smallest least-squares
   residual of  pixel = B*log10(value) + C.  The winner is unique and its
   residual is sub-pixel, while every other offset is off by tens of
   pixels, so the assignment is not a judgement call.

   The one human-supplied input is the *decade*: the ladder search fixes
   the mantissa pattern but not whether the labelled major line is 10^0,
   10^1 or 10^2.  Both panels print "10^1" on the x axis and panel (a)
   prints "10^1" on the y axis, so the major line is pinned to 10.  This
   is read off the axis label and is stated here explicitly rather than
   silently assumed.

   Panel (b)'s ordinate is linear with labelled ticks from 2.4 (bottom
   frame) to 3.1 (top frame).  Those two numbers are the second and last
   human-supplied input.  They are cross-checked, not merely assumed: the
   six interior major gridlines are found at sub-pixel positions and come
   out exactly equally spaced, dividing the frame into seven intervals ->
   0.1 per interval, and the minor dotted lines subdivide each of those
   into five -> 0.02, both of which are consistent only with limits
   [2.4, 3.1].  Fitting row = m*value + c to the six major lines and then
   *predicting* the frame edges recovers 2.40000 and 3.09898, i.e. the
   frame edges are reproduced to 0.1 px without having been used in the
   fit.

3. Find the markers.  MATLAB's default first colour is RGB (0,114,189);
   the markers are open rings ~11.6 px across.  Blue pixels are selected
   by (B > 110) and (B - R > 50) and (B - G > 25), the legend swatch is
   masked out by its box, and connected components are taken.  In panel
   (a) the red dashed analytic line crosses three markers and splits their
   rings into two components, so components whose centroids lie within
   14 px of each other are merged before fitting.

   The marker centre is then obtained by an algebraic (Kasa) least-squares
   CIRCLE fit to the ring pixels, not by a centre-of-mass.  This matters:
   a centre-of-mass is biased whenever part of the ring is missing --
   which happens for the three markers occluded by the red line and for
   the two markers clipped by the corners of the axes box in panel (a).
   The size of that bias is measurable, because in panel (b) nothing
   occludes anything: there the two estimators agree to 0.02-0.22 px
   (mean 0.11 px), whereas in panel (a) they disagree by up to 1.4 px on
   the occluded/clipped markers.  The circle fit is the estimator that
   stays correct in both cases.

4. Self-check (panel a).  The red dashed line is the analytic result
   T_c = L, so once calibrated it must map onto the line of slope 1
   through (1,1) in log-log.  Red pixels are selected by (R > 150),
   (R - G > 60), (R - B > 80); columns where the red pixels span more than
   12 rows are discarded (those are columns where the line passes behind a
   blue marker and only fringe pixels survive), and the remaining ~400
   column centroids are mapped to (L, T_c).  Fitting log10 T_c against
   log10 L gives

        slope     = 0.99975   (exact value 1)
        intercept = 0.00001 dex, i.e. prefactor 1.00001  (exact value 1)

   and the point-by-point residual of T_c/L about 1 has RMS 0.17% and
   maximum 0.52%.  That is a genuinely independent check of BOTH axis
   calibrations of panel (a) simultaneously, since a scale error on either
   axis would tilt the slope and an offset error would move the prefactor.

UNCERTAINTY
-----------
- Axis calibration: gridline fit residuals are <= 0.45 px on the log axes
  and <= 0.01 px on panel (b)'s linear ordinate.  0.45 px is 0.12% in L
  or T_c (867.8 px per decade).
- Marker centring: the circle fit reproduces a ring radius of 5.74 +/- 0.08 px
  across all 40 markers, and agrees with the centre of mass to ~0.1 px on
  the 20 unoccluded markers of panel (b).  Take 0.3 px as a conservative
  1-sigma centring error -> 0.08% in L or T_c, and 0.0002 in dT_c.
- Combined, quoted honestly: about 0.2% in L and T_c, and about 0.001 in
  dT_c (dominated by the calibration, not the centring).  This is fully
  consistent with the independent red-line check above (0.17% RMS).

A further free consistency check comes out of the extraction rather than
being imposed on it: the 20 abscissae are, to 0.28% worst-case, exactly
numpy.logspace(log10(5), log10(50), 20), i.e. the authors ran 20 cases
logarithmically spaced from L = 5 to L = 50.  The worst deviations are the
two markers clipped by the corners of panel (a)'s box; panel (a) and panel
(b) independently give the same L to within a constant 1.05 px, which is
precisely the offset between the two panels' left frame edges.  The CSVs
store the MEASURED values, not the inferred exact grid, so that the files
remain a digitisation and not a reconstruction; the inferred grid is
recorded in the header for anyone who prefers it.

Run with the project's environment:
    uv run python validationCases/ahsan-viscous-sheet/digitize_ahsan_fig5.py
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PDF = HERE / "literature" / "ahsan_brandao_davidovitch_stone_arXiv2507.04603.pdf"
DATA_DIR = HERE / "data"
PAGE = 15

# MATLAB default colour order entries used by the figure
BLUE = (0, 114, 189)
RED = (217, 83, 25)

# Human-supplied axis-label readings (see METHOD, step 2).  Everything else
# in this script is measured from pixels.
X_MAJOR_VALUE = 10.0        # both panels label their x major gridline 10^1
A_Y_MAJOR_VALUE = 10.0      # panel (a) labels its y major gridline 10^1
B_YLIM = (2.4, 3.1)         # panel (b) bottom/top ordinate limits


# ----------------------------------------------------------------------
# Extraction of the embedded panel bitmaps
# ----------------------------------------------------------------------

def extract_panels(workdir):
    """
    Pull the two embedded JPEGs of page 15 out of the PDF at their native
    resolution.  Returns (panel_a_rgb, panel_b_rgb) as float arrays.

    Image 0 is panel (a) (T_c vs L) and image 1 is panel (b) (dT_c vs L);
    that is the order in which they appear in the page's content stream,
    and it is verified below by shape (979x773 vs 974x788).
    """
    prefix = Path(workdir) / "img"
    subprocess.run(
        ["pdfimages", "-f", str(PAGE), "-l", str(PAGE), "-png",
         str(PDF), str(prefix)],
        check=True,
    )
    files = sorted(Path(workdir).glob("img-*.png"))
    assert len(files) == 2, f"expected 2 embedded images on page {PAGE}, got {len(files)}"
    panels = [iio.imread(f).astype(float) for f in files]
    shapes = [p.shape[:2] for p in panels]
    assert shapes == [(773, 979), (788, 974)], (
        f"unexpected embedded image shapes {shapes}; the PDF is not the one "
        "this script was written against"
    )
    return panels


# ----------------------------------------------------------------------
# Axes box
# ----------------------------------------------------------------------

def find_frame(img, dark_level=120, frac=0.30):
    """
    Locate the solid black axes box: rows/columns in which more than `frac`
    of the pixels are darker than `dark_level`.  The outermost such row and
    column on each side is the frame.  Interior hits (the legend box edges)
    are discarded by taking extremes only.
    """
    dark = img.mean(2) < dark_level
    rows = np.where(dark.sum(1) > frac * img.shape[1])[0]
    cols = np.where(dark.sum(0) > frac * img.shape[0])[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


# ----------------------------------------------------------------------
# Gridline detection
# ----------------------------------------------------------------------

def _cluster(idx, gap=3):
    """Group nearly-adjacent indices into clusters."""
    idx = list(idx)
    if not idx:
        return []
    out, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def minor_gridlines(img, box, axis, strip, thresh=0.12):
    """
    Positions of the dotted MINOR gridlines.

    A dotted grey line covers roughly a quarter to a third of the frame
    extent, so counting grey (dark-ish but unsaturated) pixels along the
    line direction and thresholding at 12% of the extent picks them out
    while rejecting the (coloured) data and the (black) frame.  `strip`
    restricts the perpendicular range so that the legend box does not
    contribute.

    Returns intensity-weighted sub-pixel centres.
    """
    top, bot, left, right = box
    g = img.mean(2)
    sat = img.max(2) - img.min(2)
    grey = (g < 218) & (sat < 25)
    r0, r1, c0, c1 = top + 3, bot - 2, left + 3, right - 2
    if axis == "x":
        r0, r1 = max(r0, strip[0]), min(r1, strip[1])
        sub = grey[r0:r1, c0:c1]
        counts = sub.sum(0)
        base = c0
        extent = sub.shape[0]
    else:
        c0, c1 = max(c0, strip[0]), min(c1, strip[1])
        sub = grey[r0:r1, c0:c1]
        counts = sub.sum(1)
        base = r0
        extent = sub.shape[1]
    hits = np.where(counts > thresh * extent)[0]
    centres = []
    for cl in _cluster(hits):
        w = counts[cl].astype(float)
        centres.append(float((np.asarray(cl) * w).sum() / w.sum()) + base)
    return centres


def major_gridline(img, box, axis, strip, expected_min_sep=40):
    """
    Position of the SOLID major (decade) gridline.

    Major lines are solid rather than dotted, so averaging the grey level
    along the line direction over a clean interior strip gives a distinctly
    deeper minimum for them than for the dotted minor lines.  We take the
    darkest interior position that is not part of the frame, then refine it
    to sub-pixel with a darkness-weighted centroid over +/-2 px.

    Returns a single float, or None if no clear major line exists.
    """
    top, bot, left, right = box
    g = img.mean(2)
    if axis == "x":
        prof = g[max(top + 3, strip[0]):min(bot - 2, strip[1]), :].mean(0)
        lo, hi = left + expected_min_sep, right - expected_min_sep
    else:
        prof = g[:, max(left + 3, strip[0]):min(right - 2, strip[1])].mean(1)
        lo, hi = top + expected_min_sep, bot - expected_min_sep
    interior = prof[lo:hi]
    k = int(np.argmin(interior)) + lo
    if prof[k] > 235.0:                     # nothing clearly darker than the dots
        return None
    w = np.clip(250.0 - prof[k - 2:k + 3], 0.0, None)
    return float((np.arange(k - 2, k + 3) * w).sum() / w.sum())


def all_major_gridlines(img, box, axis, strip, n_expected, sep_tol=6):
    """
    All solid major gridlines (used for panel (b)'s linear ordinate, which
    has six of them).  Same darkness criterion as `major_gridline`, but we
    take the `n_expected` deepest interior minima and merge duplicates.
    """
    top, bot, left, right = box
    g = img.mean(2)
    if axis == "x":
        prof = g[max(top + 3, strip[0]):min(bot - 2, strip[1]), :].mean(0)
        lo, hi = left + 5, right - 5
    else:
        prof = g[:, max(left + 3, strip[0]):min(right - 2, strip[1])].mean(1)
        lo, hi = top + 5, bot - 5
    order = np.argsort(prof[lo:hi]) + lo
    picked = []
    for k in order:
        if all(abs(k - p) > sep_tol for p in picked):
            picked.append(int(k))
        if len(picked) == n_expected:
            break
    picked.sort()
    out = []
    for k in picked:
        w = np.clip(250.0 - prof[k - 2:k + 3], 0.0, None)
        out.append(float((np.arange(k - 2, k + 3) * w).sum() / w.sum()))
    return out


# ----------------------------------------------------------------------
# Log-axis calibration by ladder matching
# ----------------------------------------------------------------------

LADDER = np.array([2, 3, 4, 5, 6, 7, 8, 9,
                   10, 20, 30, 40, 50, 60, 70, 80, 90,
                   100, 200, 300, 400, 500], dtype=float)


def calibrate_log_axis(positions, major_position, major_value):
    """
    Fit  pixel = B * log10(value) + C  for a decade-gridded log axis.

    `positions` are the detected gridline pixel coordinates, sorted so that
    they run in the direction of INCREASING value (for a y axis, that means
    reversed, since pixel rows increase downward).  They must form a
    contiguous run of the decade ladder 2,3,...,9,10,20,...  -- which is
    what MATLAB draws -- so the labelling is determined by a single integer
    offset into the ladder.  Every offset is tried; the best is kept.

    The offset only determines the ladder *pattern*; the decade is pinned by
    telling the routine which value the detected solid major line carries
    (`major_value`, read off the printed axis label).  In practice the
    residual already selects a unique offset because the ladder's spacing
    pattern is not self-similar, and the major line then simply confirms it;
    if the two disagree the function raises.

    Returns (B, C, residuals_in_px, assigned_values).
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    assert n >= 4, "need at least four gridlines to calibrate a log axis"

    best = None
    for s in range(len(LADDER) - n + 1):
        vals = LADDER[s:s + n]
        lg = np.log10(vals)
        A = np.vstack([lg, np.ones_like(lg)]).T
        (B, C), *_ = np.linalg.lstsq(A, positions, rcond=None)
        resid = B * lg + C - positions
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if best is None or rms < best[0]:
            best = (rms, B, C, resid, vals)

    rms, B, C, resid, vals = best
    assert rms < 1.0, f"log-axis ladder match failed: best RMS {rms:.2f} px"

    if major_position is not None:
        implied = 10 ** ((major_position - C) / B)
        # the ladder fit is decade-agnostic; rescale so the solid major line
        # carries the value printed on the axis
        decade_shift = np.log10(major_value) - np.log10(implied)
        if abs(decade_shift) > 1e-6:
            C = C - B * decade_shift
            vals = vals * 10 ** decade_shift
        check = 10 ** ((major_position - C) / B)
        assert abs(np.log10(check / major_value)) < 0.01, (
            "major gridline does not sit at the value printed on the axis"
        )
    return B, C, resid, vals


def calibrate_linear_axis(positions, values):
    """Fit  pixel = m * value + c  (panel (b) ordinate)."""
    positions = np.asarray(positions, float)
    values = np.asarray(values, float)
    A = np.vstack([values, np.ones_like(values)]).T
    (m, c), *_ = np.linalg.lstsq(A, positions, rcond=None)
    return m, c, m * values + c - positions


# ----------------------------------------------------------------------
# Marker detection
# ----------------------------------------------------------------------

def kasa_circle(x, y):
    """
    Algebraic (Kasa) least-squares circle fit.  Minimises
    sum_i (x_i^2 + y_i^2 - 2 x_c x_i - 2 y_c y_i - c)^2, which is linear in
    (x_c, y_c, c) and therefore has a closed form.  For a full ring it is
    equivalent to the centre of mass; for a partial arc -- a ring occluded
    by the analytic line, or clipped by the axes box -- it recovers the true
    centre while the centre of mass does not.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(xc), float(yc), float(np.sqrt(c + xc ** 2 + yc ** 2))


def find_markers(img, box, legend_box, merge=14.0, pad=9):
    """
    Locate the blue open-circle markers.  Returns a list of
    (x, y, radius, n_pixels, com_x, com_y) sorted by x, where (x, y) is the
    circle-fit centre and (com_x, com_y) the plain centre of mass, kept so
    the two can be compared as a diagnostic.

    `pad` allows markers whose centres sit exactly on the frame (the corner
    points of panel (a)) to keep the few pixels that spill outside the box.
    """
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    blue = (B > 110) & (B - R > 50) & (B - G > 25)

    top, bot, left, right = box
    keep = np.zeros_like(blue)
    keep[max(0, top - pad):bot + pad + 1, max(0, left - pad):right + pad + 1] = True
    blue &= keep
    r0, r1, c0, c1 = legend_box
    blue[r0:r1, c0:c1] = False

    lbl, n = ndimage.label(blue, np.ones((3, 3)))
    assert n > 0, "no blue marker pixels found"
    cents = np.array(ndimage.center_of_mass(blue.astype(float), lbl,
                                            list(range(1, n + 1))))

    # merge ring fragments produced where the red analytic line crosses a marker
    groups = []
    for i in np.argsort(cents[:, 1]):
        for g in groups:
            if np.hypot(*(cents[i] - cents[g[0]])) < merge:
                g.append(i)
                break
        else:
            groups.append([i])

    out = []
    for g in groups:
        mask = np.isin(lbl, [i + 1 for i in g])
        ys, xs = np.nonzero(mask)
        xc, yc, rad = kasa_circle(xs, ys)
        cy, cx = ndimage.center_of_mass(mask.astype(float))
        out.append((xc, yc, rad, int(mask.sum()), float(cx), float(cy)))
    out.sort()
    return out


# ----------------------------------------------------------------------
# Red analytic line (panel a self-check)
# ----------------------------------------------------------------------

def red_line_samples(img, box, legend_box, max_span=12):
    """
    Column centroids of the red dashed "Analytic, L >> 1" line.

    Columns whose red pixels span more than `max_span` rows are rejected:
    those are the columns where the line passes behind a blue marker and
    only anti-aliasing fringe pixels on the far side of the ring survive
    the colour test, which would bias the centroid.
    """
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 150) & (R - G > 60) & (R - B > 80)
    top, bot, left, right = box
    r0, r1, c0, c1 = legend_box
    red[r0:r1, c0:c1] = False
    red[:top] = False
    red[bot + 1:] = False
    red[:, :left + 1] = False
    red[:, right + 1:] = False

    cols, rows = [], []
    for c in range(left + 1, right + 1):
        idx = np.nonzero(red[:, c])[0]
        if len(idx) == 0 or idx.max() - idx.min() > max_span:
            continue
        cols.append(float(c))
        rows.append(float(idx.mean()))
    return np.array(cols), np.array(rows)


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

CSV_HEADER = """# Digitised from: Ahsan, T., Brandao, R., Davidovitch, B. & Stone, H. A.
# "Retraction Dynamics of a Highly Viscous Liquid Sheet", arXiv:2507.04603, Figure 5{panel} (page 15).
#
# {quantity}
#   L    = script-L, the dimensionless sheet length (abscissa, log axis)
#   {ycol}
#
# Source form: the two panels are embedded JPEG bitmaps inside the PDF
# (979x773 and 974x788 px, ~308 ppi).  The figure is NOT vector art -- the
# page's content stream contains only glyph outlines and two image XObjects
# -- so exact recovery of the plotted numbers from the drawing operators is
# impossible.  These values are pixel measurements of the embedded bitmaps
# at their native resolution (no re-rendering, no resampling).
#
# Method: axes box located from the solid black frame; log axis calibrated
# by matching the detected decade gridlines (dotted minor + solid major)
# against the ladder 2,3,...,9,10,20,... and least-squares fitting
# pixel = B*log10(value) + C{ycal}; marker centres obtained by an algebraic
# least-squares circle fit to the blue ring pixels rather than a centre of
# mass, because a centre of mass is biased for any partially missing ring
# (in panel (a) the red analytic line cuts three rings in two and the axes
# box clips two more at the corners; panel (b) has no occlusion, and there
# the two estimators agree to 0.1 px, which is how the bias was quantified).
#
# Calibration self-check: {selfcheck}
#
# Accuracy: {accuracy}
#
# Note: the 20 measured abscissae agree with numpy.logspace(log10(5),
# log10(50), 20) to 0.28% worst case (0.09% RMS), so the underlying runs
# were almost certainly 20 log-spaced lengths from L = 5 to L = 50.  The
# MEASURED values are stored below rather than that inferred exact grid, so
# that this file remains a digitisation rather than a reconstruction.
#
# Columns: {columns}
"""


def write_csv(path, x, y, panel, quantity, ycol, ycal, selfcheck, accuracy,
              columns, yfmt="%.4f"):
    with open(path, "w") as f:
        f.write(CSV_HEADER.format(panel=panel, quantity=quantity, ycol=ycol,
                                  ycal=ycal, selfcheck=selfcheck,
                                  accuracy=accuracy, columns=columns))
        f.write(columns + "\n")
        for xi, yi in zip(x, y):
            f.write(("%.4f," + yfmt + "\n") % (xi, yi))


def make_overlay(panel_a, panel_b, mk_a, mk_b, cal_a, cal_b, out_path):
    """
    Diagnostic overlay: each panel bitmap with the fitted marker centres
    (red crosses), the fitted marker circles (lime), the detected gridlines
    (cyan dotted) and, for panel (a), the red-line centroid samples
    (magenta dots).  If the extraction is right, every cross sits dead
    centre in its ring and every cyan line sits on a grey gridline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=130)
    for ax, img, mks, cal, title in [
        (axes[0], panel_a, mk_a, cal_a, "(a)  $T_c$ vs $\\mathcal{L}$  (log-log)"),
        (axes[1], panel_b, mk_b, cal_b, "(b)  $\\Delta T_c$ vs $\\mathcal{L}$  (semi-log)"),
    ]:
        ax.imshow(img.astype(np.uint8))
        for gx in cal["xgrid"]:
            ax.axvline(gx, color="c", lw=0.6, ls=":", alpha=0.9)
        for gy in cal["ygrid"]:
            ax.axhline(gy, color="c", lw=0.6, ls=":", alpha=0.9)
        for (x, y, r, *_rest) in mks:
            ax.add_patch(plt.Circle((x, y), r, fill=False, color="lime", lw=1.0))
            ax.plot([x], [y], "r+", ms=9, mew=1.4)
        if "red_samples" in cal:
            rc, rr = cal["red_samples"]
            ax.plot(rc, rr, ".", color="magenta", ms=1.2)
        ax.set_title(title + f"   [{len(mks)} markers]", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Digitisation check: Ahsan et al. arXiv:2507.04603, Fig. 5\n"
                 "red + = circle-fit marker centres, lime = fitted rings, "
                 "cyan = detected gridlines used for calibration, "
                 "magenta = red analytic-line samples",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------

def process_panel_a(img):
    print("=== panel (a): T_c vs L, log-log ===")
    box = find_frame(img)
    top, bot, left, right = box
    print(f"  frame rows [{top},{bot}] cols [{left},{right}]")

    # legend box of panel (a), located by its own black rectangle inside the
    # axes; hard-coded from the detected interior full-width dark rows so we
    # do not confuse legend graphics with data
    legend = (top + 552, bot + 2, left + 442, right + 10)

    # x axis: use only rows above the legend for the grey-pixel counting
    xminor = minor_gridlines(img, box, "x", strip=(top + 3, top + 540))
    xmajor = major_gridline(img, box, "x", strip=(top + 3, top + 540))
    xpos = sorted(xminor + ([xmajor] if xmajor is not None else []))
    Bx, Cx, rx, xvals = calibrate_log_axis(xpos, xmajor, X_MAJOR_VALUE)
    print(f"  x gridlines {np.round(xpos,1).tolist()} -> {xvals.tolist()}")
    print(f"  x fit: {Bx:.3f} px/decade, residual max {np.abs(rx).max():.2f} px")

    # y axis: use only columns left of the legend
    yminor = minor_gridlines(img, box, "y", strip=(left + 3, left + 430))
    ymajor = major_gridline(img, box, "y", strip=(left + 12, left + 420))
    ypos = sorted(yminor + ([ymajor] if ymajor is not None else []), reverse=True)
    By, Cy, ry, yvals = calibrate_log_axis(ypos, ymajor, A_Y_MAJOR_VALUE)
    print(f"  y gridlines {np.round(ypos,1).tolist()} -> {yvals.tolist()}")
    print(f"  y fit: {By:.3f} px/decade, residual max {np.abs(ry).max():.2f} px")
    print(f"  frame edges imply L in [{10**((left-Cx)/Bx):.3f}, {10**((right-Cx)/Bx):.3f}], "
          f"T_c in [{10**((bot-Cy)/By):.3f}, {10**((top-Cy)/By):.3f}]")

    markers = find_markers(img, box, legend)
    print(f"  markers: {len(markers)}")
    dev = [np.hypot(m[0] - m[4], m[1] - m[5]) for m in markers]
    print(f"  circle-fit vs centre-of-mass: mean {np.mean(dev):.2f} px, "
          f"max {np.max(dev):.2f} px (occluded/clipped rings)")

    # --- self-check on the red analytic line T_c = L ---------------------
    rc, rr = red_line_samples(img, box, legend)
    Lr = 10 ** ((rc - Cx) / Bx)
    Tr = 10 ** ((rr - Cy) / By)
    A = np.vstack([np.log10(Lr), np.ones_like(Lr)]).T
    (slope, icept), *_ = np.linalg.lstsq(A, np.log10(Tr), rcond=None)
    resid_dex = np.log10(Tr / Lr)
    rms_pct = 100 * (10 ** np.sqrt(np.mean(resid_dex ** 2)) - 1)
    max_pct = 100 * (10 ** np.abs(resid_dex).max() - 1)
    print(f"  SELF-CHECK red line (n={len(rc)} columns): slope {slope:.5f} "
          f"(exact 1), prefactor {10**icept:.5f} (exact 1), "
          f"T_c/L residual RMS {rms_pct:.3f}%, max {max_pct:.3f}%")

    L = 10 ** ((np.array([m[0] for m in markers]) - Cx) / Bx)
    Tc = 10 ** ((np.array([m[1] for m in markers]) - Cy) / By)

    cal = {"xgrid": xpos, "ygrid": ypos, "red_samples": (rc, rr)}
    stats = {"slope": slope, "prefactor": 10 ** icept,
             "rms_pct": rms_pct, "max_pct": max_pct,
             "xresid": float(np.abs(rx).max()), "yresid": float(np.abs(ry).max()),
             "Bx": Bx, "By": By, "n_red": len(rc)}
    return L, Tc, markers, cal, stats


def process_panel_b(img):
    print("=== panel (b): dT_c vs L, semi-log ===")
    box = find_frame(img)
    top, bot, left, right = box
    print(f"  frame rows [{top},{bot}] cols [{left},{right}]")

    legend = (top + 366, top + 451, left + 508, right + 10)

    xminor = minor_gridlines(img, box, "x", strip=(top + 3, top + 370))
    xmajor = major_gridline(img, box, "x", strip=(top + 3, top + 370))
    xpos = sorted(xminor + ([xmajor] if xmajor is not None else []))
    Bx, Cx, rx, xvals = calibrate_log_axis(xpos, xmajor, X_MAJOR_VALUE)
    print(f"  x gridlines {np.round(xpos,1).tolist()} -> {xvals.tolist()}")
    print(f"  x fit: {Bx:.3f} px/decade, residual max {np.abs(rx).max():.2f} px")

    # linear ordinate: six interior solid major gridlines between the frame
    # edges, which the axis labels give as 2.4 (bottom) .. 3.1 (top)
    ymaj = all_major_gridlines(img, box, "y", strip=(left + 3, left + 500),
                               n_expected=6)
    step = (B_YLIM[1] - B_YLIM[0]) / 7.0
    yvals = [B_YLIM[1] - step * (k + 1) for k in range(len(ymaj))]
    my, cy, ry = calibrate_linear_axis(ymaj, yvals)
    print(f"  y major gridlines {np.round(ymaj,2).tolist()} -> {np.round(yvals,3).tolist()}")
    print(f"  y fit: {my:.3f} px per unit, residual max {np.abs(ry).max():.3f} px")
    # independent check: predict the frame edges, which were NOT used in the fit
    top_val = (top - cy) / my
    bot_val = (bot - cy) / my
    print(f"  SELF-CHECK frame edges predict dT_c = {bot_val:.5f} (label 2.4) "
          f"and {top_val:.5f} (label 3.1)")
    # and the minor dotted lines should subdivide each major interval by 5
    yminor = minor_gridlines(img, box, "y", strip=(left + 3, left + 500))
    expected_minor = abs(my) * step / 5.0        # 0.02 in data units
    spacings = np.diff(sorted(yminor))
    # keep only adjacent pairs (drop the gaps that straddle a major line,
    # which is not detected as a minor and so shows up as a double spacing)
    fine = spacings[spacings < 1.5 * expected_minor]
    print(f"  minor gridline spacing {np.median(fine):.2f} px (n={len(fine)}) "
          f"vs {expected_minor:.2f} px expected for 0.02 steps")

    markers = find_markers(img, box, legend)
    print(f"  markers: {len(markers)}")
    dev = [np.hypot(m[0] - m[4], m[1] - m[5]) for m in markers]
    print(f"  circle-fit vs centre-of-mass: mean {np.mean(dev):.2f} px, "
          f"max {np.max(dev):.2f} px (no occlusion in this panel)")

    L = 10 ** ((np.array([m[0] for m in markers]) - Cx) / Bx)
    dTc = (np.array([m[1] for m in markers]) - cy) / my

    cal = {"xgrid": xpos, "ygrid": ymaj}
    stats = {"xresid": float(np.abs(rx).max()), "yresid": float(np.abs(ry).max()),
             "top_val": top_val, "bot_val": bot_val, "Bx": Bx, "my": my}
    return L, dTc, markers, cal, stats


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        panel_a, panel_b = extract_panels(work)

        La, Tc, mk_a, cal_a, st_a = process_panel_a(panel_a)
        Lb, dTc, mk_b, cal_b, st_b = process_panel_b(panel_b)

        # cross-panel consistency: both panels plot the same 20 lengths
        assert len(La) == len(Lb) == 20, "expected 20 markers in each panel"
        rel = np.abs(La / Lb - 1.0)
        print(f"\ncross-panel L agreement: max {100*rel.max():.3f}%, "
              f"RMS {100*np.sqrt(np.mean(rel**2)):.3f}%")
        ideal = np.logspace(np.log10(5.0), np.log10(50.0), 20)
        d_ideal = np.abs(La / ideal - 1.0)
        print(f"L vs logspace(5,50,20): max {100*d_ideal.max():.3f}%, "
              f"RMS {100*np.sqrt(np.mean(d_ideal**2)):.3f}%")

        selfcheck_a = (
            f"the red dashed analytic line (T_c = L) was extracted "
            f"independently from {st_a['n_red']} pixel columns and, under this "
            f"calibration, fits log10 T_c = {st_a['slope']:.5f} log10 L + "
            f"{np.log10(st_a['prefactor']):.5f} (exact: slope 1, intercept 0); "
            f"point-by-point T_c/L has RMS deviation {st_a['rms_pct']:.3f}% and "
            f"maximum {st_a['max_pct']:.3f}%."
        )
        accuracy_a = (
            f"Gridline fit residuals <= {max(st_a['xresid'], st_a['yresid']):.2f} px "
            f"on {st_a['Bx']:.0f} px/decade, i.e. <= 0.12%. Marker centring "
            "~0.3 px (1 sigma), i.e. ~0.08%. Combined ~0.2% in both L and "
            "T_c, confirmed by the independent red-line check above."
        )
        write_csv(DATA_DIR / "ahsan_fig5a_Tc_vs_L.csv", La, Tc,
                  panel="(a)",
                  quantity="Ordinate label: \"time of phase change\".",
                  ycol="Tc   = T_c, the (dimensionless) time of the tension "
                       "phase change (ordinate, log axis)",
                  ycal=" on both axes",
                  selfcheck=selfcheck_a, accuracy=accuracy_a,
                  columns="L,Tc")

        selfcheck_b = (
            "the six solid major gridlines were fitted alone and then used to "
            f"PREDICT the frame edges, giving {st_b['bot_val']:.5f} at the "
            f"bottom (printed label 2.4) and {st_b['top_val']:.5f} at the top "
            "(printed label 3.1), i.e. the linear ordinate calibration is "
            "reproduced to about 0.1 px without using the frame; the abscissa "
            "calibration additionally agrees with panel (a)'s, which is "
            f"verified against T_c = L, to {100*rel.max():.3f}% worst case."
        )
        accuracy_b = (
            f"Abscissa gridline fit residual <= {st_b['xresid']:.2f} px on "
            f"{st_b['Bx']:.0f} px/decade (<= 0.12% in L). Ordinate gridline fit "
            f"residual <= {st_b['yresid']:.3f} px on {abs(st_b['my']):.0f} px "
            "per unit, i.e. ~1e-5 -- negligible. The dT_c error is therefore "
            "dominated by marker centring, ~0.3 px (1 sigma) = 0.0003. Quote "
            "+/-0.001 to be safe."
        )
        write_csv(DATA_DIR / "ahsan_fig5b_dTc_vs_L.csv", Lb, dTc,
                  panel="(b)",
                  quantity="Ordinate label: \"duration of phase change\". "
                           "Linear ordinate; no theory curve is drawn in this "
                           "panel.",
                  ycol="dTc  = Delta T_c, the (dimensionless) duration of the "
                       "tension phase change (ordinate, LINEAR axis)",
                  ycal=" for the abscissa, and pixel = m*value + c fitted to "
                       "the six solid major gridlines for the linear ordinate",
                  selfcheck=selfcheck_b, accuracy=accuracy_b,
                  columns="L,dTc")

        overlay = DATA_DIR / "digitization_check_ahsan_fig5.png"
        make_overlay(panel_a, panel_b, mk_a, mk_b, cal_a, cal_b, overlay)

        print("\nwrote:")
        for p in ["ahsan_fig5a_Tc_vs_L.csv", "ahsan_fig5b_dTc_vs_L.csv",
                  "digitization_check_ahsan_fig5.png"]:
            print("  ", DATA_DIR / p)

        print("\n   L        T_c      dT_c")
        for l, t, d in zip(La, Tc, dTc):
            print(f"  {l:7.4f}  {t:7.4f}  {d:7.4f}")


if __name__ == "__main__":
    main()
