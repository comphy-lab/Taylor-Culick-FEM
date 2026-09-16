#!/usr/bin/env python3
"""Render axisymmetric viscoelastic retraction in paired rim-following views.

The accepted four-panel protocol shows laboratory velocity and full
axisymmetric solvent dissipation in a reflected rim-following window,
conformation measures in the unreflected window, and the two tip histories
``R_tip(t)`` and ``V_tip(R_tip)``.  The script reads run metadata instead of
hard-coding the case parameters and records every presentation choice in a
JSON receipt beside the video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.tri import Triangulation

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.25,
    }
)

SUBTRIS = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
FOLLOW_XLIM = (-0.7, 7.5)
LEFT_XLIM = (-7.5, 0.7)
FIELD_YLIM = (-0.82, 0.82)
SPEED_CLIM = (0.0, 0.25)
DISSIPATION_CLIM = (1e-6, 0.32)
SPEED_CMAP = "Blues"
DISSIPATION_CMAP = "hot_r"
AZIMUTHAL_CMAP = "RdBu_r"
MEAN_CMAP = "PuOr_r"
STRAIN_CLIM = (-1.0, 1.0)
FIGSIZE = (16.0, 9.12)
DPI = 100
INITIAL_MESH = Path("_gmsh/ComovingAnnularSheetMesh.msh")
T0_HOLD_SECONDS = 1.0
EARLY_HOLD_SECONDS = 0.5
EARLY_TIME_LIMIT = 0.2
TIME_MATCH_ATOL = 1e-6


def read_pvd(path: Path) -> list[tuple[float, Path]]:
    """Return finite, strictly increasing ``(time, VTU)`` entries."""
    root = ET.parse(path).getroot()
    entries = [
        (
            float(dataset.attrib["timestep"]),
            path.parent / dataset.attrib["file"],
        )
        for dataset in root.iter("DataSet")
    ]
    times = np.asarray([time for time, _ in entries])
    if not entries or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path} must contain finite, strictly increasing times")
    missing = [str(frame) for _, frame in entries if not frame.is_file()]
    if missing:
        raise FileNotFoundError(f"PVD references missing frame: {missing[0]}")
    return entries


def concatenate_pvd_segments(paths: list[Path]) -> list[tuple[float, Path]]:
    """Load ordered PVD segments and reject overlaps at every boundary."""
    if not paths:
        raise ValueError("at least one PVD segment is required")
    entries: list[tuple[float, Path]] = []
    for path in paths:
        segment = read_pvd(path)
        if entries and segment[0][0] <= entries[-1][0]:
            raise ValueError(
                "PVD segments must be ordered and non-overlapping: "
                f"{path} starts at t={segment[0][0]:g} after t={entries[-1][0]:g}"
            )
        entries.extend(segment)
    return entries


def load_case_metadata(path: Path) -> dict[str, float | str]:
    """Load and validate the metadata needed by the renderer."""
    metadata = json.loads(path.read_text())
    expected = {
        "coordinate_system": "axisymmetric",
        "velocity_frame": "laboratory",
        "time_scale": "capillary",
        "constitutive_model": "Oldroyd-B",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"case metadata requires {key}={value!r}")
    values = {}
    for key in ("elastic_modulus", "V_TC", "r0"):
        value = float(metadata[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"case metadata requires finite positive {key}")
        values[key] = value
    for key in ("Oh", "mu_s"):
        value = float(metadata[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"case metadata requires finite non-negative {key}")
        values[key] = value
    if not math.isclose(values["mu_s"], math.sqrt(2.0) * values["Oh"], rel_tol=1e-10):
        raise ValueError("case metadata violates mu_s=sqrt(2)*Oh")
    relaxation = metadata["relaxation_time"]
    if str(relaxation).lower() != "inf" and float(relaxation) <= 0:
        raise ValueError("case metadata requires positive relaxation_time")
    values["relaxation_time"] = relaxation
    return values


def _triangulation(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    blocks = [block.data for block in mesh.cells if block.type == "triangle6"]
    if not blocks:
        raise ValueError("mesh contains no quadratic triangle6 cells")
    points = np.asarray(mesh.points[:, :2], dtype=float)
    tri6 = np.vstack(blocks)
    return points, tri6[:, SUBTRIS].reshape(-1, 3)


def axisymmetric_dissipation(
    points: np.ndarray,
    subtris: np.ndarray,
    velocity: np.ndarray,
    *,
    viscosity: float,
) -> np.ndarray:
    """Return nodal ``2 mu D:D``, including ``D_theta_theta=u_r/r``."""
    triangle_points = points[subtris]
    area2 = (
        (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
        - (triangle_points[:, 2, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
    )
    if np.any(area2 == 0):
        raise ValueError("degenerate linear subtriangle in output mesh")
    gradients = np.empty((len(subtris), 3, 2))
    gradients[:, 0, 0] = (triangle_points[:, 1, 1] - triangle_points[:, 2, 1]) / area2
    gradients[:, 0, 1] = (triangle_points[:, 2, 0] - triangle_points[:, 1, 0]) / area2
    gradients[:, 1, 0] = (triangle_points[:, 2, 1] - triangle_points[:, 0, 1]) / area2
    gradients[:, 1, 1] = (triangle_points[:, 0, 0] - triangle_points[:, 2, 0]) / area2
    gradients[:, 2, 0] = (triangle_points[:, 0, 1] - triangle_points[:, 1, 1]) / area2
    gradients[:, 2, 1] = (triangle_points[:, 1, 0] - triangle_points[:, 0, 0]) / area2
    velocity_gradient = np.einsum("tnc,tnk->tck", velocity[subtris], gradients)
    radius = triangle_points[:, :, 0].mean(axis=1)
    if np.any(radius <= 0):
        raise ValueError("axisymmetric output contains non-positive radius")
    radial_velocity = velocity[subtris, 0].mean(axis=1)
    d_rr = velocity_gradient[:, 0, 0]
    d_zz = velocity_gradient[:, 1, 1]
    d_rz = 0.5 * (velocity_gradient[:, 0, 1] + velocity_gradient[:, 1, 0])
    d_tt = radial_velocity / radius
    cell_value = 2.0 * viscosity * (d_rr**2 + d_zz**2 + d_tt**2 + 2.0 * d_rz**2)
    weights = np.abs(area2)
    nodal_value = np.zeros(len(points))
    nodal_weight = np.zeros(len(points))
    for local_node in range(3):
        np.add.at(nodal_value, subtris[:, local_node], cell_value * weights)
        np.add.at(nodal_weight, subtris[:, local_node], weights)
    return nodal_value / np.maximum(nodal_weight, 1e-300)


def load_mesh(path: Path, *, viscosity: float, v_tc: float, initial: bool = False):
    """Load one VTU, or the saved Gmsh mesh for a prescribed initial state."""
    mesh = meshio.read(path)
    points, subtris = _triangulation(mesh)
    if initial:
        zeros = np.zeros(len(points))
        return points, subtris, zeros, zeros, zeros, zeros

    required = {"velocity", "conformation", "conformation_trace"}
    missing = sorted(required.difference(mesh.point_data))
    if missing:
        raise ValueError(f"{path} lacks point arrays: {', '.join(missing)}")
    velocity = np.asarray(mesh.point_data["velocity"], dtype=float)[:, :2]
    conformation = np.asarray(mesh.point_data["conformation"], dtype=float)
    trace = np.asarray(mesh.point_data["conformation_trace"], dtype=float)
    if conformation.ndim != 2 or conformation.shape[1] < 3:
        raise ValueError("conformation must contain the axisymmetric diagonal component")
    if np.any(conformation[:, 2] <= 0) or np.any(trace <= 0):
        raise ValueError("logarithmic conformation display requires positive values")
    speed = np.linalg.norm(velocity, axis=1) / v_tc
    dissipation = axisymmetric_dissipation(
        points, subtris, velocity, viscosity=viscosity
    )
    azimuthal = np.log10(conformation[:, 2])
    mean = np.log10(trace / 3.0)
    return points, subtris, speed, dissipation, azimuthal, mean


def free_surface(points: np.ndarray, subtris: np.ndarray) -> list[np.ndarray]:
    """Extract the free-surface edges, excluding symmetry and far boundaries."""
    edges = np.vstack(
        [subtris[:, [0, 1]], subtris[:, [1, 2]], subtris[:, [2, 0]]]
    )
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    x_max = float(points[:, 0].max())
    segments = []
    for first, second in boundary:
        start, end = points[first], points[second]
        if max(start[1], end[1]) < 1e-10:
            continue
        if min(start[0], end[0]) > x_max - 1e-6:
            continue
        segments.append(np.vstack([start, end]))
    return segments


def add_free_surface(ax, points: np.ndarray, subtris: np.ndarray) -> None:
    segments = free_surface(points, subtris)
    drawn = list(segments)
    drawn.extend(segment * np.array([1.0, -1.0]) for segment in segments)
    ax.add_collection(
        LineCollection(
            drawn,
            colors="#16c7e8",
            linewidths=1.6,
            zorder=5,
            capstyle="round",
        )
    )
    ax.axhline(0.0, color="0.68", lw=0.75, ls=(0, (5, 4)), zorder=4)
    ax.set_ylim(*FIELD_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()


def style_trace(ax, *, t_end: float, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xlim(0.0, t_end)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$t\sqrt{\sigma/(\rho h_0^3)}$", fontsize=17, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=5)
    ax.tick_params(direction="out", width=1.15, length=4.5, labelsize=13, pad=3)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def playback_repeats(
    time: float,
    *,
    fps: int,
    t0_hold_seconds: float,
    early_hold_seconds: float,
    early_time_limit: float,
) -> int:
    """Return the documented number of encoded frames for one source state."""
    if time == 0.0:
        return max(1, round(fps * t0_hold_seconds))
    if time <= early_time_limit or math.isclose(
        time, early_time_limit, rel_tol=0.0, abs_tol=TIME_MATCH_ATOL
    ):
        return max(1, round(fps * early_hold_seconds))
    return 1


def plot_kinematic_fields(
    ax,
    points: np.ndarray,
    subtris: np.ndarray,
    speed: np.ndarray,
    dissipation: np.ndarray,
    *,
    speed_limits: tuple[float, float],
):
    """Plot the horizontally reflected comoving kinematic view."""
    reflected_x = -points[:, 0]
    upper = ax.tripcolor(
        Triangulation(reflected_x, points[:, 1], subtris),
        speed,
        shading="gouraud",
        cmap=SPEED_CMAP,
        norm=Normalize(vmin=speed_limits[0], vmax=speed_limits[1], clip=True),
    )
    lower = ax.tripcolor(
        Triangulation(reflected_x, -points[:, 1], subtris),
        dissipation,
        shading="gouraud",
        cmap=DISSIPATION_CMAP,
        norm=LogNorm(
            vmin=DISSIPATION_CLIM[0], vmax=DISSIPATION_CLIM[1], clip=True
        ),
    )
    segments = free_surface(points, subtris)
    reflected = [segment * np.array([-1.0, 1.0]) for segment in segments]
    reflected.extend(segment * np.array([-1.0, -1.0]) for segment in segments)
    ax.add_collection(
        LineCollection(
            reflected,
            colors="#16c7e8",
            linewidths=1.6,
            zorder=5,
            capstyle="round",
        )
    )
    ax.axhline(0.0, color="0.68", lw=0.75, ls=(0, (5, 4)), zorder=4)
    ax.set_ylim(*FIELD_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return upper, lower


def plot_strain_fields(
    ax,
    points: np.ndarray,
    subtris: np.ndarray,
    azimuthal: np.ndarray,
    mean: np.ndarray,
):
    upper = ax.tripcolor(
        Triangulation(points[:, 0], points[:, 1], subtris),
        np.clip(azimuthal, *STRAIN_CLIM),
        shading="gouraud",
        cmap=AZIMUTHAL_CMAP,
        norm=TwoSlopeNorm(
            vmin=STRAIN_CLIM[0], vcenter=0.0, vmax=STRAIN_CLIM[1]
        ),
    )
    lower = ax.tripcolor(
        Triangulation(points[:, 0], -points[:, 1], subtris),
        np.clip(mean, *STRAIN_CLIM),
        shading="gouraud",
        cmap=MEAN_CMAP,
        norm=TwoSlopeNorm(vmin=STRAIN_CLIM[0], vcenter=0.0, vmax=STRAIN_CLIM[1]),
    )
    add_free_surface(ax, points, subtris)
    return upper, lower


def _parameter_label(metadata: dict[str, float | str]) -> str:
    relaxation = metadata["relaxation_time"]
    relaxation_text = r"\infty" if str(relaxation).lower() == "inf" else f"{float(relaxation):g}"
    return (
        rf"$(a)\quad Oh={float(metadata['Oh']):g},\quad "
        rf"Ec={float(metadata['elastic_modulus']):g},\quad \lambda={relaxation_text}$"
    )


def render_frame(
    item,
    *,
    frames_dir: Path,
    history: np.ndarray,
    metadata: dict[str, float | str],
    radius_ylim: tuple[float, float],
    speed_ylim: tuple[float, float],
    field_speed_limits: tuple[float, float],
) -> None:
    index, (time, path) = item
    destination = frames_dir / f"frame_{index:05d}.png"
    points, subtris, speed_field, dissipation, azimuthal, mean = load_mesh(
        path,
        viscosity=float(metadata["mu_s"]),
        v_tc=float(metadata["V_TC"]),
        initial=path.suffix == ".msh",
    )
    radius = float(np.interp(time, history[:, 0], history[:, 1]))
    speed = float(np.interp(time, history[:, 0], history[:, 3]))
    relative = points.copy()
    relative[:, 0] -= radius

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    left_ax = fig.add_axes([0.035, 0.650, 0.445, 0.210])
    right_ax = fig.add_axes([0.520, 0.650, 0.445, 0.210])
    radius_ax = fig.add_axes([0.080, 0.090, 0.375, 0.395])
    speed_ax = fig.add_axes([0.570, 0.090, 0.375, 0.395])
    speed_cax = fig.add_axes([0.080, 0.900, 0.375, 0.020])
    dissipation_cax = fig.add_axes([0.080, 0.600, 0.375, 0.020])
    azimuthal_cax = fig.add_axes([0.570, 0.900, 0.375, 0.020])
    mean_cax = fig.add_axes([0.570, 0.600, 0.375, 0.020])

    speed_map, dissipation_map = plot_kinematic_fields(
        left_ax,
        relative,
        subtris,
        speed_field,
        dissipation,
        speed_limits=field_speed_limits,
    )
    azimuthal_map, mean_map = plot_strain_fields(
        right_ax, relative, subtris, azimuthal, mean
    )
    left_ax.set_xlim(*LEFT_XLIM)
    right_ax.set_xlim(*FOLLOW_XLIM)

    box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "black",
        "edgecolor": "none",
        "alpha": 0.68,
    }
    left_ax.text(
        0.018,
        1.04,
        _parameter_label(metadata),
        transform=left_ax.transAxes,
        fontsize=16,
        color="white",
        bbox=box,
        clip_on=False,
        va="bottom",
        zorder=8,
    )
    right_ax.text(
        0.018,
        1.04,
        r"$(b)$",
        transform=right_ax.transAxes,
        fontsize=16,
        color="white",
        bbox=box,
        clip_on=False,
        va="bottom",
        zorder=8,
    )
    right_ax.text(
        0.982,
        1.04,
        rf"$t={time:.2f}$",
        transform=right_ax.transAxes,
        ha="right",
        fontsize=16,
        color="white",
        bbox=box,
        clip_on=False,
        va="bottom",
        zorder=8,
    )

    speed_bar = fig.colorbar(speed_map, cax=speed_cax, orientation="horizontal")
    speed_bar.ax.xaxis.set_ticks_position("top")
    speed_bar.ax.xaxis.set_label_position("top")
    speed_bar.set_label(r"$|\mathbf{u}|/V_{\mathrm{TC}}$", fontsize=14, labelpad=3)
    speed_bar.set_ticks(np.linspace(*field_speed_limits, 6))
    speed_bar.ax.tick_params(labelsize=10.5, width=0.9, length=3, pad=1)

    dissipation_bar = fig.colorbar(
        dissipation_map, cax=dissipation_cax, orientation="horizontal"
    )
    dissipation_bar.set_label(r"$2\mu\,\mathbf{D}{:}\mathbf{D}$", fontsize=14, labelpad=1)
    dissipation_bar.ax.tick_params(labelsize=10.5, width=0.9, length=3, pad=1)

    azimuthal_bar = fig.colorbar(
        azimuthal_map, cax=azimuthal_cax, orientation="horizontal"
    )
    azimuthal_bar.ax.xaxis.set_ticks_position("top")
    azimuthal_bar.ax.xaxis.set_label_position("top")
    azimuthal_bar.set_label(r"$\log_{10} A_{\theta\theta}$", fontsize=17, labelpad=3)
    azimuthal_bar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    azimuthal_bar.ax.tick_params(labelsize=12, width=1.0, length=3.5, pad=2)

    mean_bar = fig.colorbar(mean_map, cax=mean_cax, orientation="horizontal")
    mean_bar.set_label(r"$\log_{10}[\mathrm{tr}(\mathbf{A})/3]$", fontsize=17, labelpad=2)
    mean_bar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    mean_bar.ax.tick_params(labelsize=12, width=1.0, length=3.5, pad=2)

    mask = history[:, 0] <= time + 1e-10
    radius_ax.plot(history[:, 0], history[:, 1], color="0.75", lw=1.8, zorder=1)
    radius_ax.plot(history[mask, 0], history[mask, 1], color="#c1121f", lw=2.6, zorder=3)
    radius_ax.plot(
        time,
        radius,
        "o",
        ms=7,
        color="#c1121f",
        mec="black",
        mew=1.0,
        clip_on=False,
        zorder=4,
    )
    radius_ax.text(0.025, 0.91, r"$(c)$", transform=radius_ax.transAxes, fontsize=16)
    style_trace(
        radius_ax,
        t_end=float(history[-1, 0]),
        ylabel=r"$R_{\mathrm{tip}}/h_0$",
        ylim=radius_ylim,
    )

    radius_limits = (float(history[:, 1].min()), float(history[:, 1].max()))
    speed_ax.plot(history[:, 1], history[:, 3], color="0.75", lw=1.8, zorder=1)
    speed_ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=2)
    speed_ax.plot(history[mask, 1], history[mask, 3], color="#c1121f", lw=2.6, zorder=3)
    speed_ax.plot(
        radius,
        speed,
        "o",
        ms=7,
        color="#c1121f",
        mec="black",
        mew=1.0,
        clip_on=False,
        zorder=4,
    )
    speed_ax.text(0.025, 0.91, r"$(d)$", transform=speed_ax.transAxes, fontsize=16)
    speed_ax.set_xlim(*radius_limits)
    speed_ax.set_ylim(*speed_ylim)
    speed_ax.set_xlabel(r"$R_{\mathrm{tip}}/h_0$", fontsize=17, labelpad=2)
    speed_ax.set_ylabel(r"$u^*=V_{\mathrm{tip}}/V_{\mathrm{TC}}$", fontsize=18, labelpad=5)
    speed_ax.tick_params(direction="out", width=1.15, length=4.5, labelsize=13, pad=3)
    speed_ax.minorticks_on()
    for spine in speed_ax.spines.values():
        spine.set_linewidth(1.25)

    fig.savefig(destination, dpi=DPI)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pvd_source_receipt(paths: list[Path]) -> list[dict[str, str]]:
    """Return the complete, ordered identity of the PVD inputs."""
    return [{"path": str(path), "sha256": sha256(path)} for path in paths]


def load_history(path: Path) -> np.ndarray:
    """Load the ``t,R,U,u*`` prefix and reject ambiguous histories."""
    history = np.atleast_2d(np.loadtxt(path, comments="#"))
    if history.ndim != 2 or history.shape[1] < 4:
        raise ValueError(f"{path} must contain at least t, R, U and u*")
    if not np.all(np.isfinite(history[:, :4])):
        raise ValueError(f"{path} contains non-finite t, R, U or u*")
    if np.any(np.diff(history[:, 0]) <= 0):
        raise ValueError(f"{path} times must be strictly increasing")
    return history


def padded_limits(values: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    """Return finite plot limits with a relative margin."""
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, max(abs(low), abs(high), 1.0) * 1e-3)
    return low - fraction * span, high + fraction * span


def stride_selection(
    entries: list[tuple[float, Path]], *, stride: int, early_time_limit: float
) -> list[tuple[int, tuple[float, Path]]]:
    """Subsample long histories while retaining every held early state and the endpoint."""

    if stride < 1:
        raise ValueError("frame stride must be positive")
    if stride == 1:
        return list(enumerate(entries))
    early = [
        index
        for index, (time, _) in enumerate(entries)
        if time <= early_time_limit
        or math.isclose(
            time,
            early_time_limit,
            rel_tol=0.0,
            abs_tol=TIME_MATCH_ATOL,
        )
    ]
    first_later = early[-1] + 1 if early else 0
    indices = [*early, *range(first_later, len(entries), stride)]
    if entries and (not indices or indices[-1] != len(entries) - 1):
        indices.append(len(entries) - 1)
    return [(index, entries[index]) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--case-json",
        type=Path,
        help="case metadata (default: case.json beside the run directory)",
    )
    parser.add_argument(
        "--segment-pvd",
        action="append",
        type=Path,
        help=(
            "ordered PVD segment to concatenate; repeat for restarts "
            "(default: <run-dir>/liquid.pvd)"
        ),
    )
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="encode every Nth post-early source frame while retaining the final frame",
    )
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument(
        "--initial-state",
        choices=("auto", "reconstruct", "omit"),
        default="auto",
        help="use stored t=0, reconstruct it from prescribed ICs, or omit reconstruction",
    )
    parser.add_argument("--t0-hold-seconds", type=float, default=T0_HOLD_SECONDS)
    parser.add_argument("--early-hold-seconds", type=float, default=EARLY_HOLD_SECONDS)
    parser.add_argument("--early-time-limit", type=float, default=EARLY_TIME_LIMIT)
    parser.add_argument("--radius-limits", type=float, nargs=2)
    parser.add_argument("--tip-speed-limits", type=float, nargs=2)
    parser.add_argument(
        "--speed-limits",
        type=float,
        nargs=2,
        default=SPEED_CLIM,
        metavar=("MIN", "MAX"),
        help="velocity-field colour limits (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.fps < 1 or args.procs < 1 or args.stride < 1:
        parser.error("--fps, --procs and --stride must be positive")
    if min(args.t0_hold_seconds, args.early_hold_seconds, args.early_time_limit) < 0:
        parser.error("playback hold durations and early-time limit must be non-negative")
    if args.indices is not None and not args.skip_encode:
        parser.error("--indices is a QA-only selection and requires --skip-encode")
    if args.indices is not None and args.stride != 1:
        parser.error("--indices and --stride cannot be combined")

    args.run_dir = args.run_dir.resolve()
    pvd_paths = (
        [path.resolve() for path in args.segment_pvd]
        if args.segment_pvd
        else [args.run_dir / "liquid.pvd"]
    )
    case_json = (
        args.case_json.resolve()
        if args.case_json is not None
        else args.run_dir.parent / "case.json"
    )
    if not case_json.is_file():
        raise SystemExit(f"missing case metadata: {case_json}")
    metadata = load_case_metadata(case_json)
    pvd_entries = concatenate_pvd_segments(pvd_paths)
    initial_mesh = args.run_dir / INITIAL_MESH
    stored_t0 = abs(pvd_entries[0][0]) <= 1e-12
    reconstruct_t0 = not stored_t0 and args.initial_state in {"auto", "reconstruct"}
    if args.initial_state == "reconstruct" and stored_t0:
        raise SystemExit("PVD already contains t=0; refusing a duplicate reconstruction")
    if reconstruct_t0 and not initial_mesh.is_file():
        raise SystemExit(f"missing saved initial mesh: {initial_mesh}")
    entries = ([(0.0, initial_mesh)] if reconstruct_t0 else []) + pvd_entries

    history = load_history(args.run_dir / "control.txt")
    if entries[0][0] == 0.0 and history[0, 0] > 0.0:
        initial_row = np.array(history[0], copy=True)
        initial_row[:4] = [0.0, float(metadata["r0"]), 0.0, 0.0]
        history = np.vstack([initial_row, history])
    if history[0, 0] > entries[0][0] + 1e-12 or history[-1, 0] < entries[-1][0] - 1e-12:
        raise ValueError("control history does not cover all rendered frame times")

    radius_ylim = tuple(args.radius_limits) if args.radius_limits else padded_limits(history[:, 1])
    speed_ylim = (
        tuple(args.tip_speed_limits)
        if args.tip_speed_limits
        else (min(-0.025, float(history[:, 3].min())), max(0.025, 1.1 * float(history[:, 3].max())))
    )
    field_speed_limits = tuple(args.speed_limits)
    if (
        radius_ylim[0] >= radius_ylim[1]
        or speed_ylim[0] >= speed_ylim[1]
        or not all(math.isfinite(value) for value in field_speed_limits)
        or field_speed_limits[0] < 0
        or field_speed_limits[0] >= field_speed_limits[1]
    ):
        parser.error("plot limits must be increasing")

    selected = stride_selection(
        entries,
        stride=args.stride,
        early_time_limit=args.early_time_limit,
    )
    if args.indices is not None:
        bad = [index for index in args.indices if index < 0 or index >= len(entries)]
        if bad:
            raise SystemExit(f"frame indices out of range: {bad}")
        selected = [(index, entries[index]) for index in args.indices]

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    expected_frames = [
        args.frames_dir / f"frame_{index:05d}.png" for index, _ in selected
    ]
    stale_frames = sorted(set(args.frames_dir.glob("frame_*.png")).difference(expected_frames))
    if stale_frames:
        raise SystemExit(f"frames directory contains stale frame: {stale_frames[0]}")
    print(f"Rendering {len(selected)}/{len(entries)} frames", flush=True)
    work = partial(
        render_frame,
        frames_dir=args.frames_dir,
        history=history,
        metadata=metadata,
        radius_ylim=radius_ylim,
        speed_ylim=speed_ylim,
        field_speed_limits=field_speed_limits,
    )
    if args.procs == 1:
        for item in selected:
            work(item)
    else:
        with Pool(args.procs) as pool:
            for done, _ in enumerate(pool.imap_unordered(work, selected), 1):
                if done % 50 == 0 or done == len(selected):
                    print(f"{done}/{len(selected)}", flush=True)

    if args.skip_encode:
        return
    missing_frames = [str(frame) for frame in expected_frames if not frame.is_file()]
    if missing_frames:
        raise SystemExit(f"missing rendered frame: {missing_frames[0]}")
    playback_frames = [
        frame
        for (_, (time, _)), frame in zip(selected, expected_frames, strict=True)
        for _ in range(
            playback_repeats(
                time,
                fps=args.fps,
                t0_hold_seconds=args.t0_hold_seconds,
                early_hold_seconds=args.early_hold_seconds,
                early_time_limit=args.early_time_limit,
            )
        )
    ]
    with imageio.get_writer(
        args.out,
        fps=args.fps,
        codec="libx264",
        quality=None,
        ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in playback_frames:
            writer.append_data(imageio.imread(frame))

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(args.out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = {
        "case_metadata": str(case_json),
        "case_metadata_sha256": sha256(case_json),
        "source_pvd": str(pvd_paths[0]) if len(pvd_paths) == 1 else None,
        "source_pvds": pvd_source_receipt(pvd_paths),
        "initial_condition_source": (
            {
                "mesh": str(initial_mesh),
                "kind": "reconstructed from saved initial mesh and prescribed ICs",
                "fields": (
                    f"R_tip={float(metadata['r0']):g}, u=0, "
                    "2*mu*D:D=0, Psi=log(A)=0"
                ),
            }
            if reconstruct_t0
            else {"kind": "stored VTU" if stored_t0 else "omitted"}
        ),
        "source_control": str(args.run_dir / "control.txt"),
        "source_arrays": ["velocity", "conformation[:,2]", "conformation_trace"],
        "rendered_quantities": [
            "|u|/V_TC in the rim-following spatial view",
            "2*mu*(D_rr^2+D_zz^2+D_tt^2+2*D_rz^2)",
            "log10(A_theta_theta)",
            "log10(trace(A)/3)",
        ],
        "layout": {
            "top_left": "horizontally reflected rim-following window; tip right, film left",
            "top_right": "rim-following window; tip left, film right",
            "bottom_left": "R_tip versus capillary time",
            "bottom_right": "u* versus R_tip",
            "upper_colourbars": "above field panels; ticks and labels on upper side",
            "lower_colourbars": "below field panels with a clear label gap",
            "annotations": "boxed labels above and clear of the sheet interface",
        },
        "dissipation_reconstruction": {
            "coordinate_order": "x=r, y=z; velocity=(u_r,u_z)",
            "mu": float(metadata["mu_s"]),
            "hoop_term": "D_tt=u_r/r included",
            "projection": "piecewise-linear subtriangle value area-averaged to nodes",
        },
        "source_frames": len(entries),
        "selected_source_frames": len(selected),
        "frame_stride": args.stride,
        "frames": len(playback_frames),
        "time_first": selected[0][1][0],
        "time_last": selected[-1][1][0],
        "fps": args.fps,
        "early_time_playback": {
            "t0_hold_frames": playback_repeats(
                0.0,
                fps=args.fps,
                t0_hold_seconds=args.t0_hold_seconds,
                early_hold_seconds=args.early_hold_seconds,
                early_time_limit=args.early_time_limit,
            ),
            "t0_hold_seconds": args.t0_hold_seconds,
            "positive_times_held": [
                time
                for _, (time, _) in selected
                if 0 < time
                and (
                    time <= args.early_time_limit
                    or math.isclose(
                        time,
                        args.early_time_limit,
                        rel_tol=0.0,
                        abs_tol=TIME_MATCH_ATOL,
                    )
                )
            ],
            "frames_per_positive_early_state": max(
                1, round(args.fps * args.early_hold_seconds)
            ),
            "seconds_per_positive_early_state": args.early_hold_seconds,
            "later_frames_per_state": 1,
        },
        "crop": {"left_x": LEFT_XLIM, "right_x": FOLLOW_XLIM, "field_y": FIELD_YLIM},
        "colour_limits": {
            "speed": field_speed_limits,
            "dissipation": DISSIPATION_CLIM,
            "azimuthal": STRAIN_CLIM,
            "mean": STRAIN_CLIM,
        },
        "colormaps": {
            "speed": SPEED_CMAP,
            "dissipation": DISSIPATION_CMAP,
            "azimuthal": AZIMUTHAL_CMAP,
            "mean": MEAN_CMAP,
        },
        "display_saturation": "both logarithmic conformation fields are clipped to [-1, 1]",
        "history_limits": {"radius": radius_ylim, "tip_speed": speed_ylim},
        "video": str(args.out),
        "video_bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "ffprobe": json.loads(probe.stdout),
    }
    receipt_path = args.out.with_name(args.out.stem + "_render_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
