#!/usr/bin/env python3
"""Plot common and per-case tip histories for the viscoelastic Ec campaign.

Each repeated ``--case`` argument is ``EC SCHEMA CONTROL``.  The output
directory receives ``ec_campaign.png/.pdf`` and one ``ec_<value>.png/.pdf``
pair per case.  Each figure contains ``R_tip(t)``, ``u*(t)`` and the ordered
trajectory ``u*(R_tip)``; the latter is never sorted by radius because recoil
can produce loops.  Solver histories are augmented by the exact initial state
``(t,R_tip,u*)=(0,r0,0)`` when needed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from viscoelastic_campaign_history import TipHistory, load_control


def configure_matplotlib(*, use_tex: bool = True) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": use_tex,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _colour(ec: float, positive_values: list[float]) -> str | tuple[float, ...]:
    if ec == 0:
        return "#202020"
    logarithms = np.log10(positive_values)
    position = 0.5 if np.ptp(logarithms) == 0 else (
        np.log10(ec) - logarithms.min()
    ) / np.ptp(logarithms)
    return plt.get_cmap("viridis")(0.12 + 0.78 * position)


def _style(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="out", width=1.2, labelsize=15, pad=4)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def _draw_history(axes: np.ndarray, history: TipHistory, *, colour, label: str) -> None:
    axes[0].plot(history.time, history.radius, color=colour, lw=2.2, label=label)
    axes[1].plot(history.time, history.ustar, color=colour, lw=2.2, label=label)
    axes[2].plot(history.radius, history.ustar, color=colour, lw=2.2, label=label)
    abscissae = (history.time, history.time, history.radius)
    ordinates = (history.radius, history.ustar, history.ustar)
    for ax, abscissa, ordinate in zip(axes, abscissae, ordinates, strict=True):
        ax.scatter(
            [abscissa[0], abscissa[-1]],
            [ordinate[0], ordinate[-1]],
            s=24,
            color=[colour],
            edgecolor="black",
            linewidth=0.55,
            zorder=4,
            clip_on=False,
        )


def _finish_axes(axes: np.ndarray) -> None:
    axes[0].set_ylabel(r"$R_{\mathrm{tip}}/h_0$", fontsize=20, labelpad=6)
    axes[1].set_ylabel(r"$u^*=V_{\mathrm{tip}}/V_{\mathrm{TC}}$", fontsize=20, labelpad=6)
    axes[2].set_ylabel(r"$u^*=V_{\mathrm{tip}}/V_{\mathrm{TC}}$", fontsize=20, labelpad=6)
    for ax in axes[:2]:
        ax.set_xlabel(r"$t\sqrt{\sigma/(\rho h_0^3)}$", fontsize=19, labelpad=5)
        ax.axvline(0, color="0.75", lw=0.7, zorder=0)
        _style(ax)
    axes[2].set_xlabel(r"$R_{\mathrm{tip}}/h_0$", fontsize=19, labelpad=5)
    _style(axes[2])
    for ax in axes[1:]:
        ax.axhline(0, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=0)
    for ax, label in zip(axes, (r"$(a)$", r"$(b)$", r"$(c)$"), strict=True):
        ax.text(
            0.0,
            1.02,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            clip_on=False,
        )


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_campaign(
    cases: list[tuple[float, TipHistory]],
    output_dir: Path,
    *,
    use_tex: bool = True,
) -> list[Path]:
    """Render the aligned campaign plot and each individual case."""
    if not cases:
        raise ValueError("at least one case is required")
    ecs = [ec for ec, _ in cases]
    if any(not math.isfinite(ec) or ec < 0 for ec in ecs) or len(set(ecs)) != len(ecs):
        raise ValueError("Ec values must be finite, non-negative and unique")
    configure_matplotlib(use_tex=use_tex)
    output_dir.mkdir(parents=True, exist_ok=True)
    positive = sorted(ec for ec in ecs if ec > 0)
    ordered = sorted(cases, key=lambda item: item[0])
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.4))
    for ec, history in ordered:
        _draw_history(axes, history, colour=_colour(ec, positive), label=rf"$Ec={ec:g}$")
    _finish_axes(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(cases)),
        frameon=False,
        fontsize=14,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.2)
    common = output_dir / "ec_campaign"
    _save(fig, common)
    outputs.extend([common.with_suffix(".pdf"), common.with_suffix(".png")])

    for ec, history in ordered:
        fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.4))
        colour = _colour(ec, positive)
        _draw_history(axes, history, colour=colour, label=rf"$Ec={ec:g}$")
        _finish_axes(axes)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", frameon=False, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.2)
        # ``repr`` is the shortest round-trippable spelling, so distinct
        # binary64 values stay distinct without turning 0.1 into a long tag.
        tag = str(int(ec)) if ec.is_integer() else repr(ec)
        base = output_dir / f"ec_{tag.replace('+', '').replace('-', 'm').replace('.', 'p')}"
        _save(fig, base)
        outputs.extend([base.with_suffix(".pdf"), base.with_suffix(".png")])
    return outputs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("EC", "SCHEMA", "CONTROL"),
        required=True,
        help="repeatable case: Ec, newtonian|viscoelastic, control.txt",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--r0", type=float, default=1.0)
    return result


def main() -> None:
    args = parser().parse_args()
    cases = [
        (float(ec), load_control(path, schema=schema, r0=args.r0))
        for ec, schema, path in args.case
    ]
    render_campaign(cases, args.output_dir)


if __name__ == "__main__":
    main()
