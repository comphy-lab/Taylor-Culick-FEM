from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_POST = Path(__file__).parents[1] / "postProcess"
sys.path.insert(0, str(_POST))
_SPEC = importlib.util.spec_from_file_location(
    "plot_viscoelastic_campaign", _POST / "plot_viscoelastic_campaign.py"
)
assert _SPEC is not None and _SPEC.loader is not None
plotter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plotter)


def test_plot_generation_writes_common_and_individual_png_pdf(tmp_path: Path) -> None:
    time = np.linspace(0, 2, 21)
    cases = []
    for ec in (0.0, 0.1):
        history = plotter.TipHistory(
            time,
            1 + (0.2 - 0.05 * ec) * time,
            (0.2 - 0.05 * ec) * (1 - np.exp(-5 * time)),
            "newtonian" if ec == 0 else "viscoelastic",
            Path("synthetic"),
        )
        cases.append((ec, history))

    outputs = plotter.render_campaign(cases, tmp_path, use_tex=False)

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 1000 for path in outputs)
    assert (tmp_path / "ec_campaign.png").is_file()
    assert (tmp_path / "ec_0.png").is_file()
    assert (tmp_path / "ec_0p1.png").is_file()


def test_phase_trajectory_preserves_recoil_order_and_endpoints() -> None:
    history = plotter.TipHistory(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 1.7, 1.9]),
        np.array([0.0, 0.2, -0.1, 0.05]),
        "viscoelastic",
        Path("synthetic"),
    )
    figure, axes = plotter.plt.subplots(1, 3)

    plotter._draw_history(axes, history, colour="black", label=r"$Ec=1$")

    assert np.array_equal(axes[2].lines[0].get_xdata(), history.radius)
    assert np.array_equal(axes[2].lines[0].get_ydata(), history.ustar)
    assert np.array_equal(
        axes[2].collections[0].get_offsets(),
        np.array([[history.radius[0], history.ustar[0]], [history.radius[-1], history.ustar[-1]]]),
    )
    plotter.plt.close(figure)


def test_close_ec_values_get_distinct_output_names(tmp_path: Path) -> None:
    time = np.array([0.0, 1.0])
    cases = [
        (
            ec,
            plotter.TipHistory(
                time,
                np.array([1.0, 1.1]),
                np.array([0.0, 0.1]),
                "viscoelastic",
                Path("synthetic"),
            ),
        )
        for ec in (0.010000000000000002, 0.010000000000000004)
    ]

    outputs = plotter.render_campaign(cases, tmp_path, use_tex=False)

    case_stems = {path.stem for path in outputs if path.stem != "ec_campaign"}
    assert len(case_stems) == 2
