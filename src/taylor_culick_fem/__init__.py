"""Taylor-Culick sheet retraction in a tip co-moving frame (pyoomph/oomph-lib FEM).

Importing this package sets runtime environment defaults that must be in place
*before* pyoomph (and therefore MPI) is imported. Case scripts import this
package first for that reason.
"""

import os

__version__ = "0.1.0"


def _set_runtime_defaults() -> None:
    """Environment fixes that must precede the first pyoomph/MPI import.

    ``HWLOC_COMPONENTS=-gl`` disables hwloc's OpenGL/X11 topology backend.

    hwloc enumerates NVIDIA displays by connecting to ``/tmp/.X11-unix/X0``,
    ``X1``, ... directly; it does not consult ``DISPLAY``, so unsetting that
    does not help. On a machine running a GNOME/Wayland session the compositor
    accepts the connection but never completes the X handshake for an
    unauthorised client, and hwloc then blocks forever in ``poll(..., -1)``.
    Because OpenMPI initialises hwloc during ``MPI_Init``, the symptom is that
    ``import pyoomph`` hangs indefinitely with no error -- and that
    ``mpirun -n 1 python ...`` appears to "fix" it.

    A solver has no reason to probe the display for topology, so the component
    is simply turned off. Set ``HWLOC_COMPONENTS`` yourself to override.
    """
    os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
    # Never let a plotting backend try to open a display on a compute host.
    os.environ.setdefault("MPLBACKEND", "Agg")


_set_runtime_defaults()
