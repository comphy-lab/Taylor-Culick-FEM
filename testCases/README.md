# testCases/

Software tests. From the repository root:

```bash
HWLOC_COMPONENTS=-gl python -m pytest testCases
```

`testCases/__init__.py` puts `src/` on the import path. The case, layout, runner and
provenance tests need only the standard library; a test that needs the pinned pyoomph
environment skips without it. The remaining modules test the solver library and run
short coarse solves. `HWLOC_COMPONENTS=-gl` stops hwloc probing graphics displays when
MPI starts, which can block on an unresponsive display; production launches set it too.
