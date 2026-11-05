# ruff: noqa -- kept exactly as run; lint fixes would change the recorded prototype.
"""Retained-grid continuation of a full-domain equilibrium after a physical-state restart gate.

Restores a saved full-domain equilibrium (state.npz) on its own retained reference grid (grids.npz) without
interpolation: the saved total displacement becomes a pinned Q2 background at identical nodes, and the unknown
displacement is the increment on it (the pinned background translates the Q2 trial space; exported states carry
the total displacement). The restored state is re-solved and screened (sampled J error <= 0.03, negative leading
returned eigenvalue, residual below the case tolerance), saved, rebuilt from the saved file and re-solved again;
this restart gate requires R, H and J to agree to 1e-8. With --target-ec the branch is then continued at fixed
log-Ec steps of at most 0.002 down to the target, moving the increment into the background before every step,
with a J and residual screen at every step and an eigenvalue screen every fifth step and at the end.

Options --gamma and --beta override the case's volume penalty and cap stiffening; --ramp-beta ramps the cap
stiffening from the case value to --beta first. --cpus must equal the CPU list the launcher bound the process to;
every thread's mask is checked at start-up. The output folder must lie inside the working directory.
usage: equilibrium_retained_checks-v6.py --case <case.json> --grids <grids.npz> --state <state.npz> --out <new folder>
       --ndof <unknown count> --cpus <list> [--target-ec <Ec>] [--gamma <g>] [--beta <b> [--ramp-beta]]
This is a diagnostic continuation tool, not a production runner.
"""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=Path, required=True)
parser.add_argument('--grids', type=Path, required=True)
parser.add_argument('--state', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--ndof', type=int, required=True)
parser.add_argument('--gamma', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--ramp-beta', action='store_true')
parser.add_argument('--target-ec', type=float)  # continue the branch down to this Ec
parser.add_argument('--cpus', required=True)  # the CPU list the launcher bound this process to
args = parser.parse_args()
CPUS = os.sched_getaffinity(0)
SOURCE = Path(__file__).resolve().parents[2]
PARENT = args.grids.parent
ROOT = Path.cwd().resolve()
OUT = args.out.resolve()
assert OUT.is_relative_to(ROOT)
CASE = args.case.resolve(strict=True)
sys.path.insert(0, str(SOURCE / 'src'))

import numpy as np
from pyoomph.generic.jit_cache import set_enabled
import taylor_culick_fem.elastocapillary_equilibrium as equation_module
from taylor_culick_fem.equilibrium_branch import SegmentSettings, _point
from taylor_culick_fem.equilibrium_geometry import FieldMap, incompressibility_error
from taylor_culick_fem.run_contract import enforce_launcher_affinity

set_enabled(False)
enforce_launcher_affinity(CPUS)
def cpu_set(text):
    out = set()
    for part in text.split(','):
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return out
assert CPUS == cpu_set(args.cpus)
base_var = equation_module.var
base_equation = equation_module.NeoHookeanDisplacementEquations
base_problem = equation_module.ElastocapillaryEquilibriumProblem


def physical_var(name, *args, **kwargs):
    value = base_var(name, *args, **kwargs)
    if name == 'displacement':
        value = value + base_var('background', *args, **kwargs)
    return value


class IncrementEquation(base_equation):
    def define_fields(self):
        super().define_fields()
        self.define_vector_field('background', self.displacement_space)

    def define_residuals(self):
        super().define_residuals()
        self.set_Dirichlet_condition('background_x', True)
        self.set_Dirichlet_condition('background_y', True)


class IncrementProblem(base_problem):
    def deformed_surface(self):
        mesh, ix, iy = self._displacement('solid/interface')
        indices = mesh.get_nodal_field_indices()
        bx, by = indices['background_x'], indices['background_y']
        return np.asarray([(nd.x(0) + nd.value(ix) + nd.value(bx),
                            nd.x(1) + nd.value(iy) + nd.value(by), nd.x(0), nd.x(1))
                           for nd in mesh.nodes()])

    def rim_height(self):
        mesh, _, iy = self._displacement('solid/interface')
        by = mesh.get_nodal_field_indices()['background_y']
        maximum = -math.inf
        for element in mesh.elements():
            points = []
            for k in range(element.nnode()):
                node = element.node_pt(k)
                points.append((float(element.local_coordinate_of_node(k)[0]),
                               float(node.x(1) + node.value(iy) + node.value(by))))
            maximum = max(maximum, *(value for _, value in points))
            if len(points) >= 3:
                s = np.asarray([p[0] for p in points])
                h = np.asarray([p[1] for p in points])
                c, linear, quadratic = np.polynomial.polynomial.polyfit(s, h, 2)
                if quadratic < 0:
                    stationary = -linear / (2 * quadratic)
                    if s.min() <= stationary <= s.max():
                        maximum = max(maximum, c + linear * stationary + quadratic * stationary**2)
        return 2 * maximum

    def far_edge_displacement(self):
        mesh, ix, _ = self._displacement('solid/far')
        bx = mesh.get_nodal_field_indices()['background_x']
        node = min(mesh.nodes(), key=lambda nd: nd.x(1))
        return float(node.value(ix) + node.value(bx))


equation_module.var = physical_var
equation_module.NeoHookeanDisplacementEquations = IncrementEquation
case = json.loads(CASE.read_text())
original_beta = float(case['regularisation']['cap_beta'])
if args.ramp_beta:
    assert args.beta in (.0005, .002), 'Only the two fixed beta targets are authorised'
if args.gamma is not None:
    case['regularisation']['volume_penalty'] = args.gamma
if args.beta is not None:
    case['regularisation']['cap_beta'] = args.beta
settings = SegmentSettings.from_case(case)
OUT.mkdir(exist_ok=False)
thread_masks = {}
for task in Path('/proc/self/task').iterdir():
    status = (task / 'status').read_text()
    mask = next(line.split(':', 1)[1].strip() for line in status.splitlines()
                if line.startswith('Cpus_allowed_list:'))
    assert cpu_set(mask) == cpu_set(args.cpus), (task.name, mask)
    thread_masks[task.name] = mask
(OUT / 'startup-proof.json').write_text(json.dumps({
    'checked_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
    'uid': os.getuid(), 'cwd': str(Path.cwd().resolve()),
    'cgroup': Path('/proc/self/cgroup').read_text(), 'thread_masks': thread_masks,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'case_sha256': hashlib.sha256(CASE.read_bytes()).hexdigest(),
}, indent=2) + '\n')


def physical_map(problem):
    fmap = FieldMap.from_problem(problem)
    mesh = problem.get_mesh('solid')
    indices = mesh.get_nodal_field_indices()
    bx, by = indices['background_x'], indices['background_y']
    background = []
    for element in mesh.elements():
        background.append([[element.node_pt(k).value(bx), element.node_pt(k).value(by)]
                           for k in range(element.nnode())])
    return FieldMap(fmap.Xe, fmap.ue + np.asarray(background), fmap.ij)


def build(state_path, folder, *, seed_beta=None):
    with np.load(state_path) as saved:
        old = FieldMap(saved['Xe'], saved['ue'], saved['ij'])
        log_ec = float(saved['log_ec'])
    p = IncrementProblem()
    p.ignore_command_line = True
    p.set_output_directory(str(folder))
    p.r0 = settings.r0
    p.surface_tension = settings.surface_tension
    with np.load(args.grids) as grids:
        p.grids = dict(grids)
    p.quadratic_grids = True
    p.length = settings.length
    p.cap_stiffening = settings.cap_beta if seed_beta is None else seed_beta
    p.volume_penalty = settings.volume_penalty
    p.newton_solver_tolerance = settings.newton_tolerance
    p.set_linear_solver('superlu')
    p.set_eigensolver('scipy')
    p.initialise()
    p.get_global_parameter('log_Ec').value = log_ec
    mesh = p.get_mesh('solid')
    indices = mesh.get_nodal_field_indices()
    # This is a retained grid: restore coefficients at identical Q2 nodes.
    # Interpolating a crushed element can introduce material-state error.
    nodal_displacements = {}
    for coordinates, displacements in zip(old.Xe, old.ue):
        for coordinate, displacement in zip(coordinates, displacements):
            key = tuple(float(value) for value in coordinate)
            previous = nodal_displacements.setdefault(key, displacement)
            assert np.array_equal(previous, displacement), 'Saved shared-node values disagree'
    for node in mesh.nodes():
        key = (float(node.x(0)), float(node.x(1)))
        assert key in nodal_displacements, 'Retained reference node is missing'
        u = nodal_displacements[key]
        assert node.x(1) > 1e-14 or abs(float(u[1])) <= 1e-12
        node.set_value(indices['background_x'], float(u[0]))
        node.set_value(indices['background_y'], float(u[1]) if node.x(1) > 1e-14 else 0.0)
        node.set_value(indices['displacement_x'], 0.0)
        node.set_value(indices['displacement_y'], 0.0)
    assert p.ndof() == args.ndof, 'Physical unknown count changed'
    restored = physical_map(p)
    assert np.array_equal(restored.Xe, old.Xe), 'Retained reference nodes differ'
    assert np.array_equal(restored.ij, old.ij), 'Retained connectivity differs'
    delta = float(np.max(np.abs(restored.ue-old.ue)))
    assert delta <= 1e-12, 'Restoration changed material displacement'
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder)/'restore-proof.json').write_text(json.dumps({
        'reference_nodes_exact': True, 'connectivity_exact': True,
        'max_physical_displacement_change': delta,
        'initial_R': p.hole_radius(), 'initial_H': p.rim_height(),
        'initial_J_err': incompressibility_error(restored),
        'input_state_sha256': hashlib.sha256(Path(state_path).read_bytes()).hexdigest(),
        'grids_sha256': hashlib.sha256(args.grids.read_bytes()).hexdigest(),
        'ndof': p.ndof()}, indent=2)+'\n')
    return p


def rebase(problem):
    """Translate coefficients at identical nodes without changing pressure."""
    mesh = problem.get_mesh('solid')
    indices = mesh.get_nodal_field_indices()
    before = physical_map(problem)
    for node in mesh.nodes():
        for axis in ('x', 'y'):
            increment = indices['displacement_' + axis]
            background = indices['background_' + axis]
            total = node.value(increment) + node.value(background)
            node.set_value(background, total)
            node.set_value(increment, 0.0)
    after = physical_map(problem)
    assert np.array_equal(before.Xe, after.Xe)
    assert np.array_equal(before.ue, after.ue), 'Rebase changed physical nodal displacement'
    assert problem.ndof() == args.ndof


def save_physical(problem, path):
    physical_map(problem).save(path, log_ec=problem.get_global_parameter('log_Ec').value,
                               ds=.002, cap_beta=settings.cap_beta)


def finite_values(*values):
    """Reject non-finite observables rather than relying on ordered comparisons."""
    return all(math.isfinite(float(value)) for value in values)



def store_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def screen_point(point):
    assert finite_values(point['R'], point['H'], point['J_err'],
                         point['leading_eigenvalue'], *point['eigenvalues'])
    assert point['J_err'] <= .03, 'Fixed sampled-J quality screen failed'
    assert point['leading_eigenvalue'] < 0, 'Returned spectral quality screen failed'
    assert float(max(abs(problem.get_residuals()))) < settings.newton_tolerance

store_json('inputs.json', {'source_commit': 'df351e965fd80389006880b599685c200297567d',
    'pyoomph_commit': 'a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618',
    'case': {'path': str(CASE), 'sha256': hashlib.sha256(CASE.read_bytes()).hexdigest()},
    'state': {'path': str(args.state), 'sha256': hashlib.sha256(args.state.read_bytes()).hexdigest()},
    'grids': {'path': str(args.grids), 'sha256': hashlib.sha256(args.grids.read_bytes()).hexdigest()},
    'effective_case': case, 'reserved_cpus': sorted(CPUS)})
problem = build(args.state, OUT / 'seed',
                seed_beta=original_beta if args.ramp_beta else None)
try:
    problem.solve(max_newton_iterations=settings.max_newton)
    first = _point(problem, settings, physical_map(problem))
    first['max_residual'] = float(max(abs(problem.get_residuals())))
    store_json('seed-point.json', first)
    assert finite_values(first['R'], first['H'], first['J_err'], first['max_residual'])
    screen_point(first)
    if args.ramp_beta:
        beta_parameter = problem.get_global_parameter('cap_beta')
        nsteps = math.ceil(abs(math.log(args.beta / original_beta)) / .02)
        assert nsteps <= 35
        for index, value in enumerate(np.geomspace(original_beta, args.beta, nsteps + 1)[1:], 1):
            rebase(problem)
            beta_parameter.value = float(value)
            problem.solve(max_newton_iterations=settings.max_newton)
            record = {'step': index, 'beta': float(value), 'R': problem.hole_radius(),
                      'H': problem.rim_height(), 'J_err': incompressibility_error(physical_map(problem)),
                      'max_residual': float(max(abs(problem.get_residuals())))}
            assert finite_values(*record.values())
            assert record['J_err'] <= .03
            assert record['max_residual'] < settings.newton_tolerance
            with (OUT/'beta-progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        first = _point(problem, settings, physical_map(problem))
        first['max_residual'] = float(max(abs(problem.get_residuals())))
        screen_point(first)
        store_json('target-point.json', first)
    save_physical(problem, OUT / 'physical-seed.npz')
    restarted = build(OUT / 'physical-seed.npz', OUT / 'restart')
    restarted.solve(max_newton_iterations=settings.max_newton)
    second = _point(restarted, settings, physical_map(restarted))
    restart = {'relative_R_change': second['R']/first['R']-1,
               'relative_H_change': second['H']/first['H']-1,
               'J_change': second['J_err']-first['J_err'],
               'max_residual': float(max(abs(restarted.get_residuals()))),
               'seed': first, 'restart': second}
    store_json('restart-gate.json', restart)
    assert finite_values(*[restart[k] for k in ['relative_R_change','relative_H_change','J_change','max_residual']])
    assert abs(restart['relative_R_change']) < 1e-8
    assert abs(restart['relative_H_change']) < 1e-8
    assert abs(restart['J_change']) < 1e-8
    assert restart['max_residual'] < settings.newton_tolerance
    problem = restarted
    screen_point(second)
    points = [second]
    if args.target_ec is not None:
        parameter = problem.get_global_parameter('log_Ec')
        target = math.log(args.target_ec)
        assert target < parameter.value <= math.log(.35)
        steps = math.ceil(abs(target-parameter.value)/.002)
        for index, value in enumerate(np.linspace(parameter.value,target,steps+1)[1:], 1):
            rebase(problem)
            parameter.value = float(value)
            problem.solve(max_newton_iterations=settings.max_newton)
            fmap = physical_map(problem)
            summary = {'step': index, 'Ec': math.exp(value),
                       'R': problem.hole_radius(), 'H': problem.rim_height(),
                       'J_err': incompressibility_error(fmap),
                       'max_residual': float(max(abs(problem.get_residuals())))}
            print('CONTINUATION', json.dumps(summary), flush=True)
            with (OUT/'progress.jsonl').open('a') as f:
                f.write(json.dumps(summary,allow_nan=False)+'\n')
            assert finite_values(*summary.values())
            assert summary['J_err'] <= .03
            assert summary['max_residual'] < settings.newton_tolerance
            save_physical(problem, OUT/'last-converged-state.npz')
            if index % 5 == 0 or index == steps:
                point = _point(problem, settings, fmap)
                point['max_residual'] = summary['max_residual']
                points.append(point)
                store_json('points.json',points)
                assert finite_values(point['leading_eigenvalue'],*point['eigenvalues'])
                assert point['leading_eigenvalue'] < 0
                save_physical(problem, OUT/'last-screened-state.npz')
    else:
        save_physical(problem, OUT/'last-screened-state.npz')
    np.savetxt(OUT/'surface.txt',problem.deformed_surface(),header='r z R0 Z0')
    store_json('outcome.json', {'status': 'bounded diagnostic passed', 'points': points,
        'endpoint': points[-1], 'spectrum_scope': 'returned near-zero m0 reflection-even modes only',
        'limits': 'No global stability or continuum regularisation certificate.'})
    print('ENDPOINT', json.dumps(points[-1]),flush=True)
except Exception as error:
    store_json('outcome.json', {'status': 'failed diagnostic', 'type': type(error).__name__,
        'error': str(error), 'Ec': problem.elastocapillary_number,
        'residual_history': list(problem.get_last_residual_convergence())})
    raise

