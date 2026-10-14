"""Elastic workspace: observables, vectorization, autodiff and nonlocal terms."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jitr.optical_potentials import kduq
from jitr.optical_potentials.potential_forms import (
    coulomb_charged_sphere,
    thomas_safe,
    woods_saxon_safe,
)
from jitr.reactions import Reaction
from jitr.xs.coupled import Potential
from jitr.xs.elastic import Workspace

GOLDENS = Path(__file__).parent / "characterization" / "data"
P48CA = Reaction(target=(48, 20), projectile=(1, 1), process="El")
ANGLES = np.linspace(0.05, np.pi - 0.05, 40)


@pytest.mark.parametrize("name", ["elastic_p48ca", "elastic_alpha44ca"])
def test_observables_reproduce_v3_goldens(name):
    d = np.load(GOLDENS / f"{name}.npz")
    proton = "p48" in name
    rxn = Reaction(
        target=(48, 20) if proton else (44, 20),
        projectile=(1, 1) if proton else (4, 2),
        process="El",
    )
    ws = Workspace(
        rxn,
        rxn.kinematics(d["kin_Elab"]),
        float(d["radius_fm"]),
        int(d["lmax"]),
        int(d["nbasis"]),
        angles=d["angles"],
    )
    potential = Potential(
        central=d["central"],
        spin_orbit=d["spin_orbit"] if proton else None,
        coulomb=d["coulomb"],
    )
    xs = ws.xs(potential)
    # deep backward minima (1e-5 mb/sr, from strongly cancelling partial-wave
    # sums) carry round-off at the 1e-13 mb/sr level
    np.testing.assert_allclose(xs.dsdo, np.atleast_2d(d["dsdo"]), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(xs.t, np.atleast_1d(d["t"]), rtol=1e-9)
    np.testing.assert_allclose(xs.rxn, np.atleast_1d(d["rxn"]), rtol=1e-9)
    if proton:
        np.testing.assert_allclose(xs.Ay, d["Ay"], atol=1e-6)
        np.testing.assert_allclose(xs.Q, d["Q"], atol=1e-6)
        S = ws.to_lj(ws.smatrix(potential))
        np.testing.assert_allclose(S[..., 0], d["splus"], atol=1e-11)
        np.testing.assert_allclose(S[:, 1:, 1], d["sminus"][:, 1:], atol=1e-11)
    else:
        np.testing.assert_allclose(ws.rutherford[0], d["rutherford"], rtol=1e-12)


def kd03(ws, kinematics):
    """KD03 for p + 48Ca at each workspace energy, shapes (NE, N)."""
    params = kduq.get_kd03((1, 1))
    model = kduq.KDUQ((1, 1))
    terms = [
        model.evaluate(ws.r, P48CA, P48CA.kinematics(E), *params)
        for E in np.atleast_1d(kinematics.Elab)
    ]
    central, spin_orbit, coulomb = (np.array(t) for t in zip(*terms, strict=True))
    return Potential(central=central, spin_orbit=spin_orbit, coulomb=coulomb)


def test_energy_grid_equals_single_energy_workspaces():
    """One workspace over five energies, with energy-dependent potentials,
    equals five single-energy workspaces."""
    E = np.array([5.0, 12.0, 30.0, 65.0, 100.0])
    kin = P48CA.kinematics(E)
    ws = Workspace(P48CA, kin, 12.0, 25, 50, angles=ANGLES)
    xs = ws.xs(kd03(ws, kin))
    for e, Ee in enumerate(E):
        kin1 = P48CA.kinematics(Ee)
        ws1 = Workspace(P48CA, kin1, 12.0, 25, 50, angles=ANGLES)
        xs1 = ws1.xs(kd03(ws1, kin1))
        for a, b in zip(xs, xs1, strict=True):
            np.testing.assert_allclose(a[e], b[0], rtol=1e-10, atol=1e-12)


def test_per_energy_angles_match_full_grid():
    E = np.array([10.0, 40.0])
    kin = P48CA.kinematics(E)
    ws = Workspace(P48CA, kin, 12.0, 20, 40, angles=ANGLES)
    S = ws.smatrix(kd03(ws, kin))
    ragged = ws.xs_from_smatrix(S, [ANGLES[::2], ANGLES[5:9]])
    full = ws.xs_from_smatrix(S)
    np.testing.assert_allclose(ragged[0].dsdo[0], full.dsdo[0, ::2], rtol=1e-12)
    np.testing.assert_allclose(ragged[1].Ay[0], full.Ay[1, 5:9], rtol=1e-12)
    np.testing.assert_allclose(ragged[1].rxn, full.rxn[1:], rtol=1e-12)


def test_spin_table():
    alpha = Reaction(target=(40, 20), projectile=(4, 2), process="El")
    ws = Workspace(alpha, alpha.kinematics(30.0), 10.0, 10, 30, angles=ANGLES)
    assert ws.s == 0 and ws.blocks.l.shape == (11, 1)
    central = -100 * woods_saxon_safe(ws.r, 4.5, 0.6)
    assert np.all(ws.xs(Potential(central=central)).Ay == 0)
    with pytest.raises(ValueError):
        ws.smatrix(Potential(central=central, spin_orbit=0 * central))
    triton = Reaction(target=(40, 20), projectile=(3, 1), process="El")
    assert Workspace(triton, triton.kinematics(30.0), 10.0, 10, 30).s == 0.5


def woods_saxon_xs(ws, depth, radius):
    r = ws.r
    central = (-depth - 10j) * woods_saxon_safe(r, radius, 0.65)
    spin_orbit = 6.0 * thomas_safe(r, radius, 0.6)
    return ws.xs(
        Potential(
            central=central,
            spin_orbit=spin_orbit,
            coulomb=coulomb_charged_sphere(r, 20, 4.5),
        )
    )


@pytest.mark.parametrize("argnum", [0, 1])
def test_gradient_through_potential_parameters(argnum):
    """d(observable)/d(depth or radius) by autodiff through the Woods-Saxon
    forms (evaluated on the numpy mesh) equals a central difference."""
    ws = Workspace(
        P48CA, P48CA.kinematics(np.array([20.0, 40.0])), 12.0, 20, 40, angles=ANGLES
    )

    def loss(depth, radius):
        xs = woods_saxon_xs(ws, depth, radius)
        return jnp.sum(xs.rxn) + jnp.sum(jnp.log(xs.dsdo))

    x = [50.0, 4.4]
    grad = jax.grad(loss, argnums=argnum)(*x)
    h = 1e-5
    up, down = list(x), list(x)
    up[argnum] += h
    down[argnum] -= h
    np.testing.assert_allclose(grad, (loss(*up) - loss(*down)) / (2 * h), rtol=1e-5)


def test_jit_of_user_function_equals_eager():
    ws = Workspace(P48CA, P48CA.kinematics(30.0), 12.0, 20, 40, angles=ANGLES)
    eager = woods_saxon_xs(ws, 50.0, 4.4).dsdo
    jitted = jax.jit(lambda d, R: woods_saxon_xs(ws, d, R).dsdo)(50.0, 4.4)
    # fused vs. op-by-op evaluation differ at round-off
    np.testing.assert_allclose(jitted, eager, rtol=1e-10)


def neutron_workspace():
    n = Reaction(target=(40, 20), projectile=(1, 0), process="El")
    return Workspace(
        n, n.kinematics(np.array([5.0, 20.0])), 12.0, 12, 40, angles=ANGLES
    )


def gaussian_kernel(r, depth, width=0.9):
    """Perey-Buck-like nonlocality times a Woods-Saxon at the mean radius."""
    rr = (r[:, None] + r[None, :]) / 2
    shape = np.exp(-(((r[:, None] - r[None, :]) / width) ** 2)) / (
        np.sqrt(np.pi) * width
    )
    return depth * woods_saxon_safe(rr, 4.2, 0.6) * shape


def test_absorptive_nonlocal_potential_reduces_flux():
    ws = neutron_workspace()
    S = np.asarray(
        ws.smatrix(Potential(nonlocal_central=gaussian_kernel(ws.r, -60 - 12j)))
    )
    assert np.all(np.abs(S) <= 1 + 1e-12)
    assert np.any(np.abs(S) < 0.999)


def test_nonlocal_diagonal_kernel_equals_local_potential():
    """A kernel v(r) delta(r - r') on the mesh, v(r_n) delta_nm / (a w_n), is
    the local potential v."""
    ws = neutron_workspace()
    m = ws.solver.mesh
    v = (-45 - 8j) * woods_saxon_safe(ws.r, 4.2, 0.6)
    so = 5.0 * thomas_safe(ws.r, 4.0, 0.6)
    local = ws.smatrix(Potential(central=v, spin_orbit=so))
    kernel = np.diag(v / (m.a * m.w))
    nonlocal_ = ws.smatrix(Potential(nonlocal_central=kernel, spin_orbit=so))
    np.testing.assert_allclose(nonlocal_, local, atol=1e-12)
    mixed = ws.smatrix(
        Potential(
            central=v / 2,
            nonlocal_central=kernel / 2,
            nonlocal_spin_orbit=np.diag(so / (m.a * m.w)),
        )
    )
    np.testing.assert_allclose(mixed, local, atol=1e-12)


def test_l_dependent_nonlocal_kernel():
    """Identical l slices equal the l-independent kernel; an l-dependent
    depth changes only the corresponding partial waves."""
    ws = neutron_workspace()
    K = gaussian_kernel(ws.r, -60 - 12j)
    L = ws.lmax + 1
    S = ws.smatrix(Potential(nonlocal_central=K))
    stack = np.broadcast_to(K, (L, 1, *K.shape))
    np.testing.assert_allclose(
        ws.smatrix(Potential(nonlocal_central=stack)), S, atol=1e-13
    )
    scaled = stack * np.where(np.arange(L) == 3, 1.2, 1.0)[:, None, None, None]
    S3 = np.asarray(ws.smatrix(Potential(nonlocal_central=scaled)))
    changed = np.any(np.abs(S3 - np.asarray(S)) > 1e-10, axis=(0, 2, 3))
    np.testing.assert_array_equal(changed, ws.blocks.l[:, 0] == 3)


def test_gradient_through_nonlocal_kernel():
    ws = neutron_workspace()
    K = gaussian_kernel(ws.r, -60 - 12j)

    def loss(scale):
        return jnp.sum(ws.xs(Potential(nonlocal_central=scale * K)).rxn)

    h = 1e-6
    np.testing.assert_allclose(
        jax.grad(loss)(1.0), (loss(1 + h) - loss(1 - h)) / (2 * h), rtol=1e-6
    )


@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="no GPU"
)
def test_gpu_matches_cpu():
    kin = P48CA.kinematics(np.array([20.0, 60.0]))
    ws = Workspace(P48CA, kin, 12.0, 25, 50, angles=ANGLES)
    potential = kd03(ws, kin)
    gpu = ws.xs(potential).dsdo
    with jax.default_device(jax.devices("cpu")[0]):
        cpu = ws.xs(potential).dsdo
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10)


def test_deuteron_is_spin_one():
    d = Reaction(target=(40, 20), projectile=(2, 1), process="El")
    ws = Workspace(d, d.kinematics(np.array([20.0, 50.0])), 12.0, 16, 40, angles=ANGLES)
    assert ws.s == 1.0
    r = ws.r
    xs = ws.xs(
        Potential(
            central=(-90 - 15j) * woods_saxon_safe(r, 4.3, 0.75),
            spin_orbit=3.0 * thomas_safe(r, 4.0, 0.6),
            coulomb=coulomb_charged_sphere(r, 20, 4.4),
        )
    )
    assert xs.T.shape == (2, 5, ANGLES.size)  # iT11 ... T22
    assert np.abs(np.real(1j * xs.T[:, 1])).max() > 1e-2  # vector analyzing power
    assert np.all(xs.rxn > 0) and np.all(np.isfinite(xs.dsdo))


def test_identical_particles_are_rejected():
    alpha = Reaction(target=(4, 2), projectile=(4, 2), process="El")
    with pytest.raises(NotImplementedError, match="identical"):
        Workspace(alpha, alpha.kinematics(20.0), 10.0, 10, 30)


def test_to_lj_rejects_spin_above_half():
    d = Reaction(target=(40, 20), projectile=(2, 1), process="El")
    ws = Workspace(d, d.kinematics(20.0), 12.0, 8, 30, angles=ANGLES)
    S = ws.smatrix(Potential(central=-90 * woods_saxon_safe(ws.r, 4.3, 0.75)))
    with pytest.raises(ValueError, match="spin"):
        ws.to_lj(S)


def test_per_energy_angles_reuse_the_geometry(monkeypatch):
    """Ragged angle lists rebuild only the angle-dependent tables."""
    from jitr.xs import amplitudes

    kin = P48CA.kinematics(np.array([10.0, 40.0]))
    ws = Workspace(P48CA, kin, 12.0, 20, 40, angles=ANGLES)
    S = ws.smatrix(kd03(ws, kin))
    calls = []
    original = amplitudes.transition_geometry
    monkeypatch.setattr(
        amplitudes, "transition_geometry", lambda *a: calls.append(a) or original(*a)
    )
    ws.xs_from_smatrix(S, [ANGLES[::2], ANGLES[5:9]])
    ws.xs_from_smatrix(S, ANGLES[::3])
    assert calls == []


def test_angle_list_must_match_energies_and_rutherford_needs_angles():
    kin = P48CA.kinematics(np.array([10.0, 40.0]))
    ws = Workspace(P48CA, kin, 12.0, 20, 40, angles=ANGLES)
    S = ws.smatrix(kd03(ws, kin))
    with pytest.raises(ValueError, match="one angle array per energy"):
        ws.xs_from_smatrix(S, [ANGLES])
    assert Workspace(P48CA, kin, 12.0, 20, 40).rutherford is None


def test_integrated_cross_sections_need_no_angles():
    kin = P48CA.kinematics(np.array([10.0, 40.0]))
    with_angles = Workspace(P48CA, kin, 12.0, 20, 40, angles=ANGLES)
    without = Workspace(P48CA, kin, 12.0, 20, 40)
    a, b = with_angles.xs(kd03(with_angles, kin)), without.xs(kd03(without, kin))
    np.testing.assert_allclose(b.rxn, a.rxn, rtol=1e-12)
    np.testing.assert_allclose(b.t, a.t, rtol=1e-12)
    assert b.dsdo.shape == (2, 0)
