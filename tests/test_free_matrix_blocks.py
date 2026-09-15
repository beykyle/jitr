"""The uncoupled free-matrix blocks equal the blocks of the coupled matrix, without
keeping the coupled matrix alive; the vectorised Legendre kinetic matrix equals the
element-wise assembly."""

import numpy as np

from jitr.rmatrix import Solver
from jitr.utils import block


def _element_wise_kinetic(q, a, l):
    F = np.zeros((q.nbasis, q.nbasis), dtype=np.complex128)
    for n in range(1, q.nbasis + 1):
        for m in range(n, q.nbasis + 1):
            F[n - 1, m - 1] = q.kinetic_operator_element(n, m, a, l)
    return F + np.triu(F, k=1).T


def test_legendre_kinetic_matrix_matches_element_wise_assembly():
    for nbasis in (5, 20, 45):
        q = Solver(nbasis).kernel.quadrature
        for a, l in ((3.7, 0), (37.3, 7), (120.0, 40)):
            expected = _element_wise_kinetic(q, a, l)
            got = q.kinetic_matrix(a, l)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_uncoupled_free_matrix_blocks_match_coupled_matrix():
    for nbasis in (10, 30):
        solver = Solver(nbasis)
        a = 25.0
        l = np.arange(0, 12)
        E = np.linspace(1.0, 0.7, l.size)
        mu = np.linspace(1.0, 1.1, l.size)
        coupled = solver.free_matrix(a, l, E=E, mu=mu, coupled=True)
        blocks = solver.free_matrix(a, l, E=E, mu=mu, coupled=False)
        assert len(blocks) == l.size
        for i, b in enumerate(blocks):
            expected = block(coupled, (i, i), (nbasis, nbasis))
            np.testing.assert_allclose(b, expected, rtol=1e-12, atol=0)
            assert b.shape == (nbasis, nbasis)
            # a block owns its memory (nbasis**2), not the coupled matrix
            assert b.nbytes == nbasis * nbasis * 16
            base = b.base
            while base is not None and hasattr(base, "shape"):
                assert base.size <= nbasis * nbasis
                base = getattr(base, "base", None)


def test_uncoupled_free_matrix_defaults_are_unit_energy_and_mass():
    solver = Solver(8)
    l = np.array([0, 1, 2])
    default = solver.free_matrix(4.0, l, coupled=False)
    explicit = solver.free_matrix(4.0, l, E=np.ones(3), mu=np.ones(3), coupled=False)
    for d, e in zip(default, explicit):
        np.testing.assert_array_equal(d, e)
