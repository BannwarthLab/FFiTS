"""Analytic Fortran Hessians vs finite differences of the analytic gradients.

The terms are tested away from their reference values (r != r0, theta != theta0,
phi != phi0): every Hessian part that is multiplied by (theta - theta0) etc.
vanishes at the reference geometry, so a test there cannot catch errors in it.
This is exactly the situation of the TS FF optimization (reference values are
a mix of reactant and product values), while during fitting theta == theta0.

Regression tests for:
- the packed-index mapping j = floor((sqrt(8*real(m)-7)+1)/2), which was
  miscompiled by ifx (-fp-model fast replaced sqrt by an approximate rsqrt) and
  gave wrong angle/dihedral Hessians in the 0.9.1 ifx build,
- d2theta/dcos^2 being -c/(1-c^2)**0.75 instead of -c/(1-c^2)**1.5.
"""

import os

import numpy as np
import pytest

import ffits.forcefield.fortran.fortran_forcefield as F
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import angstrom2bohr, convert_xyz_to_fortranstyle
from ffits.forcefield.python_interface.ff_energy import (
    complete_gradient,
    complete_hessian,
)
from ffits.io.reader import readin_xyz

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "small_single_molecule"
)

# geometries in bohr, rows = atoms (generic, no symmetry)
GEO_BOND = np.array([[0.0, 0.0, 0.0], [1.9, 0.6, -0.4]])
GEO_ANGLE = np.array([[1.6, 0.2, -0.1], [0.0, 0.0, 0.0], [-0.5, 1.7, 0.3]])
GEO_DIHEDRAL = np.array(
    [[1.4, 0.9, 0.2], [0.0, 0.0, 0.0], [0.3, -1.5, 0.1], [1.2, -2.1, 1.3]]
)


def _fd_hessian(grad, geo, h=1e-5):
    """Central finite differences of ``grad`` (geo rows = atoms) -> (3n, 3n)."""
    n = geo.size
    fd = np.zeros((n, n))
    for k in range(n):
        gp, gm = geo.copy(), geo.copy()
        gp.flat[k] += h
        gm.flat[k] -= h
        fd[:, k] = (grad(gp) - grad(gm)) / (2 * h)
    return fd


def _single_term(module, name, geo, atoms, ref, k=1.0):
    """Gradient/Hessian callables of one Fortran term (geo rows = atoms, atoms 1-based)."""
    mod = getattr(F, module)
    atoms = np.array(atoms, dtype=np.int32)
    n = geo.size

    def grad(g):
        out = np.zeros(n)
        getattr(mod, f"get_single_{name}_gradient")(
            np.asfortranarray(g.T), atoms, ref, k, out
        )
        return out

    def hess(g):
        out = np.zeros((n, n), order="F")
        getattr(mod, f"get_single_{name}_hessian")(
            np.asfortranarray(g.T), atoms, ref, k, out
        )
        return out

    return grad, hess


def _angle(geo):
    u, v = geo[0] - geo[1], geo[2] - geo[1]
    return np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


def _dihedral(geo):
    b1, b2, b3 = geo[1] - geo[0], geo[2] - geo[1], geo[3] - geo[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return np.arctan2(np.cross(n1, n2) @ b2 / np.linalg.norm(b2), n1 @ n2)


def _packed_upper(mat):
    """Upper triangle in the Fortran packing m = (j-1)*j/2 + i, i <= j."""
    n = mat.shape[0]
    return np.array([mat[i, j] for j in range(n) for i in range(j + 1)])


# ---------------------------------------------------------------------------
# single terms away from their reference value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dr", [0.0, 0.3, -0.5])
def test_bond_hessian_matches_fd(dr):
    r = np.linalg.norm(GEO_BOND[1] - GEO_BOND[0])
    grad, hess = _single_term("bond_derivatives", "bond", GEO_BOND, [1, 2], r + dr)
    np.testing.assert_allclose(hess(GEO_BOND), _fd_hessian(grad, GEO_BOND), atol=1e-7)


@pytest.mark.parametrize("sigma", [1.5, 2.5])
def test_lj_hessian_matches_fd(sigma):
    grad, hess = _single_term("lj_derivatives", "lj", GEO_BOND, [1, 2], sigma, 1e-3)
    np.testing.assert_allclose(hess(GEO_BOND), _fd_hessian(grad, GEO_BOND), atol=1e-7)


@pytest.mark.parametrize("dtheta_deg", [0.0, 10.0, 30.0, 60.0, -40.0])
def test_angle_hessian_matches_fd(dtheta_deg):
    theta0 = _angle(GEO_ANGLE) + np.radians(dtheta_deg)
    grad, hess = _single_term(
        "angle_derivatives", "angle", GEO_ANGLE, [1, 2, 3], theta0
    )
    np.testing.assert_allclose(hess(GEO_ANGLE), _fd_hessian(grad, GEO_ANGLE), atol=1e-7)


@pytest.mark.parametrize("dphi_deg", [0.0, 17.0, 57.0, 120.0, -90.0])
def test_dihedral_hessian_matches_fd(dphi_deg):
    phi0 = _dihedral(GEO_DIHEDRAL) + np.radians(dphi_deg)
    grad, hess = _single_term(
        "dihedral_derivatives", "dihedral", GEO_DIHEDRAL, [1, 2, 3, 4], phi0
    )
    np.testing.assert_allclose(
        hess(GEO_DIHEDRAL), _fd_hessian(grad, GEO_DIHEDRAL), atol=1e-7
    )


# exactly collinear (alkynes, nitriles, CO2): 1/sin(theta) used to give NaN
GEO_LINEAR = np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [4.4, 0.0, 0.0]])


def test_linear_angle_hessian_matches_fd():
    """theta0 = theta = pi (FF fitted on a linear structure): finite and exact."""
    grad, hess = _single_term(
        "angle_derivatives", "angle", GEO_LINEAR, [1, 2, 3], np.pi
    )
    assert not grad(GEO_LINEAR).any()
    np.testing.assert_allclose(
        hess(GEO_LINEAR), _fd_hessian(grad, GEO_LINEAR), atol=1e-7
    )


def test_linear_angle_with_bent_reference_is_finite():
    grad, hess = _single_term(
        "angle_derivatives", "angle", GEO_LINEAR, [1, 2, 3], np.radians(170.0)
    )
    assert np.isfinite(grad(GEO_LINEAR)).all()
    assert np.isfinite(hess(GEO_LINEAR)).all()


def test_dihedral_on_linear_angle_is_skipped():
    geo = np.vstack([GEO_LINEAR, [[5.4, 1.5, 0.3]]])
    grad, hess = _single_term(
        "dihedral_derivatives", "dihedral", geo, [1, 2, 3, 4], 0.5
    )
    assert not grad(geo).any()
    assert not hess(geo).any()


# ---------------------------------------------------------------------------
# building blocks: second derivatives of cos(theta), cos/sin(phi)
# (covers get_theta_hessian / get_phi_hessian and the c-remap loops)
# ---------------------------------------------------------------------------


def test_theta_second_derivatives_match_fd():
    def first(g):
        return F.angle_derivatives.get_theta_derivatives(
            np.asfortranarray(g.T), 1, 2, 3
        )

    d2cos = first(GEO_ANGLE)[1]
    fd = _fd_hessian(lambda g: first(g)[0], GEO_ANGLE)
    np.testing.assert_allclose(d2cos, _packed_upper(fd), atol=1e-8)


def test_phi_second_derivatives_match_fd():
    def first(g):
        return F.dihedral_derivatives.get_phi_derivatives(
            np.asfortranarray(g.T), 1, 2, 3, 4
        )

    _, d2cos, _, d2sin, *_ = first(GEO_DIHEDRAL)
    fd_cos = _fd_hessian(lambda g: first(g)[0], GEO_DIHEDRAL)
    fd_sin = _fd_hessian(lambda g: first(g)[2], GEO_DIHEDRAL)
    np.testing.assert_allclose(d2cos, _packed_upper(fd_cos), atol=1e-8)
    np.testing.assert_allclose(d2sin, _packed_upper(fd_sin), atol=1e-8)


# ---------------------------------------------------------------------------
# packed-index scatter: every (i, j), i <= j, filled exactly once
# ---------------------------------------------------------------------------


def _expected_index_matrix(n):
    exp = np.zeros((n, n))
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            exp[i - 1, j - 1] = exp[j - 1, i - 1] = (j - 1) * j // 2 + i
    return exp


@pytest.mark.parametrize(
    "name, n, call",
    [
        # first derivatives zero, d2(m) = m -> H(i,j) = prefactor * m
        (
            "bond",
            6,
            lambda d2, x, H: F.bond_derivatives.build_bond_hessian(
                1.0, 2.0, np.zeros(6), d2, 1.0, x, H
            ),
        ),
        (
            "lj",
            6,
            lambda d2, x, H: F.lj_derivatives.build_lj_hessian(
                2.0, 2.0, np.zeros(6), d2, 1.0, x, H
            ),
        ),
        (
            "angle",
            9,
            lambda d2, x, H: F.angle_derivatives.build_angle_hessian(
                np.cos(1.0), np.cos(1.5), np.zeros(9), d2, 1.0, x, H
            ),
        ),
        (
            "dihedral",
            12,
            lambda d2, x, H: F.dihedral_derivatives.build_dihedral_hessian(
                1.0,
                np.cos(1.0),
                np.sin(1.0),
                np.sin(1.5),
                np.cos(1.5),
                np.zeros(12),
                d2,
                np.zeros(12),
                np.zeros_like(d2),
                1.0,
                x,
                H,
            ),
        ),
    ],
)
def test_hessian_scatter_covers_upper_triangle_once(name, n, call):
    d2 = np.arange(1, n * (n + 1) // 2 + 1, dtype=float)
    xhat = np.arange(1, n + 1, dtype=np.int32)
    H = np.zeros((n, n), order="F")
    call(d2, xhat, H)
    assert H[0, 0] != 0.0
    np.testing.assert_allclose(H / H[0, 0], _expected_index_matrix(n), rtol=1e-12)


# ---------------------------------------------------------------------------
# whole TS FF away from its reference values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ["struc1.xyz", "struc2.xyz"])
def test_tsff_hessian_matches_fd_of_gradient(structure):
    """TS FF at the reactant/product geometry: theta != theta0 for most terms."""
    nat, _, xyz, _ = readin_xyz(os.path.join(EXAMPLE_DIR, structure))
    ff = ForceField(nat, os.path.join(EXAMPLE_DIR, "tsff.csv"), readff=True)
    geo = angstrom2bohr(convert_xyz_to_fortranstyle(xyz))  # (3, nat), bohr
    analytic = complete_hessian(np.asfortranarray(geo), ff)

    def grad(g):  # g rows = atoms
        return complete_gradient(np.asfortranarray(g.T), ff)

    fd = _fd_hessian(grad, np.ascontiguousarray(geo.T))
    np.testing.assert_allclose(analytic, fd, atol=1e-6)
