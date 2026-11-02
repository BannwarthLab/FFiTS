"""
Unit tests for the internal coordinate functions in ffits.utils.geometry.

The vectorized functions (bondlengths, angles, dihedral_angles) are the single
implementation; the scalar functions (bondlength, angle, dihedral_angle) wrap them.
The tests check them against analytic values, against an independent per-term
reference implementation (the previous scalar code, kept here frozen), and for
consistency between the scalar and vectorized interfaces.
"""

from itertools import combinations, permutations

import numpy as np
import pytest

from ffits.utils.geometry import (
    angle,
    angles,
    bondlength,
    bondlengths,
    convert_xyz_to_fortranstyle,
    dihedral_angle,
    dihedral_angles,
)
from tests.test_utils import NAT, XYZ


# -------------------------------------------------------------------------
# --- Independent per-term reference implementation -----------------------
# -------------------------------------------------------------------------
def _reference_bondlength(geometry, a1, a2):
    return np.linalg.norm(geometry[:, a1] - geometry[:, a2])


def _reference_angle(geometry, a1, a2, a3):
    v12 = geometry[:, a1] - geometry[:, a2]
    v23 = geometry[:, a3] - geometry[:, a2]
    cos_theta = np.dot(v12, v23) / (np.linalg.norm(v12) * np.linalg.norm(v23))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def _reference_dihedral(geometry, a1, a2, a3, a4):
    v12 = geometry[:, a2] - geometry[:, a1]
    v23 = geometry[:, a3] - geometry[:, a2]
    v34 = geometry[:, a4] - geometry[:, a3]
    normal1 = np.cross(v12, v23)
    normal2 = np.cross(v23, v34)
    x = np.dot(normal1, normal2)
    y = np.dot(np.cross(normal1, normal2), v23 / np.linalg.norm(v23))
    return np.arctan2(y, x)


# -------------------------------------------------------------------------
# --- Fixtures ------------------------------------------------------------
# -------------------------------------------------------------------------
@pytest.fixture
def molecule():
    """Test molecule geometry, shape (3, nat)."""
    return convert_xyz_to_fortranstyle(XYZ)


@pytest.fixture
def random_geometry():
    """Random geometry of 12 atoms, shape (3, 12)."""
    return np.random.default_rng(42).normal(scale=2.0, size=(3, 12))


def _atom_tuples(nat, k, max_terms=500):
    """Distinct atom tuples of length k (ordered for k > 2, so sign/vertex conventions are covered)."""
    tuples = combinations(range(nat), 2) if k == 2 else permutations(range(nat), k)
    atoms = []
    for t in tuples:
        atoms.append(t)
        if len(atoms) == max_terms:
            break
    return np.array(atoms, dtype=np.int32)


def _dihedral_geometry(phi):
    """Four atoms with a 1-2-3-4 dihedral of phi: 2-3 along z, 1 along +x, 4 rotated by phi about z."""
    return np.array(
        [
            [1.0, 0.0, 0.0, np.cos(phi)],
            [0.0, 0.0, 0.0, np.sin(phi)],
            [0.0, 0.0, 1.5, 1.5],
        ]
    )


# -------------------------------------------------------------------------
# --- Analytic values -----------------------------------------------------
# -------------------------------------------------------------------------
class TestAnalyticValues:
    """Geometries with known internal coordinates."""

    def test_bondlength_345_triangle(self):
        geometry = np.array([[0.0, 3.0], [0.0, 4.0], [0.0, 0.0]])
        assert bondlengths(geometry, [[0, 1]])[0] == pytest.approx(5.0)

    @pytest.mark.parametrize("theta", [0.0, np.pi / 3, np.pi / 2, 2.0, np.pi])
    def test_angle(self, theta):
        geometry = np.array(
            [
                [1.0, 0.0, 2.0 * np.cos(theta)],
                [0.0, 0.0, 2.0 * np.sin(theta)],
                [0.0, 0.0, 0.0],
            ]
        )
        assert angles(geometry, [[0, 1, 2]])[0] == pytest.approx(theta, abs=1e-7)

    @pytest.mark.parametrize("phi", [0.0, np.pi / 3, np.pi / 2, 2.5, -np.pi / 2, -1.0])
    def test_dihedral_sign_and_value(self, phi):
        geometry = _dihedral_geometry(phi)
        assert dihedral_angles(geometry, [[0, 1, 2, 3]])[0] == pytest.approx(phi)

    def test_dihedral_trans(self):
        geometry = _dihedral_geometry(np.pi)
        assert abs(dihedral_angles(geometry, [[0, 1, 2, 3]])[0]) == pytest.approx(np.pi)


# -------------------------------------------------------------------------
# --- Comparison to the per-term reference implementation ----------------
# -------------------------------------------------------------------------
class TestAgainstReference:
    """Vectorized results equal the independent per-term implementation."""

    @pytest.mark.parametrize("geometry_name", ["molecule", "random_geometry"])
    def test_bondlengths(self, geometry_name, request):
        geometry = request.getfixturevalue(geometry_name)
        atoms = _atom_tuples(geometry.shape[1], 2)
        expected = [_reference_bondlength(geometry, *a) for a in atoms]
        np.testing.assert_allclose(bondlengths(geometry, atoms), expected, rtol=1e-12)

    @pytest.mark.parametrize("geometry_name", ["molecule", "random_geometry"])
    def test_angles(self, geometry_name, request):
        geometry = request.getfixturevalue(geometry_name)
        atoms = _atom_tuples(geometry.shape[1], 3)
        expected = [_reference_angle(geometry, *a) for a in atoms]
        np.testing.assert_allclose(angles(geometry, atoms), expected, atol=1e-12)

    @pytest.mark.parametrize("geometry_name", ["molecule", "random_geometry"])
    def test_dihedral_angles(self, geometry_name, request):
        geometry = request.getfixturevalue(geometry_name)
        atoms = _atom_tuples(geometry.shape[1], 4)
        expected = [_reference_dihedral(geometry, *a) for a in atoms]
        np.testing.assert_allclose(
            dihedral_angles(geometry, atoms), expected, atol=1e-12
        )


# -------------------------------------------------------------------------
# --- Scalar wrappers and interface ---------------------------------------
# -------------------------------------------------------------------------
class TestScalarInterface:
    """Scalar functions give the same values as the vectorized ones, term by term."""

    def test_scalar_matches_vectorized(self, molecule):
        for k, scalar, vectorized in [
            (2, bondlength, bondlengths),
            (3, angle, angles),
            (4, dihedral_angle, dihedral_angles),
        ]:
            atoms = _atom_tuples(NAT, k, max_terms=100)
            batch = vectorized(molecule, atoms)
            for n, a in enumerate(atoms):
                assert scalar(molecule, *a) == batch[n]

    def test_scalar_returns_scalar(self, molecule):
        assert np.ndim(bondlength(molecule, 0, 1)) == 0
        assert np.ndim(angle(molecule, 0, 1, 2)) == 0
        assert np.ndim(dihedral_angle(molecule, 0, 1, 2, 3)) == 0

    @pytest.mark.parametrize(
        "vectorized, k", [(bondlengths, 2), (angles, 3), (dihedral_angles, 4)]
    )
    def test_empty_input(self, molecule, vectorized, k):
        assert vectorized(molecule, np.zeros((0, k), dtype=np.int32)).shape == (0,)


# -------------------------------------------------------------------------
# --- Invariances ---------------------------------------------------------
# -------------------------------------------------------------------------
class TestInvariances:
    """Internal coordinates don't change under rigid motions; dihedrals flip sign under mirroring."""

    @staticmethod
    def _rotation_matrix():
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
        return q * np.sign(np.linalg.det(q))  # proper rotation (det = +1)

    def test_rigid_motion(self, molecule):
        moved = self._rotation_matrix() @ molecule + np.array([[1.0], [-2.0], [0.5]])
        for k, vectorized in [(2, bondlengths), (3, angles), (4, dihedral_angles)]:
            atoms = _atom_tuples(NAT, k)
            np.testing.assert_allclose(
                vectorized(moved, atoms), vectorized(molecule, atoms), atol=1e-10
            )

    def test_mirror_flips_dihedral_sign(self, molecule):
        mirrored = molecule * np.array([[-1.0], [1.0], [1.0]])
        atoms = _atom_tuples(NAT, 4)
        np.testing.assert_allclose(
            dihedral_angles(mirrored, atoms),
            -dihedral_angles(molecule, atoms),
            atol=1e-10,
        )

    def test_reversed_order(self, molecule):
        for k, vectorized in [(2, bondlengths), (3, angles), (4, dihedral_angles)]:
            atoms = _atom_tuples(NAT, k)
            np.testing.assert_allclose(
                vectorized(molecule, atoms[:, ::-1]),
                vectorized(molecule, atoms),
                atol=1e-10,
            )
