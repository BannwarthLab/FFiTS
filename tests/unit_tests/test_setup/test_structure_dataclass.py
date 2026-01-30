"""
Tests for Structure dataclass.
Tests the complete information container for a given structure.
"""

import pytest
import numpy as np
import networkx as nx
from ffits.datatype.structure_data import Structure, StructurePath, ForceField, StructuralInformation
from tests.unit_tests.mock_functions import (
    create_mock_structure,
    create_mock_structure_path,
    create_mock_forcefield,
    create_mock_structural_information,
    create_mock_xyz_array,
    create_mock_wbo_dict,
    create_mock_atom_types,
)


class TestStructureDataclass:
    """Tests for the Structure dataclass consistency and behavior."""
    
    def test_structure_consistency_nat(self):
        """Test that all components have consistent nat (number of atoms)."""
        nat = 5
        structure = create_mock_structure(nat=nat)

        assert structure.ff.nat == nat
        assert structure.info.nat == nat
        assert structure.info.xyz.shape[0] == nat
        assert len(structure.info.atom_types) == nat
    
    def test_structure_hessian_dimensions(self):
        """Test that Hessian has correct dimensions (3*nat x 3*nat)."""
        for nat in [1, 2, 3, 4, 5]:
            structure = create_mock_structure(nat=nat)
            expected_dim = 3 * nat
            assert structure.info.hessian.shape == (expected_dim, expected_dim)
    
    def test_structure_graph_creation(self):
        """Test that Structure creates graph from WBO."""
        structure = create_mock_structure(nat=3)
        
        assert hasattr(structure.info, 'complete_graph')
        assert hasattr(structure.info, 'seperate_molecule_list')
        assert type(structure.info.complete_graph) is nx.Graph
        assert structure.info.molecule_count > 0
    
    def test_structure_vander_matrix_shape(self):
        """Test that van der Waals matrix has correct shape."""
        for nat in [1, 2, 3, 5]:
            structure = create_mock_structure(nat=nat)
            assert structure.info.vander_matrix.shape == (nat, nat)
    
    def test_structure_different_sizes(self):
        """Test Structure consistency with various molecule sizes."""
        for nat in [1, 3, 5, 7, 10]:
            structure = create_mock_structure(nat=nat)
            
            assert structure.info.nat == nat
            assert structure.ff.nat == nat
            assert structure.info.xyz.shape == (nat, 3)
    
    def test_structure_custom_path_values(self):
        """Test Structure with custom path values."""
        path = StructurePath(
            xyz_filename="custom.xyz",
            hessian_filename="custom.hess",
            wbo_filename="custom.wbo",
            ff_filename="custom_ff.csv"
        )
        ff = create_mock_forcefield(nat=3)
        info = create_mock_structural_information(nat=3)
        
        structure = Structure(path=path, ff=ff, info=info)
        
        assert structure.path.xyz_filename == "custom.xyz"
        assert structure.path.hessian_filename == "custom.hess"
    
    def test_structure_bo_matrix_symmetry(self):
        """Test that bond order matrix is symmetric."""
        structure = create_mock_structure(nat=3)
        np.testing.assert_array_equal(structure.info.bo_matrix, structure.info.bo_matrix.T)
    
    def test_structure_hessian_symmetry(self):
        """Test that Hessian matrix is symmetric."""
        structure = create_mock_structure(nat=3)
        
        np.testing.assert_array_almost_equal(
            structure.info.hessian, 
            structure.info.hessian.T,
            decimal=10
        )
    
    def test_structure_hessian_positive_semidefinite(self):
        """Test that Hessian is positive semi-definite."""
        structure = create_mock_structure(nat=3)
        
        eigenvalues = np.linalg.eigvals(structure.info.hessian)
        assert all(eig >= -1e-8 for eig in eigenvalues)
    
    def test_structure_xyz_magnitude(self):
        """Test that XYZ coordinates are physically reasonable."""
        structure = create_mock_structure(nat=5)
        
        assert np.all(np.abs(structure.info.xyz) < 100)
        assert np.any(structure.info.xyz != 0)
    
    def test_structure_wbo_sorted_pairs(self):
        """Test that WBO dictionary has sorted atom pairs."""
        structure = create_mock_structure(nat=5)
        
        for bond in structure.info.wbo.keys():
            assert len(bond) == 2
            assert bond[0] < bond[1]
    
    def test_structure_wbo_physical_values(self):
        """Test that WBO values are physically reasonable (0 < order <= 3)."""
        structure = create_mock_structure(nat=5)
        
        for bond, order in structure.info.wbo.items():
            assert order > 0
            assert order <= 3.0
    
    def test_structure_molecule_count(self):
        """Test that molecule count matches separated molecule list."""
        structure = create_mock_structure(nat=7)
        
        assert structure.info.molecule_count >= 1
        assert len(structure.info.seperate_molecule_list) == structure.info.molecule_count
    
    def test_structure_fortran_xyz_shape(self):
        """Test that Fortran-style XYZ has shape (3, nat)."""
        nat = 4
        structure = create_mock_structure(nat=nat)
        assert structure.info.fortran_xyz.shape == (3, nat)
    
    def test_structure_single_atom(self):
        """Test Structure with single atom."""
        structure = create_mock_structure(nat=1)
        
        assert structure.info.nat == 1
        assert structure.ff.nat == 1
        assert structure.info.xyz.shape == (1, 3)
        assert structure.info.hessian.shape == (3, 3)
    
    def test_structure_large_molecule(self):
        """Test Structure with larger molecule (20 atoms)."""
        nat = 20
        structure = create_mock_structure(nat=nat)
        
        assert structure.info.nat == nat
        assert structure.ff.nat == nat
        assert structure.info.hessian.shape == (60, 60)
    
    def test_structure_minimal_wbo(self):
        """Test Structure with minimal WBO data (single bond)."""
        nat = 2
        xyz = create_mock_xyz_array(nat)
        wbo = {(1, 2): 1.0}
        atom_types = create_mock_atom_types(nat)
        
        info = StructuralInformation(nat=nat, xyz=xyz, wbo_dict=wbo, atom_types=atom_types)
        path = create_mock_structure_path()
        ff = create_mock_forcefield(nat=nat)
        
        structure = Structure(path=path, ff=ff, info=info)
        
        assert len(structure.info.wbo) == 1
        assert structure.info.molecule_count == 1
    
    def test_structure_empty_wbo(self):
        """Test Structure with empty WBO data."""
        nat = 2
        xyz = create_mock_xyz_array(nat)
        wbo = {}
        atom_types = create_mock_atom_types(nat)
        
        info = StructuralInformation(nat=nat, xyz=xyz, wbo_dict=wbo, atom_types=atom_types)
        path = create_mock_structure_path()
        ff = create_mock_forcefield(nat=nat)
        
        structure = Structure(path=path, ff=ff, info=info)
        
        assert not hasattr(structure.info, 'molecule_count')


class TestStructureIntegration:
    """Tests for how Structure components interact."""
    
    def test_info_and_ff_nat_match(self):
        """Test that info and FF have matching number of atoms."""
        for nat in [1, 3, 5, 8]:
            structure = create_mock_structure(nat=nat)
            assert structure.info.nat == structure.ff.nat
    
    def test_xyz_and_atom_types_length_match(self):
        """Test that XYZ coordinates and atom types have matching lengths."""
        structure = create_mock_structure(nat=7)
        assert structure.info.xyz.shape[0] == len(structure.info.atom_types)
    
    def test_hessian_and_nat_relationship(self):
        """Test that Hessian size is always 3 * nat."""
        for nat in [2, 3, 5, 10]:
            structure = create_mock_structure(nat=nat)
            expected_size = 3 * nat
            assert structure.info.hessian.shape[0] == expected_size
            assert structure.info.hessian.shape[1] == expected_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
