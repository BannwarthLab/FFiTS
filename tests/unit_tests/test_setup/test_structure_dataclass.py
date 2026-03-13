import os
import pytest
import numpy as np
import networkx as nx
from tempfile import TemporaryDirectory
from pathlib import Path

from ffits.datatype.structure_data import (
    StructurePath,
    ForceField,
    StructuralInformation,
    Structure,
    Name,
    get_vander_matrix,
    atom_symbol_to_number,
    angstrom2bohr,
    convert_xyz_to_fortranstyle,
)
from ffits.io.reader import readin_xyz, read_wbo_file

import numpy as np


# same data as in small_single_molecule
XYZ = np.array([
    [-2.33287094, 3.31176687, 0.20110100],
    [-0.91630217, 2.85867268, -0.04327585],
    [0.06256276, 3.55862185, 0.08938727],
    [-2.92591176, 3.18375556, -0.71288068],
    [-2.79725946, 2.67965821, 0.96793335],
    [-0.81484498, 1.79716556, -0.36699673],
    [-2.35087245, 4.35655928, 0.51563164]
])
WBO = {
    (1, 2): 1.02668632226515, 
    (2, 3): 1.92755303185758, 
    (1, 4): 0.955689824153634,  
    (1, 5): 0.955863695522291,  
    (2, 6): 0.933812077856736,  
    (1, 7): 0.982636418257069  
}
ATOM_TYPES = ['C', 'C', 'O', 'H', 'H', 'H', 'H']

NAT = 7


class TestAtomConversionFunctions:
    """Test utility functions for atomic conversions."""

    def test_atom_symbol_to_number(self):
        """Test element symbol to atomic number conversion."""
        assert atom_symbol_to_number("H") == 1
        assert atom_symbol_to_number("C") == 6
        assert atom_symbol_to_number("N") == 7
        assert atom_symbol_to_number("O") == 8
        assert atom_symbol_to_number("He") == 2
        assert atom_symbol_to_number("Au") == 79

    def test_atom_symbol_to_number_case_insensitive(self):
        """Test that symbol conversion is case-insensitive."""
        assert atom_symbol_to_number("c") == 6
        assert atom_symbol_to_number("H") == 1
        assert atom_symbol_to_number("he") == 2

    def test_atom_symbol_to_number_invalid(self):
        """Test handling of invalid atom symbols."""
        with pytest.raises(ValueError):
            atom_symbol_to_number("Xx")

    def test_angstrom2bohr_scalar(self):
        """Test scalar angstrom to bohr conversion."""
        result = angstrom2bohr(1.0)
        expected = 1.0 / (1 / 1.8897259)
        assert pytest.approx(result) == expected

    def test_angstrom2bohr_array(self):
        """Test array angstrom to bohr conversion."""
        arr = np.array([1.0, 2.0, 3.0])
        result = angstrom2bohr(arr)
        assert result.shape == arr.shape

    def test_convert_xyz_to_fortranstyle(self):
        """Test XYZ to Fortran-style column-major conversion."""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = convert_xyz_to_fortranstyle(2, xyz)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], order='F')
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_vander_matrix(self):
        """Test van der Waals matrix generation."""
        atoms = np.array(['H', 'H', 'C'])
        vander = get_vander_matrix(atoms)
        
        # Should be symmetric
        np.testing.assert_array_equal(vander, vander.T)
        
        # Size should match number of atoms
        assert vander.shape == (3, 3)
        
        # Diagonal should be 2 * radius for same atoms
        assert vander[0, 0] > 0
        assert vander[1, 1] > 0


class TestName:
    """Test Name dataclass static methods."""

    def test_modified_ff(self):
        """Test modified_ff naming."""
        result = Name.modified_ff("test.ff")
        assert result == "test.ff"

    def test_fitted_ff(self):
        """Test fitted_ff naming."""
        result = Name.fitted_ff("test.ff")
        assert result == "test.ff"

    def test_optimized_xyz(self):
        """Test optimized_xyz naming."""
        result = Name.optimized_xyz("structure.xyz")
        assert result == "opt_structure.xyz"

    def test_aligned_xyz(self):
        """Test aligned_xyz naming."""
        result = Name.aligned_xyz("structure.xyz")
        assert result == "aligned_structure.xyz"

    def test_original_xyz(self):
        """Test original_xyz naming."""
        result = Name.original_xyz("structure.xyz")
        assert result == "original_structure.xyz"


class TestStructurePath:
    """Test StructurePath dataclass."""

    def test_structurepath_creation(self):
        """Test creating a StructurePath instance."""
        path = StructurePath(
            xyz_filename="structure.xyz",
            hessian_filename="hessian.hess",
            wbo_filename="wbo.wbo",
            ff_filename="ff.ff",
        )
        assert path.xyz_filename == "structure.xyz"
        assert path.hessian_filename == "hessian.hess"
        assert path.wbo_filename == "wbo.wbo"
        assert path.ff_filename == "ff.ff"

    def test_structurepath_with_paths(self):
        """Test StructurePath with full paths."""
        path = StructurePath(
            xyz_filename="/home/user/data/mol1.xyz",
            hessian_filename="/home/user/data/mol1.hess",
            wbo_filename="/home/user/data/mol1.wbo",
            ff_filename="/home/user/data/mol1.ff",
        )
        assert "/home/user/data" in path.xyz_filename


class TestForceField:
    """Test ForceField class."""

    @pytest.fixture
    def ff_path(self):
        """Provide path to example force field."""
        test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
        return str(test_dir / "ff1.csv")

    def test_forcefield_initialization_without_read(self):
        """Test ForceField initialization without reading from file."""
        ff = ForceField(nat=7, ff_filename="test.ff", readff=False)
        assert ff.nat == 7
        assert ff.ff_filename == "test.ff"
        assert len(ff.bonds) == 0
        assert len(ff.angles) == 0
        assert len(ff.dihedrals) == 0
        assert len(ff.repulsive) == 0

    def test_forcefield_read_from_file(self, ff_path):
        """Test reading force field from file."""
        ff = ForceField(nat=7, ff_filename=ff_path, readff=True)
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0
        assert len(ff.dihedrals) > 0
        assert len(ff.repulsive) > 0

    def test_forcefield_columns_initialization(self):
        """Test that force field has correct columns."""
        ff = ForceField(nat=7, ff_filename="test.ff", readff=False)
        expected_cols = ['type', 'atoms', 'parameter', 'reference_value']
        expected_cols_dihedral = ['type', 'atoms', 'parameter', 'reference_value', 'proper_dihedral']
        assert list(ff.bonds.columns) == expected_cols
        assert list(ff.angles.columns) == expected_cols
        assert list(ff.dihedrals.columns) == expected_cols_dihedral
        assert list(ff.repulsive.columns) == expected_cols

    def test_forcefield_write_to_csv(self, ff_path):
        """Test writing force field to CSV."""
        ff = ForceField(nat=7, ff_filename=ff_path, readff=True)
        
        with TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_ff_output.csv")
            ff.ff_filename = output_path
            ff.write()
            
            # Check file was created
            assert os.path.exists(output_path)
            
            # Read it back and verify structure
            ff2 = ForceField(nat=7, ff_filename=output_path, readff=True)
            assert len(ff2.bonds) == len(ff.bonds)
            assert len(ff2.angles) == len(ff.angles)
            assert len(ff2.dihedrals) == len(ff.dihedrals)
    def test_forcefield_with_calculators(self):
        """Test ForceField with calculator functions."""
        def dummy_energy(xyz, ff):
            return 0.0
        
        def dummy_gradient(xyz, ff):
            return np.zeros(3)
        
        def dummy_hessian(xyz, ff):
            return np.zeros((3, 3))
        
        ff = ForceField(
            nat=7,
            ff_filename="test.ff",
            readff=False,
            energy_calculator=dummy_energy,
            gradient_calculator=dummy_gradient,
            hessian_calculator=dummy_hessian,
        )
        
        assert ff.energy_calculator is not None
        assert ff.gradient_calculator is not None
        assert ff.hessian_calculator is not None


class TestStructuralInformation:
    """Test StructuralInformation class."""

    @pytest.fixture
    def structural_info(self):
        """Create a StructuralInformation instance for testing."""
        return StructuralInformation(
            nat=NAT,
            xyz=XYZ,
            wbo_dict=WBO,
            atom_types=np.array(ATOM_TYPES),
        )

    def test_structural_information_initialization(self, structural_info):
        """Test StructuralInformation initialization."""
        assert structural_info.nat == NAT
        assert structural_info.atom_types.shape[0] == NAT
        assert len(structural_info.wbo) == len(WBO)

    def test_structural_information_vander_matrix(self, structural_info):
        """Test van der Waals matrix creation."""
        assert structural_info.vander_matrix.shape == (NAT, NAT)
        # Matrix should be symmetric
        np.testing.assert_array_equal(
            structural_info.vander_matrix,
            structural_info.vander_matrix.T,
        )

    def test_structural_information_bo_matrix(self, structural_info):
        """Test bond order matrix creation."""
        bo_matrix = structural_info.bo_matrix
        # Should be square and symmetric
        assert bo_matrix.shape == (NAT, NAT)
        np.testing.assert_array_equal(bo_matrix, bo_matrix.T)
        
        # Check specific bonds from WBO
        assert bo_matrix[0, 1] == WBO[(1, 2)]  # Atoms 1-2
        assert bo_matrix[1, 2] == WBO[(2, 3)]  # Atoms 2-3

    def test_structural_information_fortran_xyz(self, structural_info):
        """Test Fortran-style XYZ coordinates."""
        # Fortran format should be (3, nat)
        assert structural_info.fortran_xyz.shape == (3, NAT)

    def test_structural_information_complete_graph(self, structural_info):
        """Test graph creation from WBO."""
        graph = structural_info.complete_graph
        assert isinstance(graph, nx.Graph)
        # Graph should have edges for bonds with BO > threshold
        assert graph.number_of_nodes() > 0

    def test_structural_information_molecule_count(self, structural_info):
        """Test molecule count (connected components)."""
        # The test structure should be a single molecule
        assert structural_info.molecule_count >= 1

    def test_structural_information_separate_molecules(self, structural_info):
        """Test split into separate molecules."""
        subgraphs = structural_info.seperate_molecule_list
        assert len(subgraphs) == structural_info.molecule_count
        assert len(subgraphs) > 0

    def test_structural_information_bo_matrix_symmetry(self, structural_info):
        """Test that BO matrix is symmetric."""
        bo_matrix = structural_info.bo_matrix
        np.testing.assert_array_equal(bo_matrix, bo_matrix.T)

    def test_structuralinfo_angstrom2bohr_method(self, structural_info):
        """Test angstrom to bohr conversion method."""
        result = structural_info.angstrom2bohr(1.0)
        assert isinstance(result, (float, np.ndarray))
        assert result > 0

    def test_structuralinfo_create_graph_low_bo_threshold(self):
        """Test that bonds with very low WBO are excluded from graph."""
        wbo_with_low = {
            (1, 2): 1.0,  # Strong bond
            (2, 3): 0.05,  # Very weak bond (below threshold)
        }
        info = StructuralInformation(
            nat=3,
            xyz=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            wbo_dict=wbo_with_low,
            atom_types=np.array(['C', 'C', 'C']),
        )
        graph = info.complete_graph
        # Only the strong bond should appear
        assert graph.number_of_edges() == 1


class TestStructure:
    """Test Structure dataclass."""

    @pytest.fixture
    def example_files(self):
        """Provide paths to example files."""
        test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
        return {
            'xyz': str(test_dir / "struc1.xyz"),
            'ff': str(test_dir / "ff1.csv"),
            'wbo': str(test_dir / "wbo1"),
        }

    @pytest.fixture
    def structure(self, example_files):
        """Create a Structure instance for testing."""
        # Load data
        nat, comment, xyz, atom_types = readin_xyz(example_files['xyz'])
        wbo = read_wbo_file(example_files['wbo'])
        
        # Create components
        path = StructurePath(
            xyz_filename=example_files['xyz'],
            hessian_filename="",
            wbo_filename=example_files['wbo'],
            ff_filename=example_files['ff'],
        )
        
        ff = ForceField(nat=nat, ff_filename=example_files['ff'], readff=True)
        
        info = StructuralInformation(
            nat=nat,
            xyz=xyz,
            wbo_dict=wbo,
            atom_types=np.array(atom_types),
        )
        
        # Create Structure
        return Structure(path=path, ff=ff, info=info)

    def test_structure_creation(self, structure):
        """Test Structure creation."""
        assert structure.path is not None
        assert structure.ff is not None
        assert structure.info is not None

    def test_structure_has_path(self, structure):
        """Test Structure path component."""
        assert isinstance(structure.path, StructurePath)
        assert structure.path.xyz_filename.endswith('.xyz')
        assert structure.path.ff_filename.endswith('ff1.csv')

    def test_structure_has_forcefield(self, structure):
        """Test Structure force field component."""
        assert isinstance(structure.ff, ForceField)
        assert structure.ff.nat > 0

    def test_structure_has_info(self, structure):
        """Test Structure structural information component."""
        assert isinstance(structure.info, StructuralInformation)
        assert structure.info.nat > 0

    def test_structure_atom_count_consistency(self, structure):
        """Test that atom counts are consistent across components."""
        assert structure.ff.nat == structure.info.nat

    def test_structure_ff_and_info_consistency(self, structure):
        """Test consistency between force field and structural info."""
        # Both should reference the same structure
        assert structure.ff.nat == structure.info.nat
        assert len(structure.info.atom_types) == structure.ff.nat

    def test_structure_with_real_data(self, example_files):
        """Test Structure with real example data."""
        nat, comment, xyz, atom_types = readin_xyz(example_files['xyz'])
        wbo = read_wbo_file(example_files['wbo'])
        
        path = StructurePath(
            xyz_filename=example_files['xyz'],
            hessian_filename="",
            wbo_filename=example_files['wbo'],
            ff_filename=example_files['ff'],
        )
        
        ff = ForceField(nat=nat, ff_filename=example_files['ff'], readff=True)
        info = StructuralInformation(
            nat=nat,
            xyz=xyz,
            wbo_dict=wbo,
            atom_types=np.array(atom_types),
        )
        
        structure = Structure(path=path, ff=ff, info=info)
        
        # Verify structure is correctly assembled
        assert structure.info.nat == 7
        assert len(structure.info.wbo) == 6
        assert len(structure.ff.bonds) > 0


class TestStructuralInformationGraphOperations:
    """Test graph-related operations in StructuralInformation."""

    def test_subgraph_node_attributes(self):
        """Test that subgraph nodes have correct attributes."""
        info = StructuralInformation(
            nat=NAT,
            xyz=XYZ,
            wbo_dict=WBO,
            atom_types=np.array(ATOM_TYPES),
        )
        
        for subgraph in info.seperate_molecule_list:
            for node in subgraph.nodes():
                assert 'id_in_subgraph' in subgraph.nodes[node]

    def test_multiple_molecules_separation(self):
        """Test separation of multiple molecules."""
        # Create WBO for two separate molecules (no connection)
        wbo_separate = {
            (1, 2): 1.0,  # Molecule 1
            (4, 5): 1.0,  # Molecule 2
        }
        
        info = StructuralInformation(
            nat=5,
            xyz=np.array([
                [0, 0, 0], [1, 0, 0], [2, 0, 0],
                [10, 0, 0], [11, 0, 0]
            ], dtype=float),
            wbo_dict=wbo_separate,
            atom_types=np.array(['C', 'C', 'C', 'C', 'C']),
        )
        
        # Should have 2 separate molecules (2 connected components)
        assert info.molecule_count == 2

    def test_empty_wbo(self):
        """Test StructuralInformation with empty WBO (single atoms)."""
        info = StructuralInformation(
            nat=2,
            xyz=np.array([[0, 0, 0], [5, 0, 0]], dtype=float),
            wbo_dict={},
            atom_types=np.array(['He', 'He']),
        )
        
        # Should still have vander_matrix
        assert info.vander_matrix.shape == (2, 2)


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def full_structure(self):
        """Create a complete structure from example files."""
        test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
        
        xyz_file = str(test_dir / "struc1.xyz")
        ff_file = str(test_dir / "ff1.csv")
        wbo_file = str(test_dir / "wbo1")
        
        nat, comment, xyz, atom_types = readin_xyz(xyz_file)
        wbo = read_wbo_file(wbo_file)
        
        path = StructurePath(
            xyz_filename=xyz_file,
            hessian_filename="",
            wbo_filename=wbo_file,
            ff_filename=ff_file,
        )
        
        ff = ForceField(nat=nat, ff_filename=ff_file, readff=True)
        info = StructuralInformation(
            nat=nat,
            xyz=xyz,
            wbo_dict=wbo,
            atom_types=np.array(atom_types),
        )
        
        return Structure(path=path, ff=ff, info=info)

    def test_structure_components_interact(self, full_structure):
        """Test that structure components work together."""
        # All components should have consistent nat
        assert full_structure.ff.nat == full_structure.info.nat
        
        # FF should have bonds/angles/dihedrals computed for this structure
        assert len(full_structure.ff.bonds) > 0
        
        # Info should have graph computed from WBO
        assert full_structure.info.complete_graph is not None

    def test_structure_with_all_data_types(self, full_structure):
        """Test structure with all types of force field parameters."""
        ff = full_structure.ff
        
        # Check that we have all FF parameter types
        assert len(ff.bonds) > 0, "Should have bond parameters"
        assert len(ff.angles) > 0, "Should have angle parameters"
        assert len(ff.dihedrals) > 0, "Should have dihedral parameters"
        assert len(ff.repulsive) > 0, "Should have repulsive parameters"

    def test_vander_radius_effect(self):
        """Test that van der Waals radii affect interaction matrix."""
        atoms = np.array(['H', 'C'])
        vander = get_vander_matrix(atoms)
        
        # Verify that different atoms give different interactions
        assert vander[0, 0] != vander[1, 1]  # Different atoms
        assert vander[0, 1] == vander[1, 0]  # Symmetric

    def test_roundtrip_ff_write_read(self):
        """Test writing and reading back force field data."""
        test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
        ff_file = str(test_dir / "ff1.csv")
        
        # Read original
        ff1 = ForceField(nat=7, ff_filename=ff_file, readff=True)
        original_bonds = len(ff1.bonds)
        
        with TemporaryDirectory() as tmpdir:
            # Write to temp file
            temp_ff = os.path.join(tmpdir, "temp_ff.csv")
            ff1.ff_filename = temp_ff
            ff1.write()
            
            # Read back
            ff1_read = ForceField(nat=7, ff_filename=temp_ff, readff=True)
            
            # Compare
            assert len(ff1_read.bonds) == original_bonds
            assert len(ff1_read.angles) == len(ff1.angles)
            assert len(ff1_read.dihedrals) == len(ff1.dihedrals)
            assert len(ff1_read.repulsive) == len(ff1.repulsive)