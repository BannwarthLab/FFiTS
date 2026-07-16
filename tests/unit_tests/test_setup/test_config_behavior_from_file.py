"""
Comprehensive tests to verify that all config file parameters are correctly loaded
into CalculationData objects. Each test class covers one configuration section,
and each test function verifies a single config parameter.
"""

import pytest
import tempfile
import os
from ffits.datatype.calculation_data import CalculationData


class TestSystemConfig:
    """Test System configuration values are correctly loaded from config."""

    def test_system_charge_from_config(self):
        """Verify that system charge from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[system]\ncharge = -2\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.charge == -2
        finally:
            os.unlink(temp_path)

    def test_system_multiplicity_from_config(self):
        """Verify that system multiplicity from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[system]\nmultiplicity = 3\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.multiplicity == 3
        finally:
            os.unlink(temp_path)

    def test_system_xtb_path_from_config(self):
        """Verify that system xtb_path from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[system]\nxtb_path = "/usr/bin/xtb"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.xtb_path == "/usr/bin/xtb"
        finally:
            os.unlink(temp_path)

    def test_system_use_gxtb_from_config(self):
        """Verify that system use_gxtb from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[system]\nuse_gxtb = true\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.use_gxtb is True
        finally:
            os.unlink(temp_path)

    def test_system_xtb_input_name_from_config(self):
        """Verify that system xtb_input_name from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[system]\nxtb_input_name = "custom_input"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.xtb_input_name == "custom_input"
        finally:
            os.unlink(temp_path)

    def test_system_xtb_alpb_solvent_from_config(self):
        """Verify that system xtb_alpb_solvent from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[system]\nxtb_alpb_solvent = "water"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.system.xtb_alpb_solvent == "water"
        finally:
            os.unlink(temp_path)


class TestReactantPathConfig:
    """Test Reactant path configuration values are correctly loaded from config."""

    def test_reactant_wbo_filename_from_config(self):
        """Verify that reactant wbo_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[reactant.path]\nwbo_filename = "reactant_custom.wbo"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_path.wbo_filename == "reactant_custom.wbo"
        finally:
            os.unlink(temp_path)

    def test_reactant_hessian_filename_from_config(self):
        """Verify that reactant hessian_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                '[reactant.path]\nhessian_filename = "reactant_hessian_matrix.hess"\n'
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert (
                calcdata.reactant_path.hessian_filename
                == "reactant_hessian_matrix.hess"
            )
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_filename_from_config(self):
        """Verify that reactant ff_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                '[reactant.path]\nff_filename = "reactant_force_field_params.csv"\n'
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert (
                calcdata.reactant_path.ff_filename == "reactant_force_field_params.csv"
            )
        finally:
            os.unlink(temp_path)


class TestProductPathConfig:
    """Test Product path configuration values are correctly loaded from config."""

    def test_product_wbo_filename_from_config(self):
        """Verify that product wbo_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[product.path]\nwbo_filename = "product_bonds.wbo"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.product_path.wbo_filename == "product_bonds.wbo"
        finally:
            os.unlink(temp_path)

    def test_product_hessian_filename_from_config(self):
        """Verify that product hessian_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[product.path]\nhessian_filename = "product.hess"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.product_path.hessian_filename == "product.hess"
        finally:
            os.unlink(temp_path)

    def test_product_ff_filename_from_config(self):
        """Verify that product ff_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[product.path]\nff_filename = "product_ff.csv"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.product_path.ff_filename == "product_ff.csv"
        finally:
            os.unlink(temp_path)


class TestReactantCalculationConfig:
    """Test Reactant calculation configuration values are correctly loaded from config."""

    def test_reactant_geometry_optimization_from_config(self):
        """Verify that reactant geometry_optimization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\ngeometry_optimization = true\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.geometry_optimization is True
        finally:
            os.unlink(temp_path)

    def test_reactant_wbo_calc_from_config(self):
        """Verify that reactant wbo_calc from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nwbo_calc = false\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.wbo_calc is False
        finally:
            os.unlink(temp_path)

    def test_reactant_hessian_calc_from_config(self):
        """Verify that reactant hessian_calc from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nhessian_calc = false\n")
            f.flush()
            temp_path = f.name
        try:
            with pytest.raises(FileNotFoundError):
                CalculationData.from_config(temp_path)
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.reactant_calc.hessian_calc is False
        finally:
            os.unlink(temp_path)

    def test_reactant_bo_threshold_from_config(self):
        """Verify that reactant bo_treshold from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nbo_treshold = 0.15\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.bo_treshold == 0.15
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameterization_from_config(self):
        """Verify that reactant ff_parameterization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nff_parameterization = false\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.ff_parameterization is False
        finally:
            os.unlink(temp_path)

    def test_reactant_test_parameterization_from_config(self):
        """Verify that reactant test_parameterization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\ntest_parameterization = true\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.test_parameterization is True
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameter_repulsion_from_config(self):
        """Verify that reactant ff_parameter_repulsion from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nff_parameter_repulsion = 0.05\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.ff_parameter_repulsion == 0.05
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameterization_maxiteration_from_config(self):
        """Verify that reactant ff_parameterization_maxiteration from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nff_parameterization_maxiteration = 2500\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.ff_parameterization_maxiteration == 2500
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameterization_stepsize_from_config(self):
        """Verify that reactant ff_parameterization_stepsize from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nff_parameterization_stepsize = 0.25\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.ff_parameterization_stepsize == 0.25
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameterization_threshold_from_config(self):
        """Verify that reactant ff_parameterization_threshold from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[reactant.calculation]\nff_parameterization_threshold = 0.0002\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.reactant_calc.ff_parameterization_threshold == 0.0002
        finally:
            os.unlink(temp_path)

    def test_reactant_ff_parameterization_constant_repulsion_from_config(self):
        """Verify that reactant ff_parameterization_constant_repulsion from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[reactant.calculation]\nff_parameterization_constant_repulsion = false\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert (
                calcdata.reactant_calc.ff_parameterization_constant_repulsion is False
            )
        finally:
            os.unlink(temp_path)


class TestProductCalculationConfig:
    """Test Product calculation configuration values are correctly loaded from config."""

    def test_product_geometry_optimization_from_config(self):
        """Verify that product geometry_optimization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\ngeometry_optimization = true\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.geometry_optimization is True
        finally:
            os.unlink(temp_path)

    def test_product_wbo_calc_from_config(self):
        """Verify that product wbo_calc from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nwbo_calc = false\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.wbo_calc is False
        finally:
            os.unlink(temp_path)

    def test_product_hessian_calc_from_config(self):
        """Verify that product hessian_calc from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nhessian_calc = false\n")
            f.flush()
            temp_path = f.name
        try:
            with pytest.raises(FileNotFoundError):
                CalculationData.from_config(temp_path)
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.product_calc.hessian_calc is False
        finally:
            os.unlink(temp_path)

    def test_product_bo_threshold_from_config(self):
        """Verify that product bo_treshold from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nbo_treshold = 0.20\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.bo_treshold == 0.20
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameterization_from_config(self):
        """Verify that product ff_parameterization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nff_parameterization = false\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameterization is False
        finally:
            os.unlink(temp_path)

    def test_product_test_parameterization_from_config(self):
        """Verify that product test_parameterization from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\ntest_parameterization = true\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.test_parameterization is True
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameter_repulsion_from_config(self):
        """Verify that product ff_parameter_repulsion from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nff_parameter_repulsion = 0.08\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameter_repulsion == 0.08
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameterization_maxiteration_from_config(self):
        """Verify that product ff_parameterization_maxiteration from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nff_parameterization_maxiteration = 1500\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameterization_maxiteration == 1500
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameterization_stepsize_from_config(self):
        """Verify that product ff_parameterization_stepsize from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nff_parameterization_stepsize = 0.30\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameterization_stepsize == 0.30
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameterization_threshold_from_config(self):
        """Verify that product ff_parameterization_threshold from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[product.calculation]\nff_parameterization_threshold = 0.0003\n")
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameterization_threshold == 0.0003
        finally:
            os.unlink(temp_path)

    def test_product_ff_parameterization_constant_repulsion_from_config(self):
        """Verify that product ff_parameterization_constant_repulsion from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[product.calculation]\nff_parameterization_constant_repulsion = true\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.product_calc.ff_parameterization_constant_repulsion is True
        finally:
            os.unlink(temp_path)


class TestTSPathConfig:
    """Test TS path configuration values are correctly loaded from config."""

    def test_ts_hessian_filename_from_config(self):
        """Verify that TS hessian_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                '[ts_guess.path]\nhessian_filename = "ts_hessian.hess"\n'
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.ts_path.hessian_filename == "ts_hessian.hess"
        finally:
            os.unlink(temp_path)

    def test_ts_ff_filename_from_config(self):
        """Verify that TS ff_filename from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[ts_guess.path]\nff_filename = "ts_forcefield.csv"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path, test=True)
            assert calcdata.ts_path.ff_filename == "ts_forcefield.csv"
        finally:
            os.unlink(temp_path)


class TestTSCalculationConfig:
    """Test TS calculation configuration values are correctly loaded from config."""

    def test_ts_factor_reactant_from_config(self):
        """Verify that TS factor_reactant from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nfactor_reactant = 0.3\nfactor_product = 0.7\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.factor_reactant == 0.3
        finally:
            os.unlink(temp_path)

    def test_ts_factor_product_from_config(self):
        """Verify that TS factor_product from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nfactor_reactant = 0.4\nfactor_product = 0.6\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.factor_product == 0.6
        finally:
            os.unlink(temp_path)

    def test_ts_optimizer_molbar_from_config(self):
        """Verify that TS optimizer molbar-optimizer from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                '[ts_guess.calculation]\noptimizer = "molbar-optimizer"\n'
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.optimizer == "molbar-optimizer"
        finally:
            os.unlink(temp_path)

    def test_ts_optimizer_scipy_from_config(self):
        """Verify that TS optimizer scipy-optimizer from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                '[ts_guess.calculation]\noptimizer = "scipy-optimizer"\n'
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.optimizer == "scipy-optimizer"
        finally:
            os.unlink(temp_path)

    def test_ts_energy_threshold_two_optimizations_from_config(self):
        """Verify that TS energy_threshold_two_optimizations from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nenergy_threshold_two_optimizations = 0.25\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.energy_threshold_two_optimizations == 0.25
        finally:
            os.unlink(temp_path)

    def test_ts_average_with_hess_weight_from_config(self):
        """Verify that TS average_with_hess_weight from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\naverage_with_hess_weight = false\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.average_with_hess_weight is False
        finally:
            os.unlink(temp_path)

    def test_ts_perform_two_optimizations_from_config(self):
        """Verify that TS perform_two_optimizations from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nperform_two_optimizations = true\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.perform_two_optimizations is True
        finally:
            os.unlink(temp_path)

    def test_ts_molbar_optimizer_e_tol_from_config(self):
        """Verify that TS molbar_optimizer_e_tol from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nmolbar_optimizer_e_tol = 1e-5\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.molbar_optimizer_e_tol == 1e-5
        finally:
            os.unlink(temp_path)

    def test_ts_molbar_optimizer_x_tol_from_config(self):
        """Verify that TS molbar_optimizer_x_tol from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nmolbar_optimizer_x_tol = 1e-6\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.molbar_optimizer_x_tol == 1e-6
        finally:
            os.unlink(temp_path)

    def test_ts_molbar_optimizer_max_micro_steps_from_config(self):
        """Verify that TS molbar_optimizer_max_micro_steps from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                "[ts_guess.calculation]\nmolbar_optimizer_max_micro_steps = 5\n"
            )
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.ts_calc.molbar_optimizer_max_micro_steps == 5
        finally:
            os.unlink(temp_path)


class TestPostprocessingConfig:
    """Test Postprocessing configuration values are correctly loaded from config."""

    def test_postprocessing_relaxation_none_from_config(self):
        """Verify that postprocessing relaxation 'None' from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[postprocessing]\nrelaxation = "None"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.postprocessing.relaxation == "None"
        finally:
            os.unlink(temp_path)

    def test_postprocessing_relaxation_gfn2xtb_from_config(self):
        """Verify that postprocessing relaxation 'gfn2-xtb' from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[postprocessing]\nrelaxation = "gfn2-xtb"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.postprocessing.relaxation == "gfn2-xtb"
        finally:
            os.unlink(temp_path)

    def test_postprocessing_relaxation_pbeh3c_from_config(self):
        """Verify that postprocessing relaxation 'pbeh-3c' from config is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[postprocessing]\nrelaxation = "pbeh-3c"\n')
            f.flush()
            temp_path = f.name
        try:
            calcdata = CalculationData.from_config(temp_path)
            assert calcdata.postprocessing.relaxation == "pbeh-3c"
        finally:
            os.unlink(temp_path)


class TestCompleteConfigLoad:
    """Test that a complete config with all values can be loaded correctly."""

    def test_complete_config_all_values(self):
        """Verify that a complete config with all custom values is loaded correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[system]
charge = -1
multiplicity = 2
xtb_path = "/custom/xtb"
use_gxtb = true
xtb_input_name = "custom"
xtb_alpb_solvent = "water"

[reactant.path]
wbo_filename = "r_wbo"
hessian_filename = "r_hess"
ff_filename = "r_ff.csv"

[reactant.calculation]
geometry_optimization = true
wbo_calc = true
hessian_calc = true
bo_treshold = 0.10
ff_parameterization = true
test_parameterization = true
ff_parameter_repulsion = 0.02
ff_parameterization_maxiteration = 2000
ff_parameterization_stepsize = 0.18
ff_parameterization_threshold = 0.0004
ff_parameterization_constant_repulsion = false

[product.path]
wbo_filename = "p_wbo"
hessian_filename = "p_hess"
ff_filename = "p_ff.csv"

[product.calculation]
geometry_optimization = false
wbo_calc = false
hessian_calc = false
bo_treshold = 0.12
ff_parameterization = false
test_parameterization = false
ff_parameter_repulsion = 0.03
ff_parameterization_maxiteration = 1800
ff_parameterization_stepsize = 0.20
ff_parameterization_threshold = 0.0006
ff_parameterization_constant_repulsion = true

[ts_guess.path]
hessian_filename = "ts_hess"
ff_filename = "ts_ff.csv"

[ts_guess.calculation]
factor_reactant = 0.4
factor_product = 0.6
optimizer = "scipy-optimizer"
energy_threshold_two_optimizations = 0.20
average_with_hess_weight = false
perform_two_optimizations = true
molbar_optimizer_e_tol = 1e-6
molbar_optimizer_x_tol = 1e-5
molbar_optimizer_max_micro_steps = 3

[postprocessing]
relaxation = "gfn2-xtb"
""")
            f.flush()
            temp_path = f.name

        try:
            calcdata = CalculationData.from_config(temp_path, test=True)

            # System
            assert calcdata.system.charge == -1
            assert calcdata.system.multiplicity == 2
            assert calcdata.system.xtb_path == "/custom/xtb"
            assert calcdata.system.use_gxtb is True
            assert calcdata.system.xtb_input_name == "custom"
            assert calcdata.system.xtb_alpb_solvent == "water"

            # Reactant path
            assert calcdata.reactant_path.wbo_filename == "r_wbo"
            assert calcdata.reactant_path.hessian_filename == "r_hess"
            assert calcdata.reactant_path.ff_filename == "r_ff.csv"

            # Reactant calculation
            assert calcdata.reactant_calc.geometry_optimization is True
            assert calcdata.reactant_calc.wbo_calc is True
            assert calcdata.reactant_calc.hessian_calc is True
            assert calcdata.reactant_calc.bo_treshold == 0.10
            assert calcdata.reactant_calc.ff_parameterization is True
            assert calcdata.reactant_calc.test_parameterization is True
            assert calcdata.reactant_calc.ff_parameter_repulsion == 0.02
            assert calcdata.reactant_calc.ff_parameterization_maxiteration == 2000
            assert calcdata.reactant_calc.ff_parameterization_stepsize == 0.18
            assert calcdata.reactant_calc.ff_parameterization_threshold == 0.0004
            assert (
                calcdata.reactant_calc.ff_parameterization_constant_repulsion is False
            )

            # Product path
            assert calcdata.product_path.wbo_filename == "p_wbo"
            assert calcdata.product_path.hessian_filename == "p_hess"
            assert calcdata.product_path.ff_filename == "p_ff.csv"

            # Product calculation
            assert calcdata.product_calc.geometry_optimization is False
            assert calcdata.product_calc.wbo_calc is False
            assert calcdata.product_calc.hessian_calc is False
            assert calcdata.product_calc.bo_treshold == 0.12
            assert calcdata.product_calc.ff_parameterization is False
            assert calcdata.product_calc.test_parameterization is False
            assert calcdata.product_calc.ff_parameter_repulsion == 0.03
            assert calcdata.product_calc.ff_parameterization_maxiteration == 1800
            assert calcdata.product_calc.ff_parameterization_stepsize == 0.20
            assert calcdata.product_calc.ff_parameterization_threshold == 0.0006
            assert calcdata.product_calc.ff_parameterization_constant_repulsion is True

            # TS path
            assert calcdata.ts_path.hessian_filename == "ts_hess"
            assert calcdata.ts_path.ff_filename == "ts_ff.csv"

            # TS calculation
            assert calcdata.ts_calc.factor_reactant == 0.4
            assert calcdata.ts_calc.factor_product == 0.6
            assert calcdata.ts_calc.optimizer == "scipy-optimizer"
            assert calcdata.ts_calc.energy_threshold_two_optimizations == 0.20
            assert calcdata.ts_calc.average_with_hess_weight is False
            assert calcdata.ts_calc.perform_two_optimizations is True
            assert calcdata.ts_calc.molbar_optimizer_e_tol == 1e-6
            assert calcdata.ts_calc.molbar_optimizer_x_tol == 1e-5
            assert calcdata.ts_calc.molbar_optimizer_max_micro_steps == 3

            # Postprocessing
            assert calcdata.postprocessing.relaxation == "gfn2-xtb"
        finally:
            os.unlink(temp_path)
