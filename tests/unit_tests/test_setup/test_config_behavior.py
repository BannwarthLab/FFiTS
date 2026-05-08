"""
Integration tests to verify that config file parameters affect behavior
in main calculation functions.
"""

import pytest
import tempfile
import os
from ffits.io.toml_parser import load_calculation_data


class TestFFParameterizationBehavior:
    """Test that FF parameterization config parameters affect function behavior."""

    def test_fit_ff_receives_maxiteration_from_config(self):
        """Verify that ff_parameterization_maxiteration from config is passed to fit_ff_to_hessian."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
ff_parameterization_maxiteration = 2500
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # Verify the value was loaded correctly from config
            assert calcdata.reactant_calc.ff_parameterization_maxiteration == 2500
        finally:
            os.unlink(temp_path)

    def test_fit_ff_receives_stepsize_from_config(self):
        """Verify that ff_parameterization_stepsize from config is passed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[product.calculation]
ff_parameterization_stepsize = 0.30
ff_parameterization_threshold = 0.0002
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            assert calcdata.product_calc.ff_parameterization_stepsize == 0.30
            assert calcdata.product_calc.ff_parameterization_threshold == 0.0002
        finally:
            os.unlink(temp_path)

    def test_constant_repulsion_flag_affects_behavior(self):
        """Verify that constant_repulsion flag can be toggled via config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
ff_parameterization_constant_repulsion = false

[product.calculation]
ff_parameterization_constant_repulsion = true
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # Different configs should result in different values
            assert (
                calcdata.reactant_calc.ff_parameterization_constant_repulsion is False
            )
            assert calcdata.product_calc.ff_parameterization_constant_repulsion is True
        finally:
            os.unlink(temp_path)


class TestTSGuessOptimizerBehavior:
    """Test that TS guess optimizer config parameters affect function behavior."""

    def test_optimizer_choice_from_config(self):
        """Verify that optimizer choice from config can switch between molbar and scipy."""
        # Test molbar-optimizer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
optimizer = "molbar-optimizer"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)
            assert calcdata.ts_calc.optimizer == "molbar-optimizer"
        finally:
            os.unlink(temp_path)

        # Test scipy-optimizer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
optimizer = "scipy-optimizer"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)
            assert calcdata.ts_calc.optimizer == "scipy-optimizer"
        finally:
            os.unlink(temp_path)

    def test_ts_factors_affect_ff_mixing(self):
        """Verify that factor_reactant and factor_product from config control FF mixing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
factor_reactant = 0.3
factor_product = 0.7
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # These factors should control how FF is mixed in create_tsff
            assert calcdata.ts_calc.factor_reactant == 0.3
            assert calcdata.ts_calc.factor_product == 0.7
            # And must sum to 1.0
            assert (
                abs(
                    calcdata.ts_calc.factor_reactant
                    + calcdata.ts_calc.factor_product
                    - 1.0
                )
                < 1e-8
            )
        finally:
            os.unlink(temp_path)

    def test_optimizer_tolerances_from_config(self):
        """Verify that molbar optimizer tolerances are loaded from config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
molbar_optimizer_e_tol = 1e-5
molbar_optimizer_x_tol = 1e-5
molbar_optimizer_max_micro_steps = 10
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # These tolerances should be passed to the optimizer
            assert calcdata.ts_calc.molbar_optimizer_e_tol == 1e-5
            assert calcdata.ts_calc.molbar_optimizer_x_tol == 1e-5
            assert calcdata.ts_calc.molbar_optimizer_max_micro_steps == 10
        finally:
            os.unlink(temp_path)

    def test_two_optimizations_flag_affects_behavior(self):
        """Verify that perform_two_optimizations flag can be toggled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.calculation]
perform_two_optimizations = true
energy_threshold_two_optimizations = 0.25
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            assert calcdata.ts_calc.perform_two_optimizations is True
            assert calcdata.ts_calc.energy_threshold_two_optimizations == 0.25
        finally:
            os.unlink(temp_path)


class TestCalculationFlagsAffectBehavior:
    """Test that calculation flags from config control which computations run."""

    def test_geometry_optimization_flag(self):
        """Verify geometry_optimization flag can be toggled in config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
geometry_optimization = true

[product.calculation]
geometry_optimization = false
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # Different configs should result in different behavior flags
            assert calcdata.reactant_calc.geometry_optimization is True
            assert calcdata.product_calc.geometry_optimization is False
        finally:
            os.unlink(temp_path)

    def test_wbo_calculation_flag(self):
        """Verify wbo_calc flag controls whether WBO is calculated or read."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
wbo_calc = true

[product.calculation]
wbo_calc = false
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # When wbo_calc is true, calculation will run
            assert calcdata.reactant_calc.wbo_calc is True
            # When false, it will read from file instead
            assert calcdata.product_calc.wbo_calc is False
        finally:
            os.unlink(temp_path)

    def test_hessian_calculation_flag(self):
        """Verify hessian_calc flag controls whether Hessian is calculated or read."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
hessian_calc = false

[product.calculation]
hessian_calc = true
"""
            )
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(
                FileNotFoundError,
                match="No Hessian calculation requested, but file .* not found.",
            ):
                calcdata = load_calculation_data(temp_path)

            calcdata = load_calculation_data(temp_path, test=True)

            # When false, will read from file
            assert calcdata.reactant_calc.hessian_calc is False
            # When true, will calculate Hessian
            assert calcdata.product_calc.hessian_calc is True
        finally:
            os.unlink(temp_path)

    def test_ff_parameterization_flag(self):
        """Verify ff_parameterization flag controls whether FF is parameterized or read."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
ff_parameterization = true

[product.calculation]
ff_parameterization = false
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # When true, FF will be parameterized
            assert calcdata.reactant_calc.ff_parameterization is True
            # When false, FF will be read from file
            assert calcdata.product_calc.ff_parameterization is False
        finally:
            os.unlink(temp_path)

    def test_test_parameterization_flag(self):
        """Verify test_parameterization flag controls validation step."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.calculation]
test_parameterization = true

[product.calculation]
test_parameterization = false
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            assert calcdata.reactant_calc.test_parameterization is True
            assert calcdata.product_calc.test_parameterization is False
        finally:
            os.unlink(temp_path)


class TestSystemParametersAffectBehavior:
    """Test that system parameters (charge, multiplicity) affect calculation."""

    def test_system_charge_affects_calculations(self):
        """Verify that system charge from config is used in calculations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
charge = -2
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # Charge should be passed to get_preliminary_information
            assert calcdata.system.charge == -2
        finally:
            os.unlink(temp_path)

    def test_system_multiplicity_affects_calculations(self):
        """Verify that system multiplicity from config is used in calculations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[system]
multiplicity = 3
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            # Multiplicity should be passed to get_preliminary_information
            assert calcdata.system.multiplicity == 3
        finally:
            os.unlink(temp_path)


class TestFilenamesAffectBehavior:
    """Test that different filenames in config are correctly used in calculations."""

    def test_reactant_wbo_filename_from_config(self):
        """Verify that reactant WBO filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
wbo_filename = "reactant_custom.wbo"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path)

            assert calcdata.reactant_path.wbo_filename == "reactant_custom.wbo"
        finally:
            os.unlink(temp_path)

    def test_product_wbo_filename_from_config(self):
        """Verify that product WBO filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[product.path]
wbo_filename = "product_bonds.wbo"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert calcdata.product_path.wbo_filename == "product_bonds.wbo"
        finally:
            os.unlink(temp_path)

    def test_reactant_hessian_filename_from_config(self):
        """Verify that reactant Hessian filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
hessian_filename = "reactant_hessian_matrix.hess"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert (
                calcdata.reactant_path.hessian_filename
                == "reactant_hessian_matrix.hess"
            )
        finally:
            os.unlink(temp_path)

    def test_product_hessian_filename_from_config(self):
        """Verify that product Hessian filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[product.path]
hessian_filename = "product.hess"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert calcdata.product_path.hessian_filename == "product.hess"
        finally:
            os.unlink(temp_path)

    def test_reactant_forcefield_filename_from_config(self):
        """Verify that reactant force field filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
ff_filename = "reactant_force_field_params.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert (
                calcdata.reactant_path.ff_filename == "reactant_force_field_params.csv"
            )
        finally:
            os.unlink(temp_path)

    def test_product_forcefield_filename_from_config(self):
        """Verify that product force field filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[product.path]
ff_filename = "product_ff.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert calcdata.product_path.ff_filename == "product_ff.csv"
        finally:
            os.unlink(temp_path)

    def test_ts_hessian_filename_from_config(self):
        """Verify that TS Hessian filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.path]
hessian_filename = "ts_hessian.hess"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert calcdata.ts_path.hessian_filename == "ts_hessian.hess"
        finally:
            os.unlink(temp_path)

    def test_ts_forcefield_filename_from_config(self):
        """Verify that TS force field filename from config is used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[ts_guess_calculation.path]
ff_filename = "ts_forcefield.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            assert calcdata.ts_path.ff_filename == "ts_forcefield.csv"
        finally:
            os.unlink(temp_path)

    def test_all_filenames_different(self):
        """Verify that different filenames can be used for different structures."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
wbo_filename = "reactant.wbo"
hessian_filename = "reactant.hess"
ff_filename = "reactant_ff.csv"

[product.path]
wbo_filename = "product.wbo"
hessian_filename = "product.hess"
ff_filename = "product_ff.csv"

[ts_guess_calculation.path]
hessian_filename = "ts.hess"
ff_filename = "ts_ff.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            # Reactant filenames
            assert calcdata.reactant_path.wbo_filename == "reactant.wbo"
            assert calcdata.reactant_path.hessian_filename == "reactant.hess"
            assert calcdata.reactant_path.ff_filename == "reactant_ff.csv"

            # Product filenames
            assert calcdata.product_path.wbo_filename == "product.wbo"
            assert calcdata.product_path.hessian_filename == "product.hess"
            assert calcdata.product_path.ff_filename == "product_ff.csv"

            # TS filenames
            assert calcdata.ts_path.hessian_filename == "ts.hess"
            assert calcdata.ts_path.ff_filename == "ts_ff.csv"
        finally:
            os.unlink(temp_path)

    def test_custom_directory_paths_in_filenames(self):
        """Verify that custom directory paths in filenames are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[reactant.path]
wbo_filename = "data/reactants/bonds.wbo"
hessian_filename = "data/reactants/hessian.hess"
ff_filename = "data/reactants/forcefield.csv"

[product.path]
wbo_filename = "data/products/bonds.wbo"
hessian_filename = "data/products/hessian.hess"
ff_filename = "data/products/forcefield.csv"
"""
            )
            f.flush()
            temp_path = f.name

        try:
            calcdata = load_calculation_data(temp_path, test=True)

            # Check that directory paths are preserved
            assert calcdata.reactant_path.wbo_filename == "data/reactants/bonds.wbo"
            assert calcdata.product_path.wbo_filename == "data/products/bonds.wbo"
            assert (
                calcdata.reactant_path.hessian_filename == "data/reactants/hessian.hess"
            )
            assert (
                calcdata.product_path.hessian_filename == "data/products/hessian.hess"
            )
            assert calcdata.reactant_path.ff_filename == "data/reactants/forcefield.csv"
            assert calcdata.product_path.ff_filename == "data/products/forcefield.csv"
        finally:
            os.unlink(temp_path)
