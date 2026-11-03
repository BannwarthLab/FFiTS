"""
Tests that the config file only accepts known sections and keywords.

Every field of the CalculationData sub-dataclasses is a valid keyword in its
matching TOML section; anything else (typos, removed options, keywords placed
in the wrong section, unknown sections) must raise a ValueError.
"""

import os
from dataclasses import fields

import pytest

from ffits.datatype.calculation_data import (
    CalculationData,
    CalculationOptions,
    PathData,
    Postprocessing,
    System,
    TSCalculationOptions,
)

# TOML section name -> dataclass whose fields are the allowed keywords
SECTIONS = {
    "system": System,
    "reactant.path": PathData,
    "reactant.calculation": CalculationOptions,
    "product.path": PathData,
    "product.calculation": CalculationOptions,
    "ts_guess.path": PathData,
    "ts_guess.calculation": TSCalculationOptions,
    "postprocessing": Postprocessing,
}


def _toml_value(value) -> str:
    """Formats a dataclass default as a TOML value (None becomes a string)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return f'"{value if value is not None else "some_value"}"'


def _load(tmp_path, content: str, test: bool = True) -> CalculationData:
    """Writes content to a config file and loads it."""
    config = tmp_path / "config.toml"
    config.write_text(content)
    return CalculationData.from_config(str(config), test=test)


# =======================
# Valid keywords
# =======================


@pytest.mark.parametrize(
    "section, keyword",
    [(section, f.name) for section, cls in SECTIONS.items() for f in fields(cls)],
)
def test_every_known_keyword_is_accepted(tmp_path, section, keyword):
    """Each dataclass field is accepted as a keyword in its own section."""
    cls = SECTIONS[section]
    default = getattr(cls(), keyword)
    _load(tmp_path, f"[{section}]\n{keyword} = {_toml_value(default)}\n")


def test_all_known_keywords_together_are_accepted(tmp_path):
    """A config containing every keyword of every section at once is accepted."""
    content = ""
    for section, cls in SECTIONS.items():
        content += f"[{section}]\n"
        for f in fields(cls):
            content += f"{f.name} = {_toml_value(getattr(cls(), f.name))}\n"
    _load(tmp_path, content)


def test_default_config_is_accepted():
    """The shipped example config contains only known keywords."""
    path = os.path.join(os.getcwd(), "tests", "examples", "default_config.toml")
    cd = CalculationData.from_config(path)
    assert cd.ts_calc.optimizer == "molbar-optimizer"


def test_empty_config_and_empty_sections_are_accepted(tmp_path):
    """Empty files and empty (sub)sections are valid."""
    _load(tmp_path, "")
    _load(
        tmp_path,
        "[system]\n[reactant]\n[reactant.path]\n[product.calculation]\n"
        "[ts_guess]\n[postprocessing]\n",
    )


def test_known_keyword_value_is_still_applied(tmp_path):
    """The keyword check does not interfere with reading values."""
    cd = _load(
        tmp_path,
        "[system]\ncharge = -1\n"
        "[reactant.calculation]\nff_parameterization = false\n"
        "[ts_guess.calculation]\nfactor_reactant = 0.3\nfactor_product = 0.7\n",
    )
    assert cd.system.charge == -1
    assert cd.reactant_calc.ff_parameterization is False
    assert cd.product_calc.ff_parameterization is True
    assert cd.ts_calc.factor_reactant == 0.3


# =======================
# Unknown / misplaced keywords
# =======================


@pytest.mark.parametrize(
    "content, section, keyword",
    [
        # does not exist anywhere
        ("[system]\nparameterize_ff = true\n", "system", "parameterize_ff"),
        # exists, but in another section
        ("[system]\nff_parameterization = true\n", "system", "ff_parameterization"),
        ("[system]\noptimizer = 'molbar-optimizer'\n", "system", "optimizer"),
        (
            "[reactant.calculation]\nfactor_reactant = 0.5\n",
            "reactant.calculation",
            "factor_reactant",
        ),
        (
            "[product.calculation]\ncharge = 0\n",
            "product.calculation",
            "charge",
        ),
        (
            "[ts_guess.calculation]\nwbo_calc = false\n",
            "ts_guess.calculation",
            "wbo_calc",
        ),
        (
            "[reactant.path]\nhessian_calc = false\n",
            "reactant.path",
            "hessian_calc",
        ),
        (
            "[product.path]\ngeometry_optimization = true\n",
            "product.path",
            "geometry_optimization",
        ),
        (
            "[ts_guess.path]\nmultiplicity = 1\n",
            "ts_guess.path",
            "multiplicity",
        ),
        (
            "[postprocessing]\noptimizer = 'scipy-optimizer'\n",
            "postprocessing",
            "optimizer",
        ),
        # removed in 0.9.2
        (
            "[ts_guess.calculation]\naverage_with_hess_weight = true\n",
            "ts_guess.calculation",
            "average_with_hess_weight",
        ),
        # keyword directly in a parent table instead of its subsection
        ("[reactant]\nwbo_calc = false\n", "reactant", "wbo_calc"),
        ("[ts_guess]\noptimizer = 'scipy-optimizer'\n", "ts_guess", "optimizer"),
    ],
)
def test_unknown_keyword_raises(tmp_path, content, section, keyword):
    """Unknown or misplaced keywords raise, naming the keyword and section."""
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, content)
    message = str(excinfo.value)
    assert f"'{keyword}'" in message
    assert f"[{section}]" in message


@pytest.mark.parametrize(
    "content, typo, suggestion",
    [
        ("[reactant.calculation]\nbo_threshold = 0.1\n", "bo_threshold", "bo_treshold"),
        ("[system]\nmultiplicty = 1\n", "multiplicty", "multiplicity"),
        ("[ts_guess.calculation]\noptimiser = 'x'\n", "optimiser", "optimizer"),
        ("[product.path]\nff_file = 'ff.csv'\n", "ff_file", "ff_filename"),
    ],
)
def test_typo_suggests_close_keyword(tmp_path, content, typo, suggestion):
    """A typo in a keyword suggests the closest allowed keyword."""
    with pytest.raises(ValueError, match=f"did you mean '{suggestion}'") as excinfo:
        _load(tmp_path, content)
    assert f"'{typo}'" in str(excinfo.value)


def test_all_unknown_keywords_are_reported(tmp_path):
    """All unknown keywords of a section are listed in one error."""
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, "[system]\ncharge = 0\nfoo = 1\nbar = 2\n")
    message = str(excinfo.value)
    assert "'foo'" in message and "'bar'" in message
    assert "Unknown keyword(s) ['foo', 'bar']" in message


def test_unknown_keyword_raises_without_test_flag(tmp_path):
    """The check also runs in the normal (non-test) code path."""
    with pytest.raises(ValueError, match="parameterize_ff"):
        _load(tmp_path, "[system]\nparameterize_ff = true\n", test=False)


# =======================
# Unknown / malformed sections
# =======================


@pytest.mark.parametrize(
    "content, section, keyword",
    [
        ("[solvent]\nname = 'water'\n", "the top level", "solvent"),
        ("[ts]\n[ts.calculation]\noptimizer = 'x'\n", "the top level", "ts"),
        ("charge = 0\n", "the top level", "charge"),
        ("[reactant.paths]\nwbo_filename = 'w'\n", "[reactant]", "paths"),
        ("[product.calc]\nwbo_calc = false\n", "[product]", "calc"),
        (
            "[ts_guess.postprocessing]\nrelaxation = 'None'\n",
            "[ts_guess]",
            "postprocessing",
        ),
        ("[system.xtb]\npath = 'xtb'\n", "[system]", "xtb"),
    ],
)
def test_unknown_section_raises(tmp_path, content, section, keyword):
    """Unknown top-level tables and subsections raise."""
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, content)
    message = str(excinfo.value)
    assert f"'{keyword}'" in message
    assert section in message


@pytest.mark.parametrize(
    "content",
    [
        "system = 1\n",
        "postprocessing = 'None'\n",
        "[reactant]\npath = 'struc1.xyz'\n",
        "[ts_guess]\ncalculation = true\n",
    ],
)
def test_section_given_as_value_raises(tmp_path, content):
    """A section written as a plain value instead of a table raises."""
    with pytest.raises(ValueError, match="must be a table"):
        _load(tmp_path, content)
