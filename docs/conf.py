# Configuration file for the Sphinx documentation builder.

import os
import sys

# Make the package importable so autodoc can find it
sys.path.insert(0, os.path.abspath(".."))

project = "FFiTS"
copyright = "2026, Daria Babushkina"
author = "Daria Babushkina"
release = "0.3.1-alpha"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # links to highlighted source
    "sphinx.ext.autosummary",  # generates per-module summary tables
    "myst_parser",  # lets Sphinx read Markdown (for the README)
]

# Auto-generate stub pages for autosummary directives
autosummary_generate = True
autodoc_mock_imports = [
    "ffits.forcefield.fortran.fortran_forcefield",
]

# By default autodoc only shows a class's own docstring, not its methods.
# These options make it recurse into members so methods/functions inside
# classes actually show up.
autodoc_default_options = {
    "members": True,  # include methods/attributes, not just the class docstring
    "undoc-members": True,  # include members even if they have no docstring yet
    "show-inheritance": True,  # show base classes
    "inherited-members": True,  # include methods inherited from base classes
}

# Show type hints in the description body rather than cluttering the signature
autodoc_typehints = "description"

# Some functions default to constructing a full object (e.g.
# `calcdata: CalculationData = CalculationData.from_default()`) or a
# function reference (e.g. `optimizer: Callable = anc_optimizer`).
# Without this, autodoc evaluates those defaults and renders their full
# repr() in the signature (a huge wall of nested dataclass fields, or
# `<function anc_optimizer at 0x...>`). This makes autodoc show the
# literal source expression instead, e.g. `CalculationData.from_default()`
# and `anc_optimizer`.
autodoc_preserve_defaults = True

# MyST: enable dollar-math so the LaTeX in the README ($$...$$) renders
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# README.md links to its own sections with GitHub-style anchors, e.g.
# [Configuration File](#configuration-file). Auto-generate matching
# heading anchors so those in-page links resolve here too.
myst_heading_anchors = 3

# Treat both .rst and .md files as documentation sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# README.md (included on the docs landing page, see index.md) references
# logo.png with a plain relative HTML <img> tag, so unlike a Markdown
# image it won't get its path rewritten or copied automatically by
# Sphinx/MyST. Copy it into the build root verbatim so that reference
# keeps resolving.
html_extra_path = ["../logo.png"]
