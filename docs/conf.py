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
    "sphinx.ext.napoleon",      # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",      # links to highlighted source
    "sphinx.ext.autosummary",   # generates per-module summary tables
    "myst_parser",              # lets Sphinx read Markdown (for the README)
]

# Auto-generate stub pages for autosummary directives
autosummary_generate = True

# MyST: enable dollar-math so the LaTeX in the README ($$...$$) renders
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# Treat both .rst and .md files as documentation sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]