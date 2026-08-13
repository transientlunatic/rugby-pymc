# Configuration file for Sphinx documentation

project = "Rugby Ranking"
copyright = "2026, Daniel Williams"
author = "Daniel Williams"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/transientlunatic/rugby-ranking",
    "show_toc_level": 2,
    "use_edit_url": False,
}

master_doc = "index"
