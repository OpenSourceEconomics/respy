import datetime as dt
import os
import sys


# Set variable so that todos are shown in local build
on_rtd = os.environ.get("READTHEDOCS") == "True"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "respy"
copyright = f"2015-{dt.datetime.now().year}, The respy Development Team"  # noqa: A001
author = "The respy Development Team"

# The full version, including alpha/beta/rc tags.
release = "2.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ------------------------------------------------

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "numpydoc",
    "autoapi.extension",
    "sphinx_tabs.tabs",
]

autodoc_mock_imports = [
    "chaospy",
    "estimagic",
    "hypothesis",
    "joblib",
    "numba",
    "numpy",
    "pandas",
    "pytest",
    "scipy",
    "yaml",
]

extlinks = {
    "ghuser": ("https://github.com/%s", "@"),
    "gh": ("https://github.com/OpenSourceEconomics/respy/pull/%s", "#"),
}

intersphinx_mapping = {
    "numba": ("http://numba.pydata.org/numba-doc/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.8", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
if not on_rtd:
    todo_include_todos = True
    todo_emit_warnings = True

# Configure Sphinx' linkcheck
linkcheck_ignore = [
    r"http://cscubs\.cs\.uni-bonn\.de/*.",
    r"https://(dx\.)?doi\.org/*.",
    r"https://jstor\.org/*.",
    r"https://zenodo\.org/*.",
]

# Configuration for nbsphinx
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. only:: html

    .. nbinfo::

        View and download the notebook `here <https://nbviewer.jupyter.org/github/OpenSourceEconomics/respy/tree/v{{ env.config.release }}/{{ docname }}>`_!

"""

# Configuration for numpydoc
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"type", "optional", "default", "of"}

# Configuration for autodoc
autosummary_generate = True

# Configuration for autoapi
autoapi_type = "python"
autoapi_dirs = ["../respy"]
autoapi_ignore = ["*/tests/*"]


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/OpenSourceEconomics/respy",
    "twitter_url": "https://twitter.com/open_econ",
}

html_css_files = ["css/custom.css"]

html_logo = "_static/images/respy-logo.svg"


html_sidebars = {
    "index": ["sidebar-search-bs.html", "custom-intro.html"],
    "about_us": ["sidebar-search-bs.html", "custom-about-us.html"],
}
