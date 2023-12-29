# Configuration file for the Sphinx documentation builder.
#
# Taken from https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import bdms

sys.path.insert(0, os.path.abspath(".."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "BDMS"
author = "William DeWitt"
copyright = "2023, William DeWitt"

# The short X.Y version
version = bdms.__version__
# The full version, including alpha/beta/rc tags
release = bdms.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autodoc",
    # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    # Link to other project's documentation (see mapping below)
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # Add a link to the Python source code for classes, functions etc.
    # NOTE: this is off to avoid a bug in sphinx parsing of ETE3 code.
    # "sphinx.ext.viewcode",
    # Automatically document param types (less noise in class signature)
    # NOTE: this disables autodoc_type_aliases used below (i.e.
    #       numpy.typing.ArrayLike are not properly condensed).
    "sphinx_autodoc_typehints",
    # support NumPy and Google style docstrings
    "sphinx.ext.napoleon",
    # autoapi
    "autoapi.extension",
    # track to do list items
    "sphinx.ext.todo",
    # Copy button for code blocks
    "sphinx_copybutton",
    # jupyter notebooks
    "myst_nb",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
]

autoapi_dirs = ["../bdms"]
autoapi_options = [
    "members",
    "inherited-members",
    # "undoc-members",  # include members without docstrings
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = False  # set to True to debug autoapi generated files
autodoc_typehints = "description"
templates_path = ["_templates"]  # the usual place for custom sphinx templates
autoapi_template_dir = "_templates/autoapi"  # custom autoapi templates

# remove the autoapi main index page, which we don't use
autoapi_add_toctree_entry = False

inheritance_graph_attrs = dict(
    dpi=200, rankdir="LR", size='"6.0, 0.5"', fontsize=14, ratio="compress"
)
inheritance_node_attrs = dict(
    shape="box",
    fontcolor="white",
    fontsize=14,
    height=0.75,
    color="cornflowerblue",
    style="filled",
    fillcolor="cornflowerblue",
)
inheritance_edge_attrs = dict(penwidth=2.0, color="cornflowerblue")

# options for myst
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ["dollarmath"]
nb_execution_mode = (
    "off"  # NOTE: this is off because we are committing executed notebooks for testing
)
nb_execution_allow_errors = False
nb_merge_streams = True

# show todos in output
todo_include_todos = True

# mappings for sphinx.ext.intersphinx. Projects have to have Sphinx docs (.inv file).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "ete3": ("http://etetoolkit.org/docs/latest/", None),
}

napoleon_use_rtype = False  # More compact, e.g. "Returns" vs. "Return type"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "show_toc_level": 2,
    # "repository_url": "https://github.com/WSDeWitt/BDMS",
    # "use_repository_button": True,  # add a "link to repository" button
    # "show_navbar_depth": 2,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

# https://stackoverflow.com/questions/67473396/shorten-display-format-of-python-type-annotations-in-sphinx
# NOTE: the sphinx_autodoc_typehints extentension above disables this,
#       so aliases are not properly condensed.
autodoc_type_aliases = {"numpy.typing.ArrayLike": "ArrayLike"}

python_use_unqualified_type_names = True
