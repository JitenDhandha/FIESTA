# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from fiesta import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fiesta'
master_doc = 'index'
version = __version__
copyright = '2022, Jiten Dhandha, Zoe Faes, Rowan J. Smith'
author = 'Jiten Dhandha, Zoe Faes, Rowan J. Smith'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'myst_parser']

#Autosummary
autosummary_generate = True
#Napolean
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_ivar = True
#Intersphinx
intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.org/stable', None),
                       'astropy': ('https://docs.astropy.org/en/stable', None)}

templates_path = ['_templates']
exclude_patterns = []
#This sets the role of `text` markup in docstrings
default_role = 'obj'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "show_navbar_depth": 2,
    "repository_url": "https://github.com/JitenDhandha/fiesta",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "secondary_sidebar_items": []
}
html_static_path = ['_static']
html_css_files = ['custom.css']