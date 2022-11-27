# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from fiesta import __version__

try:
    from sphinx_astropy.conf import *
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)

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
              'sphinx.ext.intersphinx']

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
  "github_url": "https://github.com/JitenDhandha/fiesta",
  "primary_sidebar_end": ["indices.html"],
  "secondary_sidebar_items": ["sourcelink"],
  "show_prev_next": False,
  "show_nav_level": 2,
}
html_sidebars = {
    "**": ["search-field.html", "localtoc.html", "sidebar-nav-bs.html"]
}
html_domain_indices = False
html_title = "%s v%s" % (project, version)
html_context = {"default_mode": "dark"}

html_static_path = ['_static']
html_css_files = ['custom.css']