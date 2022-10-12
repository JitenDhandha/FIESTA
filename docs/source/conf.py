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
version = __version__
copyright = '2022, Jiten Dhandha, Zoe Faes, Rowan J. Smith'
author = 'Jiten Dhandha, Zoe Faes, Rowan J. Smith'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_automodapi.automodapi',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax']
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "renku"
html_static_path = ['_static']
