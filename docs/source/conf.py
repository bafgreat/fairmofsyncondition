# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.setrecursionlimit(1500)
sys.path.insert(0, os.path.abspath('../fairmofsyncondition'))

project = 'fairmofsyncondition'
copyright = '2025, Dinga Wonanke'
author = 'Dinga Wonanke'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
]
html_static_path = ['_static']
html_css_files = [
    'style.css',
]
templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

master_doc = 'index'

html_use_index = True
html_use_search = True
html_show_sourcelink = True
html_show_copyright = True
html_show_powered_by = True
add_module_names = False
todo_include_todos = True
