# Sphinx config for Conway's Game of Life (conway.py)
import os
import sys

sys.path.insert(0, os.path.abspath("../asm2"))

project = "Conway's Game of Life"
copyright = "2025"
author = "Mahesh Venkitachalam"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {"members": True, "undoc-members": True}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]
