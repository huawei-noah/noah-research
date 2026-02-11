# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'EvoFabric'
copyright = '2025, Huawei Technologies Co., Ltd.'
author = 'Huawei Technologies Co., Ltd.'
release = '0.1.4'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.mermaid',
    'sphinx.ext.ifconfig',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_extra_path = ['../../LICENSE']

locale_dirs = ['../locales/']
gettext_compact = False
