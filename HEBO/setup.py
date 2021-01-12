# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name        = 'HEBO',
        packages    = setuptools.find_packages(),
        description = 'Heteroscedastic evolutionary bayesian optimisation',
        long_description = long_description,
        install_requires=[
            'numpy>=1.15',
            'pytest>=6.0.2',
            'pandas>=1.0.1',
            'torch>=1.4.0',
            'pymoo>=0.4.1',
            'scikit-learn>=0.22',
            'gpytorch>=1.1.1',
            'GPy>=1.9.9',
            'Sphinx>=3.2.1',
            'nbsphinx',
            'recommonmark',
            'sphinx_rtd_theme',
            'notebook'
            ],
        )
