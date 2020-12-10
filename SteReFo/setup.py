#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
#!/usr/bin/env python

from distutils.core import setup

setup(name='Refnet',
      version='0.0',
      description='Learns to refocus from stereo cameras',
      author='Benjamin Bussam, Matthieu Hog',
      packages=['blur_refinement', 'blur_baseline' , 'refocus_algorithms', 'stereonet'],
	  install_requires=['tensorflow','numpy'],
     )
