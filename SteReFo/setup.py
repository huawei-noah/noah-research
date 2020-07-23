#!/usr/bin/env python

from distutils.core import setup

setup(name='Refnet',
      version='0.0',
      description='Learns to refocus from stereo cameras',
      author='Benjamin Bussam, Matthieu Hog',
      packages=['blur_refinement', 'blur_baseline' , 'refocus_algorithms', 'stereonet'],
	  install_requires=['tensorflow','numpy'],
     )