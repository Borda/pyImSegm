"""

# RUN: python setup_cython.py build_ext --inplace

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

# extensions = [Extension("*", "*.pyx")]

NAME = 'fast implementation'
AUTHORS = 'Jiri Borovec'
DESC = """
fast implementation of some functions
* computing statistical image features(mean, std, energy, grad)
"""


setup(name = NAME,
      version='0.1',
      author=AUTHORS,
      description=DESC,
      cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize([
          Extension('segmentation/features_cython',
                    language='c++',
                    sources=['segmentation/features_cython.pyx'],
                    extra_compile_args = ["-O3", "-ffast-math",
                                          "-march=native", "-fopenmp" ],
                    extra_link_args=['-fopenmp'],)
      ]),
      include_dirs = [np.get_include()]
)
