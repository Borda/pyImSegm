"""
The build/compilations setup

>> python setup.py build_ext --inplace

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

# extensions = [Extension("*", "*.pyx")]

extension_features = Extension(
    'segmentation.features_cython',
    language='c++',
    sources=['segmentation/features_cython.pyx'],
    extra_compile_args = ['-O3', '-ffast-math',
                          '-march=native', '-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='ImSegm',
    version='0.1',
    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    url='https://github.com/Borda/pyImSegm',
    license='BSD 3-clause',
    description='superpixel image segmentation: '
                '(un)supervised, center detection, region growing',
    long_description="""
Image segmentation package contains several useful features:
 * supervised and unsupervised segmentation on superpixels using GrapCut,
 * detection object centres and cluster candidates,
 * region growing on superpixel level with a shape prior.
""",
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize([extension_features]),
    include_dirs = [np.get_include()],
    packages=["segmentation"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
