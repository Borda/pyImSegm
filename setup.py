"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py build_ext --inplace
>> python setup.py install

For uploading to PyPi follow instructions
http://peterdowns.com/posts/first-time-with-pypi.html

Pre-release package
>> python setup.py sdist upload -r pypitest
>> pip install --index-url https://test.pypi.org/simple/ your-package
Release package
>> python setup.py sdist upload -r pypi
>> pip install your-package

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
try:
    from setuptools import setup, Extension, find_packages  # , Command
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension, find_packages  # , Command
    from distutils.command.build_ext import build_ext

import imsegm

# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
# extensions = [Extension("*", "*.pyx")]


TEMP_EGG = '#egg='
HERE = os.path.abspath(os.path.dirname(__file__))


class BuildExt(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE tutorial: https://stackoverflow.com/questions/2379898
    SEE fix: https://stackoverflow.com/questions/19919905
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def _parse_requirements(file_path):
    with open(file_path) as fp:
        reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]
        # parse egg names if there are paths
        reqs = [r[r.index(TEMP_EGG) + len(TEMP_EGG):] if TEMP_EGG in r else r for r in reqs]
        return reqs


setup_reqs = ['Cython', 'numpy<1.17']  # numpy v1.17 drops support for py2
install_reqs = _parse_requirements(os.path.join(HERE, 'requirements.txt'))


setup(
    name='ImSegm',
    version=imsegm.__version__,
    url=imsegm.__homepage__,
    author=imsegm.__author__,
    author_email=imsegm.__author_email__,
    license=imsegm.__license__,
    description='General superpixel image segmentation:'
                ' (un)supervised, center detection, region growing',
    keywords='image segmentation region-growing center-detection ellipse-fitting',

    long_description=imsegm.__doc__,
    long_description_content_type='text/markdown',

    packages=find_packages(
        exclude=['docs', 'notebooks', 'handling_annotations', 'experiments_*']),
    cmdclass={'build_ext': BuildExt},
    ext_modules=[
        Extension('imsegm.features_cython',
                  language='c++',
                  sources=['imsegm/features_cython.pyx'],
                  extra_compile_args=['-O3', '-ffast-math', '-march=native'],
                  # extra_link_args=['-fopenmp'],
                  )
    ],

    setup_requires=setup_reqs,
    install_requires=install_reqs,
    # include_dirs = [np.get_include()],
    include_package_data=True,

    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
