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

# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
# extensions = [Extension("*", "*.pyx")]


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
        return reqs


HERE = os.path.abspath(os.path.dirname(__file__))
setup_reqs = ['Cython', 'numpy']
install_reqs = _parse_requirements(os.path.join(HERE, 'requirements.txt'))


setup(
    name='ImSegm',
    version='0.1.3',
    url='https://borda.github.io/pyImSegm',

    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    license='BSD 3-clause',
    description='superpixel image segmentation: '
                '(un)supervised, center detection, region growing',

    packages=find_packages(exclude=['docs', 'notebooks',
                                    'handling_annotations',
                                    'experiments_*']),
    cmdclass={'build_ext': BuildExt},
    ext_modules=[Extension('imsegm.features_cython',
                           language='c++',
                           sources=['imsegm/features_cython.pyx'],
                           extra_compile_args=['-O3', '-ffast-math',
                                               '-march=native', '-fopenmp'],
                           extra_link_args=['-fopenmp'],
                           )],
    setup_requires=setup_reqs,
    install_requires=install_reqs,
    # include_dirs = [np.get_include()],
    include_package_data=True,

    long_description="""Image segmentation package contains:
 * supervised and unsupervised segmentation on superpixels using GraphCut,
 * detection object centres and cluster candidates,
 * region growing on superpixel level with a shape prior.""",
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
