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
import pip
import logging
import pkg_resources
try:
    from setuptools import setup, Extension #, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension #, find_packages
    from distutils.command.build_ext import build_ext
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize

# extensions = [Extension("*", "*.pyx")]


class CustomBuildExtCommand(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE: https://stackoverflow.com/questions/2379898 """
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        # Call original build_ext command
        build_ext.run(self)


def _parse_requirements(file_path):
    pip_version = list(map(int, pkg_resources.get_distribution('pip').version.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path, session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = ['Cython', 'numpy']


setup(
    name='ImSegm',
    version='0.1',
    url='https://borda.github.com/pyImSegm',

    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    license='BSD 3-clause',
    description='superpixel image segmentation: '
                '(un)supervised, center detection, region growing',

    packages=["segmentation"],
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=[Extension('segmentation.features_cython',
                           language='c++',
                           sources=['segmentation/features_cython.pyx'],
                           extra_compile_args = ['-O3', '-ffast-math',
                                                 '-march=native', '-fopenmp'],
                           extra_link_args=['-fopenmp'],
                           )],
    install_requires=install_reqs,
    # include_dirs = [np.get_include()],
    include_package_data=True,

    long_description="""Image segmentation package contains several useful features:
 * supervised and unsupervised segmentation on superpixels using GrapCut,
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
