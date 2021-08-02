# How to contribute

Developing Open Source is great fun! =)

## Development process

Here's the long and short of it:

1. Develop your contribution:

   - Pull the latest changes from upstream::

   ```
      git checkout master
      git pull upstream master
   ```

   - Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name such as 'transform-speedups'::
     ```
      git checkout -b transform-speedups
     ```
   - Commit locally as you progress (`git add` and `git commit`)

1. To submit your contribution:

   - Push your changes back to your fork on GitHub::
     ```
      git push origin transform-speedups
     ```
   - Enter your GitHub username and password (repeat contributors or advanced users can remove this step by connecting to GitHub with SSH. See detailed instructions below if desired).
   - Go to GitHub. The new branch will show up with a green Pull Request  button - click it.

1. Review process:

   - Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation and style.  Every single developer working on the project has their code reviewed, and we've come to see it as friendly conversation from which we all learn and the overall code quality benefits.  Therefore, please don't let the review discourage you from contributing: its only aim is to improve the quality of project, not to criticize (we are, after all, very grateful for the time you're donating!).
   - To update your pull request, make your changes on your local repository and commit. As soon as those changes are pushed up (to the same branch as before) the pull request will update automatically.
   - `Travis-CI <http://travis-ci.org/>`\_\_, a continuous integration service, is triggered after each Pull Request update to build the code, run unit tests, measure code coverage and check coding style (PEP8) of your branch. The Travis tests must pass before your PR can be merged. If Travis fails, you can find out why by clicking on the "failed" icon (red cross) and inspecting the build and test log.
   - A pull request must be approved by two core team members before merging.

## Guidelines

- All code should have tests (see `test coverage`\_ below for more details).
- All code should be documented, to the same
  `standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`\_ as NumPy and SciPy.
- For new functionality, always add an example to the gallery.
- No changes are ever committed without review and approval by two core team members. **Never merge your own pull request.**
- Examples in the gallery should have a maximum figure width of 8 inches.

## Stylistic Guidelines

- Set up your editor to remove trailing whitespace.  Follow `PEP08 <http://www.python.org/dev/peps/pep-0008/>`\_\_.  Check code with pyflakes / flake8.
- Use numpy data types instead of strings (`np.uint8` instead of `"uint8"`).
- Use the following import conventions::
  ```
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import ndimage as ndi

   cimport numpy as cnp  # in Cython code
  ```
- When documenting array parameters, use `image : (M, N) ndarray` and then refer to `M` and `N` in the docstring, if necessary.
- Refer to array dimensions as (plane), row, column, not as x, y, z. See :ref:`Coordinate conventions <numpy-images-coordinate-conventions>` in the user guide for more information.
- Functions should support all input image dtypes.  Use utility functions such as `img_as_float` to help convert to an appropriate type.  The output format can be whatever is most efficient.  This allows us to string together several functions into a pipeline
- Use `Py_ssize_t` as data type for all indexing, shape and size variables in C/C++ and Cython code.
- Wrap Cython code in a pure Python function, which defines the API. This improves compatibility with code introspection tools, which are often not aware of Cython code.
- For Cython functions, release the GIL whenever possible, using `with nogil:`.

## Testing

This package has an extensive test suite that ensures correct execution on your system.  The test suite has to pass before a pull request can be merged, and tests should be added to cover any modifications to the code base.

We make use of the `pytest <https://docs.pytest.org/en/latest/>`\_\_ testing framework, with tests located in the various `tests` folders.

To use `pytest`, ensure that Cython extensions are built and that
the library is installed in development mode::

```
    $ pip install -e .
```

Now, run all tests using::

```
    $ pytest -v pyImSegm
```

Use `--doctest-modules` to run doctests.
For example, run all tests and all doctests using::

```
    $ pytest -v --doctest-modules --with-xunit --with-coverage pyImSegm
```

## Test coverage

Tests for a module should ideally cover all code in that module, i.e., statement coverage should be at 100%.

To measure the test coverage, install `pytest-cov <http://pytest-cov.readthedocs.io/en/latest/>`\_\_ (using `easy_install pytest-cov`) and then run::

```
  $ coverage report
```

This will print a report with one line for each file in `imsegm`,
detailing the test coverage::

```
  Name                             Stmts   Exec  Cover   Missing
  --------------------------------------------------------------
  package/module1                     77     77   100%
  package/__init__                     1      1   100%
  ...
```

## Bugs

Please `report bugs on GitHub <https://github.com/Borda/pyImSegm/issues>`\_.
