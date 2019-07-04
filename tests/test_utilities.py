"""
Unit testing for particular segmentation module


Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest
import logging

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.experiments import try_decorator


class TestUtilities(unittest.TestCase):

    @try_decorator
    def test_try_wrap(self):
        print('%i' % '42')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
