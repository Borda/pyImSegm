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
from imsegm.utilities.data_samples import LIST_ALL_IMAGES, PATH_IMAGES


class TestDataSamples(unittest.TestCase):

    def test_existing_images(self):
        for p in LIST_ALL_IMAGES:
            p = os.path.join(PATH_IMAGES, p)
            assert os.path.exists(p), 'missing: %s' % p


class TestUtilities(unittest.TestCase):

    @try_decorator
    def test_try_wrap(self):
        print('%i' % '42')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
