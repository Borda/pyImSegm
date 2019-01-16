"""
Unit testing for particular segmentation module

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Add path to root
from imsegm.utilities.data_samples import (IMAGE_LENNA, load_sample_image,
                                           sample_segment_vertical_2d,
                                           sample_segment_vertical_3d)
from imsegm.utilities.data_io import update_path
from imsegm.superpixels import (segment_slic_img2d, make_graph_segm_connect_grid2d_conn4,
                                make_graph_segm_connect_grid3d_conn6)

# set default output path
PATH_OUTPUT = update_path('output', absolute=True)


class TestSuperpixels(unittest.TestCase):

    img = load_sample_image(IMAGE_LENNA)
    seg2d = sample_segment_vertical_2d()
    seg3d = sample_segment_vertical_3d()

    def test_segm_connect(self):
        logging.debug(self.seg2d)
        vertices, edges = make_graph_segm_connect_grid2d_conn4(self.seg2d)
        logging.debug('vertices: {} -> edges: {}'.format(vertices, edges))

        logging.debug(self.seg3d)
        vertices, edges = make_graph_segm_connect_grid3d_conn6(self.seg3d)
        logging.debug('vertices: {} -> edges: {}'.format(vertices, edges))

    def test_general(self):
        slic = segment_slic_img2d(self.img, sp_size=15, relative_compact=0.2)

        logging.debug(np.max(slic))

        vertices, edges = make_graph_segm_connect_grid2d_conn4(slic)
        logging.debug('vertices: %r', vertices)
        logging.debug(len(edges))
        logging.debug('edges: %r', edges)
        fig, axarr = plt.subplots(ncols=2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(slic, cmap=plt.cm.jet)
        fig.savefig(os.path.join(PATH_OUTPUT, 'temp_superpixels.png'))
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
