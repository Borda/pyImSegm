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
import imsegm.utils.data_samples as d_spl
import imsegm.utils.data_io as tl_data
import imsegm.superpixels as seg_spx

# set default output path
PATH_OUTPUT = tl_data.update_path('output', absolute=True)


class TestSuperpixels(unittest.TestCase):

    img = d_spl.load_sample_image(d_spl.IMAGE_LENNA)
    seg2d = d_spl.sample_segment_vertical_2d()
    seg3d = d_spl.sample_segment_vertical_3d()

    def test_segm_connect(self):
        logging.debug(self.seg2d)
        vertices, edges = seg_spx.make_graph_segm_connect2d_conn4(self.seg2d)
        logging.debug('vertices: {} -> edges: {}'.format(vertices, edges))

        logging.debug(self.seg3d)
        vertices, edges = seg_spx.make_graph_segm_connect3d_conn6(self.seg3d)
        logging.debug('vertices: {} -> edges: {}'.format(vertices, edges))

    def test_general(self):
        slic = seg_spx.segment_slic_img2d(self.img, sp_size=15, rltv_compact=0.2)

        logging.debug(np.max(slic))

        vertices, edges = seg_spx.make_graph_segm_connect2d_conn4(slic)
        logging.debug(repr(vertices))
        logging.debug(len(edges))
        logging.debug(repr(edges))

        fig, axarr = plt.subplots(ncols=2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(slic, cmap=plt.cm.jet)
        fig.savefig(os.path.join(PATH_OUTPUT, 'test_superpixels.png'))
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
