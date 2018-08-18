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
import imsegm.utilities.data_samples as d_spl
import imsegm.utilities.data_io as tl_data
import imsegm.labeling as seg_lb

# set the output put directory
PATH_OUTPUT = tl_data.update_path('output', absolute=True)


class TestLabels(unittest.TestCase):

    segm = d_spl.sample_segment_vertical_2d()

    def test_label_contours(self):
        seg = self.segm
        logging.debug('matrix seg_pipe \n%s', repr(seg))
        labs = list(np.unique(seg))
        path_dir = os.path.join(PATH_OUTPUT, 'test_labels')
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        for l in labs:
            fig, axarr = plt.subplots(nrows=2)
            cnt = 1 - seg_lb.binary_image_from_coords(
                seg_lb.contour_coords(seg, l), seg.shape)
            axarr[0].imshow(cnt, interpolation='nearest', cmap=plt.cm.Greys)
            dist = seg_lb.compute_distance_map(seg, l)
            im = axarr[1].imshow(dist, cmap=plt.cm.jet)
            plt.colorbar(im, ax=axarr[1])
            fig.tight_layout()
            fig.savefig(os.path.join(path_dir, 'contours_%i.png' % l))
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                plt.show()
            plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
