import os
import logging

import matplotlib
import numpy as np

if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

if np.version.full_version >= '1.14.0':
    np.set_printoptions(legacy='1.13')
