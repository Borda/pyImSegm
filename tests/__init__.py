import logging
import os

import imsegm.utilities
from imsegm.utilities.data_io import update_path

logging.basicConfig(level=logging.DEBUG)
imsegm.utilities

PATH_OUTPUT = update_path('output', absolute=True)
# create the folder for visualisations
if not os.path.exists(PATH_OUTPUT):
    os.mkdir(PATH_OUTPUT)
