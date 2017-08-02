import os
import logging
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')