import os

import matplotlib
import numpy as np
import pandas as pd

# in case you are running on machine without display, e.g. server
if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    # https://matplotlib.org/faq/usage_faq.html
    matplotlib.use('Agg')

# parse the numpy versions
np_version = [int(i) for i in np.version.full_version.split('.')]
# comparing strings does not work for version lower 1.10
if np_version >= [1, 14]:
    np.set_printoptions(legacy='1.13')

# default display size was changed in pandas v0.23
pd.set_option('display.max_columns', 20)
