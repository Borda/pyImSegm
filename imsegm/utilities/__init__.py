import os
import subprocess
from distutils.version import LooseVersion

try:
    import matplotlib
except ImportError:
    print('Package `matplotlib` which shall be configured are missing...')
else:
    CMD_TRY_MATPLOTLIB = 'python -c "from matplotlib import pyplot; pyplot.close(pyplot.figure())"'
    # in case you are running on machine without display, e.g. server
    if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
        print('No display found. Using non-interactive Agg backend')
        matplotlib.use('Agg')
    # _tkinter.TclError: couldn't connect to display "localhost:10.0"
    elif subprocess.call(CMD_TRY_MATPLOTLIB, stdout=None, stderr=None, shell=True):
        print('Problem with display. Using non-interactive Agg backend')
        matplotlib.use('Agg')


try:
    import numpy as np
except ImportError:
    print('Package `numpy` which shall be configured are missing...')
else:
    # comparing strings does not work for version lower 1.10
    if LooseVersion(np.version.full_version) >= LooseVersion('1.14'):
        # np.set_printoptions(sign='legacy')
        np.set_printoptions(legacy='1.13')


try:
    import pandas as pd
except ImportError:
    print('Package `pandas` which shall be configured are missing...')
else:
    # default display size was changed in pandas v0.23
    pd.set_option('display.max_columns', 20)
