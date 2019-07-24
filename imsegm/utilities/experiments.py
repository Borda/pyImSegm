"""
Framework for general experiments

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import copy
import time
import types
import logging
import uuid
import multiprocessing as mproc
from functools import wraps

import yaml
import tqdm
from sklearn import metrics

#: total number of avalaible CPUs/treads
CPU_COUNT = mproc.cpu_count()
#: default date-time format
FORMAT_DT = '%Y%m%d-%H%M%S'
#: default file for loading/exporting experiment configuration
CONFIG_YAML = 'config.yml'
#: default name of file for exporting  statistics
RESULTS_TXT = 'resultStat.txt'
RESULTS_CSV = 'results.csv'
#: default file for streaming experimeny messages
FILE_LOGS = 'logging.txt'


def nb_workers(ratio):
    """get fraction of of available CPUs

    :param float ratio: range (0, 1)
    :return int: number of workers with lower bound 1

    >>> nb_workers(0)
    1
    """
    return max(1, int(CPU_COUNT * ratio))


class Experiment(object):
    """ Basic experiment class

    Example
    -------
    >>> import shutil
    >>> params = {'path_out': './my_experiments', 'name': 'My-Sample'}
    >>> expt = Experiment(params)
    Traceback (most recent call last):
    ...
    Exception: given folder "./my_experiments" does not exist!
    >>> os.mkdir(params['path_out'])
    >>> expt = Experiment(params, time_stamp=False)
    >>> expt.run()
    >>> params = expt.params.copy()
    >>> del expt
    >>> shutil.rmtree(params['path_out'], ignore_errors=True)
    """

    def __init__(self, params, time_stamp=True):
        """ constructor

        :param dict params: define experimenatl parameters
        :param bool time_stamp: add to experiment unique time stamp
        """
        self.params = copy.deepcopy(params)
        self.params['class'] = self.__class__.__name__
        self._check_exist_paths()
        self._create_folder(time_stamp)
        set_experiment_logger(self.params['path_exp'])
        logging.info(string_dict(self.params, desc='PARAMETERS'))

    def run(self, gt=True):
        """ run the main Experimental body

        :param bool gt: try to load Ground Truth
        """
        self._load_data(gt)
        self._perform()
        self._evaluate()
        self._summarise()
        logging.getLogger().handlers = []

    def _load_data(self, gt=True):
        """ loading the experiment data

        :param bool gt: try to load ground truth
        """
        logging.warning('Not implemented yet...')

    def _perform(self):
        logging.warning('Not implemented yet...')

    def _evaluate(self):
        logging.warning('Not implemented yet...')

    def _summarise(self):
        logging.warning('Not implemented yet...')

    def _check_exist_paths(self):
        """ Check all required paths in parameters whether they exist """
        for p in (self.params[n] for n in self.params
                  if 'dir' in n.lower() or 'path' in n.lower()):
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)
        for p in (self.params[n] for n in self.params if 'file' in n.lower()):
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)

    def _create_folder(self, time_stamp=True):
        """ Create the experiment folder and iterate while there is no available

        :param bool time_stamp: mark if you want an unique folder per experiment
        """
        # create results folder for experiments
        if not os.path.exists(self.params.get('path_out', 'NONE')):
            raise ValueError('no results folder "%r"' % self.params.get('path_out', None))
        self.params = create_experiment_folder(self.params,
                                               self.__class__.__name__,
                                               time_stamp)


# def check_exist_dirs_files(params):
#     res = True
#     for p in [params[total] for total in params
#               if 'dir_name' in total.lower() or 'path' in total.lower()]:
#         if not os.path.exists(p):
#             logging.error('given folder "{}" does not exist!'.format(p))
#             res = False
#     for p in [params[total] for total in params if 'file' in total.lower()]:
#         if not os.path.exists(p):
#             logging.error('given file "{}" does not exist!'.format(p))
#             res = False
#     return res


def create_experiment_folder(params, dir_name, stamp_unique=True, skip_load=True):
    """ create the experiment folder and iterate while there is no available

    :param dict params: configuration
    :param str dir_name: folder name
    :param bool stamp_unique: use unique timestamp
    :param bool skip_load: skip loading folder params
    :return dict:

    >>> import shutil
    >>> import pandas as pd
    >>> p = {'path_out': '.'}
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=True)
    >>> pd.Series(p).sort_index()  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    computer                   (...
    path_exp      ./my_test_EXAMPLE
    path_out                      .
    dtype: object
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=False)
    >>> shutil.rmtree(p['path_exp'], ignore_errors=True)
    >>> p = create_experiment_folder(p, 'my_test', stamp_unique=True)
    >>> pd.Series(p).sort_index()  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    computer                         (...
    path_exp    ./my_test_EXAMPLE_...-...
    path_out                            .
    dtype: object
    >>> shutil.rmtree(p['path_exp'], ignore_errors=True)
    """
    date = time.gmtime()
    dir_name = '%s_%s' % (dir_name, str(params.get('name', 'EXAMPLE')))
    path_expt = os.path.join(params.get('path_out'), dir_name)
    # if unique folder is required
    if stamp_unique:
        path_expt += '_' + time.strftime(FORMAT_DT, date)
        if os.path.isdir(path_expt):
            logging.warning('particular out folder already exists')
            path_expt += ':' + str(uuid.uuid4().hex)
    logging.info('creating experiment folder "{}"'.format(path_expt))
    if not os.path.exists(path_expt):
        os.mkdir(path_expt)
    # loading confing if it exists
    path_config = os.path.join(path_expt, CONFIG_YAML)
    if os.path.exists(path_config) and not skip_load:
        params_in = params
        logging.debug('loading saved params from file "%s"', CONFIG_YAML)
        params = load_config_yaml(path_config)
        params.update({k: params_in[k] for k in params_in if 'path' in k})
        logging.info('loaded following PARAMETERS: %s', string_dict(params))
    # extending parameters bu this run
    params.update({'computer': os.uname(), 'path_exp': path_expt})
    # export experiment config
    logging.debug('saving params to file "%s"', CONFIG_YAML)
    save_config_yaml(path_config, params)
    return params


def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file """
    log = logging.getLogger()
    if reset:
        log.handlers = [h for h in log.handlers
                        if not isinstance(h, logging.FileHandler)]
    path_logger = os.path.join(path_out, file_name)
    logging.info('setting logger to "%s"', path_logger)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)


def string_dict(d, offset=30, desc='DICTIONARY'):
    """ transform dictionary to a formatted string

    :param dict d:
    :param int offset: length between name and value
    :param str desc: dictionary title
    :return str:

    >>> string_dict({'abc': 123})  #doctest: +NORMALIZE_WHITESPACE
    \'DICTIONARY: \\n"abc": 123\'
    """
    s = desc + ': \n'
    tmp_name = '{:' + str(offset) + 's} {}'
    rows = [tmp_name.format('"{}":'.format(n), d[n]) for n in sorted(d)]
    s += '\n'.join(rows)
    return str(s)


def append_final_stat(out_dir, y_true, y_pred, time_sec,
                      file_name=RESULTS_TXT):
    """ append (export) statistic to existing default file

    :param str out_dir:
    :param [int] y_true: annotation
    :param [int] y_pred: predictions
    :param int time_sec:
    :param str file_name:
    :return str:

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> y_true = np.random.randint(0, 2, 25)
    >>> y_pred = np.random.randint(0, 2, 25)
    >>> f_path = append_final_stat('.', y_true, y_pred, 256)
    >>> os.path.exists(f_path)
    True
    >>> os.remove(f_path)
    """
    # y_true, y_pred = np.array(y_true), np.array(y_pred)
    logging.debug('export compare labeling sizes %r with %r [px]',
                  y_true.shape, y_pred.shape)
    res = metrics.classification_report(y_true, y_pred, digits=4)
    logging.info('FINAL results: \n {}'.format(res))

    s = '\n\n\nFINAL results: \n {} \n\n' \
        'complete experiment took: {:.1f} min'.format(res, time_sec / 60.)
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, 'a') as fp:
        fp.write(s)
    return file_path


def is_iterable(var):
    """ check if the variable is iterable

    :param var:
    :return bool:

    >>> is_iterable('abc')
    False
    >>> is_iterable([0])
    True
    >>> is_iterable((1, ))
    True
    """
    return any(isinstance(var, cls) for cls in [list, tuple, types.GeneratorType])


def extend_list_params(params, name_param, options):
    """ extend the parameter list by all sub-datasets

    :param list(dict) params: list of parameters
    :param str name_param: parameter name
    :param [] options: lost of options
    :return list(dict):

    >>> import pandas as pd
    >>> params = extend_list_params([{'a': 1}], 'a', [3, 4])
    >>> pd.DataFrame(params)[sorted(pd.DataFrame(params))]  # doctest: +NORMALIZE_WHITESPACE
       a param_idx
    0  3     a-2#1
    1  4     a-2#2
    >>> params = extend_list_params([{'a': 1}], 'b', 5)
    >>> pd.DataFrame(params)[sorted(pd.DataFrame(params))]  # doctest: +NORMALIZE_WHITESPACE
       a  b param_idx
    0  1  5     b-1#1
    """
    if not is_iterable(options):
        options = [options]
    list_params_new = []
    for p in params:
        p['param_idx'] = p.get('param_idx', '')
        for i, v in enumerate(options):
            p_new = p.copy()
            p_new.update({name_param: v})
            if p_new['param_idx']:
                p_new['param_idx'] += '_'
            p_new['param_idx'] += \
                '%s-%i#%i' % (name_param, len(options), i + 1)
            list_params_new.append(p_new)
    return list_params_new


def try_decorator(func):
    """ costume decorator to wrap function in try/except

    :param func:
    :return:
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logging.exception('%r with %r and %r', func.__name__, args, kwargs)
    return wrap


def create_subfolders(path_out, folders):
    """ create subfolders in rood directory

    :param str path_out: root dictionary
    :param list(str) folders: list of subfolders
    :return int:

    >>> import shutil
    >>> dir_name = 'sample_dir'
    >>> create_subfolders('.', [dir_name])
    1
    >>> os.path.exists(dir_name)
    True
    >>> shutil.rmtree(dir_name, ignore_errors=True)
    """
    count = 0
    for dir_name in folders:
        path_dir = os.path.join(path_out, dir_name)
        if not os.path.exists(path_dir):
            try:
                os.mkdir(path_dir)
                count += 1
            except Exception:
                logging.exception('mkdir: %s', path_dir)
    return count


class WrapExecuteSequence:
    """ wrapper for execution paralle of single thread as for ...

    Example
    -------
    >>> it = WrapExecuteSequence(lambda x: (x, x ** 2), range(5),
    ...                          nb_workers=1, ordered=True)
    >>> list(it)
    [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]
    >>> it = WrapExecuteSequence(sum, [[0, 1]] * 5, nb_workers=2, desc=None)
    >>> [o for o in it]
    [1, 1, 1, 1, 1]
    >>> it = WrapExecuteSequence(min, ([0, 1] for i in range(5)))
    >>> [o for o in it]
    [0, 0, 0, 0, 0]
    """

    def __init__(self, wrap_func, iterate_vals, nb_workers=CPU_COUNT, desc='',
                 ordered=False):
        """ the init of this wrapper fro parallelism

        :param wrap_func: function which will be excited in the iterations
        :param [] iterate_vals: list or iterator which will ide in iterations
        :param int nb_workers: number og jobs running in parallel
        :param str desc: deception for the bar,
            if it is set None, bar is suppressed
        :param bool ordered: whether enforce ordering in the parallelism
        """
        self.wrap_func = wrap_func
        self.iterate_vals = list(iterate_vals)
        self.nb_workers = nb_workers
        self.desc = desc
        self.ordered = ordered

    def __iter__(self):
        tqdm_bar = None
        if self.desc is not None:
            desc = '%r @%i-threads' % (self.desc, self.nb_workers)
            tqdm_bar = tqdm.tqdm(total=len(self), desc=desc)

        if self.nb_workers > 1:
            logging.debug('perform parallel in %i threads', self.nb_workers)
            pool = mproc.Pool(self.nb_workers)

            pooling = pool.imap if self.ordered else pool.imap_unordered

            for out in pooling(self.wrap_func, self.iterate_vals):
                tqdm_bar.update() if tqdm_bar is not None else None
                yield out
            pool.close()
            pool.join()
        else:
            for out in map(self.wrap_func, self.iterate_vals):
                tqdm_bar.update() if tqdm_bar is not None else None
                yield out

        tqdm_bar.close() if tqdm_bar is not None else None

    def __len__(self):
        return len(self.iterate_vals)


# def wrap_execute_parallel(wrap_func, iterate_vals,
#                           nb_workers=NB_WORKERS, desc=''):
#     """ wrapper for execution paralle of single thread as for...
#
#     :param func wrap_func:
#     :param [] iterate_vals:
#     :param int nb_workers:
#     :return:
#
#     >>> [o for o in wrap_execute_parallel(lambda x: x ** 2, range(5), 1)]
#     [0, 1, 4, 9, 16]
#     >>> [o for o in wrap_execute_parallel(sum, [[0, 1]] * 5, 2)]
#     [1, 1, 1, 1, 1]
#     """
#     tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)
#     if nb_workers > 1:
#         logging.debug('perform_sequence in %i threads', nb_workers)
#         pool = mproc.Pool(nb_workers)
#         for out in pool.imap_unordered(wrap_func, iterate_vals):
#             yield out
#             tqdm_bar.update()
#         pool.close()
#         pool.join()
#     else:
#         for out in map(wrap_func, iterate_vals):
#             yield out
#             tqdm_bar.update()


def load_config_yaml(path_config):
    """ loading the

    :param str path_config:
    :return dict:

    >>> p_conf = './testing-congif.yaml'
    >>> save_config_yaml(p_conf, {'a': 2})
    >>> load_config_yaml(p_conf)
    {'a': 2}
    >>> os.remove(p_conf)
    """
    with open(path_config, 'r') as fp:
        config = yaml.load(fp)
    return config


def save_config_yaml(path_config, config):
    """ exporting configuration as YAML file

    :param str path_config:
    :param dict config:
    """
    with open(path_config, 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)
