"""
Framework for general experiments

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import json
import copy
import time
import types
import logging
import multiprocessing as mproc
# from functools import partial

import tqdm
import numpy as np
# import pandas as pd
from sklearn import metrics

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
FILE_RESULTS = 'resultStat.txt'
FORMAT_DT = '%Y%m%d-%H%M%S'
CONFIG_JSON = 'config.json'
RESULTS_TXT = FILE_RESULTS
RESULTS_CSV = 'results.csv'
FILE_LOGS = 'logging.txt'


class Experiment(object):
    """

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

    def __init__(self, dict_params, time_stamp=True):
        self.params = copy.deepcopy(dict_params)
        self.params['class'] = self.__class__.__name__
        self.__check_exist_path()
        self.__create_folder(time_stamp)
        set_experiment_logger(self.params['path_exp'])
        logging.info(string_dict(self.params, desc='PARAMETERS'))

    def run(self, gt=True):
        self._load_data(gt)
        self._perform()
        self._evaluate()
        self._summarise()
        logging.getLogger().handlers = []

    def _load_data(self, gt):
        pass

    def _perform(self):
        pass

    def _evaluate(self):
        pass

    def _summarise(self):
        pass

    def __check_exist_path(self):
        for p in [self.params[n] for n in self.params
                  if 'dir_name' in n.lower() or 'path' in n.lower()]:
            if not os.path.exists(p):
                raise Exception('given folder "{}" does not exist!'.format(p))
        for p in [self.params[n] for n in self.params if 'file' in n.lower()]:
            if not os.path.exists(p):
                raise Exception('given file "{}" does not exist!'.format(p))

    def __create_folder(self, time_stamp):
        """ create the experiment folder and iterate while there is no available
        """
        # create results folder for experiments
        assert os.path.isdir(self.params.get('path_out')), \
            'missing %s' % self.params.get('path_out')
        self.params = create_experiment_folder(self.params,
                                               self.__class__.__name__,
                                               stamp_unique=time_stamp)


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

    :param {str: any} params:
    :param str dir_name: folder name
    :param bool stamp_unique: use unique timestamp
    :param bool skip_load: skip loading folder params
    :return {str: ...}:

    >>> import shutil
    >>> p = {'path_out': '.'}
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=True)
    >>> 'computer' in p
    True
    >>> p['path_exp']
    './my_test_EXAMPLE'
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=False)
    >>> p['path_exp']
    './my_test_EXAMPLE'
    >>> shutil.rmtree(p['path_exp'], ignore_errors=True)
    """
    date = time.gmtime()
    name = params.get('name', 'EXAMPLE')
    if isinstance(name, str) and len(name) > 0:
        dir_name = '{}_{}'.format(dir_name, name)
    # if self.params.get('date_time') is None:
    #     self.params.set('date_time', time.gmtime())
    if stamp_unique:
        dir_name += '_' + time.strftime(FORMAT_DT, date)
    path_expt = os.path.join(params.get('path_out'), dir_name)
    while stamp_unique and os.path.exists(path_expt):
        logging.warning('particular out folder already exists')
        path_expt += ':' + str(np.random.randint(0, 9))
    logging.info('creating experiment folder "{}"'.format(path_expt))
    if not os.path.exists(path_expt):
        os.mkdir(path_expt)
    path_config = os.path.join(path_expt, CONFIG_JSON)
    if os.path.exists(path_config) and not skip_load:
        params_in = params
        logging.debug('loading saved params from file "%s"', CONFIG_JSON)
        with open(path_config, 'r') as fp:
            params = json.load(fp)
        params.update({k: params_in[k] for k in params_in if 'path' in k})
        logging.info('loaded following PARAMETERS: %s', string_dict(params))
    params.update({'computer': os.uname(),
                   'path_exp': path_expt})
    logging.debug('saving params to file "%s"', CONFIG_JSON)
    with open(path_config, 'w') as f:
        json.dump(params, f)
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

    :param {} d:
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
                      file_name=FILE_RESULTS):
    """ append (export) statistic to existing default file

    :param str out_dir:
    :param [int] y_true: annotation
    :param [int] y_pred: predictions
    :param int time_sec:
    :param str file_name:
    :return str:

    >>> np.random.seed(0)
    >>> y_true = np.random.randint(0, 2, 25)
    >>> y_pred = np.random.randint(0, 2, 25)
    >>> f_path = append_final_stat('.', y_true, y_pred, 256)
    >>> os.path.exists(f_path)
    True
    >>> os.remove(f_path)
    """
    # y_true, y_pred = np.array(y_true), np.array(y_pred)
    logging.debug('export compare labeling sizes {} with {} [px]'.format(
                    y_true.shape, y_pred.shape))
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


def extend_list_params(list_params, name_param, list_options):
    """ extend the parameter list by all sub-datasets

    :param [{str: ...}] list_params:
    :param str name_param:
    :param [] list_options:
    :return [{str: ...}]:

    >>> import pandas as pd
    >>> params = extend_list_params([{'a': 1}], 'a', [3, 4])
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       a param_idx
    0  3     a-2#1
    1  4     a-2#2
    >>> params = extend_list_params([{'a': 1}], 'b', 5)
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       a  b param_idx
    0  1  5     b-1#1
    """
    if not is_iterable(list_options):
        list_options = [list_options]
    list_params_new = []
    for p in list_params:
        p['param_idx'] = p.get('param_idx', '')
        for i, v in enumerate(list_options):
            p_new = p.copy()
            p_new.update({name_param: v})
            if len(p_new['param_idx']) > 0:
                p_new['param_idx'] += '_'
            p_new['param_idx'] += \
                '%s-%i#%i' % (name_param, len(list_options), i + 1)
            list_params_new.append(p_new)
    return list_params_new


def create_subfolders(path_out, list_folders):
    """ create subfolders in rood directory

    :param str path_out: root dictionary
    :param [str] list_folders: list of subfolders
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
    for dir_name in list_folders:
        path_dir = os.path.join(path_out, dir_name)
        if not os.path.exists(path_dir):
            try:
                os.mkdir(path_dir)
                count += 1
            except Exception:
                logging.exception('mkdir: %s', path_dir)
    return count


class WrapExecuteSequence:
    """ wrapper for execution paralle of single thread as for...

    >>> it = WrapExecuteSequence(lambda x: (x, x ** 2), range(5),
    ...                          nb_jobs=1, ordered=True)
    >>> list(it)
    [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]
    >>> it = WrapExecuteSequence(sum, [[0, 1]] * 5, nb_jobs=2, desc=None)
    >>> [o for o in it]
    [1, 1, 1, 1, 1]
    >>> it = WrapExecuteSequence(min, ([0, 1] for i in range(5)))
    >>> [o for o in it]
    [0, 0, 0, 0, 0]
    """

    def __init__(self, wrap_func, iterate_vals, nb_jobs=NB_THREADS, desc='',
                 ordered=False):
        """ the init of this wrapper fro parallelism

        :param wrap_func: function which will be excited in the iterations
        :param [] iterate_vals: list or iterator which will ide in iterations
        :param int nb_jobs: number og jobs running in parallel
        :param str desc: deception for the bar,
            if it is set None, bar is suppressed
        :param bool ordered: whether enforce ordering in the parallelism
        """
        self.wrap_func = wrap_func
        self.iterate_vals = list(iterate_vals)
        self.nb_jobs = nb_jobs
        self.desc = desc
        self.ordered = ordered

    def __iter__(self):
        if self.desc is not None:
            tqdm_bar = tqdm.tqdm(total=len(self), desc=self.desc)
        else:
            tqdm_bar = None

        if self.nb_jobs > 1:
            logging.debug('perform parallel in %i threads', self.nb_jobs)
            pool = mproc.Pool(self.nb_jobs)

            pooling = pool.imap if self.ordered else pool.imap_unordered

            for out in pooling(self.wrap_func, self.iterate_vals):
                yield out
                if tqdm_bar is not None:
                    tqdm_bar.update()
            pool.close()
            pool.join()
        else:
            for out in map(self.wrap_func, self.iterate_vals):
                yield out
                if tqdm_bar is not None:
                    tqdm_bar.update()

    def __len__(self):
        return len(self.iterate_vals)


# def wrap_execute_parallel(wrap_func, iterate_vals,
#                           nb_jobs=NB_THREADS, desc=''):
#     """ wrapper for execution paralle of single thread as for...
#
#     :param func wrap_func:
#     :param [] iterate_vals:
#     :param int nb_jobs:
#     :return:
#
#     >>> [o for o in wrap_execute_parallel(lambda x: x ** 2, range(5), 1)]
#     [0, 1, 4, 9, 16]
#     >>> [o for o in wrap_execute_parallel(sum, [[0, 1]] * 5, 2)]
#     [1, 1, 1, 1, 1]
#     """
#     tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)
#     if nb_jobs > 1:
#         logging.debug('perform_sequence in %i threads', nb_jobs)
#         pool = mproc.Pool(nb_jobs)
#         for out in pool.imap_unordered(wrap_func, iterate_vals):
#             yield out
#             tqdm_bar.update()
#         pool.close()
#         pool.join()
#     else:
#         for out in map(wrap_func, iterate_vals):
#             yield out
#             tqdm_bar.update()
