"""
Run supervised segmentation experiment with superpixels and training examples

Pipeline:

 1. segment SLIC superpixels
 2. compute features (color and texture)
 3. estimate model from single image or whole set
 4. segment new images

.. note:: there are a few constants to that have an impact on the experiment,
 see them bellow with explanation for each of them.

Sample usage::

    python run_segm_slic_model_graphcut.py \
       -l data_images/langerhans_islets/list_lang-isl_imgs-annot.csv \
       -i "data_images/langerhans_islets/image/*.jpg" \
       -o results -n LangIsl --nb_classes 3 --nb_workers 2 --visual

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import pickle
import argparse
import logging
import time
import gc
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from llvmpy._api.llvm.CmpInst import FCMP_OLE
from skimage import segmentation
from sklearn import metrics

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
import imsegm.pipelines as seg_pipe
import imsegm.labeling as seg_lbs
import imsegm.descriptors as seg_fts
# sometimes it freeze in "Cython: computing Colour means for image"
seg_fts.USE_CYTHON = False

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
TYPES_LOAD_IMAGE = ['2d_rgb', '2d_split']
NAME_DUMP_MODEL = 'estimated_model.npz'
NAME_CSV_ARS_CORES = 'metric_ARS.csv'
# setting experiment sub-folders
FOLDER_IMAGE = 'images'
FOLDER_ANNOT = 'annotations'
FOLDER_SEGM_GMM = 'segmentation_MixtureModel'
FOLDER_SEGM_GMM_VISU = FOLDER_SEGM_GMM + '___visual'
FOLDER_SEGM_GROUP = 'segmentation_GroupMM'
FOLDER_SEGM_GROUP_VISU = FOLDER_SEGM_GROUP + '___visual'
LIST_FOLDERS_BASE = (FOLDER_IMAGE, FOLDER_SEGM_GMM, FOLDER_SEGM_GROUP)
LIST_FOLDERS_DEBUG = (FOLDER_SEGM_GMM_VISU, FOLDER_SEGM_GROUP_VISU)

# unique experiment means adding timestemp on the end of folder name
EACH_UNIQUE_EXPERIMENT = False
# showing some intermediate debug images from segmentation
SHOW_DEBUG_IMAGES = True
# relabel annotation such that labels are in sequence no gaps in between them
ANNOT_RELABEL_SEQUENCE = False
# whether skip loading config from previous fun
FORCE_RELOAD = True
# even you have dumped data from previous time, all wil be recomputed
FORCE_RECOMP_DATA = True

FEATURES_SET_COLOR = {'color': ('mean', 'std', 'energy')}
FEATURES_SET_TEXTURE = {'tLM': ('mean', 'std', 'energy')}
FEATURES_SET_ALL = {'color': ('mean', 'std', 'median'),
                    'tLM': ('mean', 'std', 'energy', 'meanGrad')}
FEATURES_SET_MIN = {'color': ('mean', 'std', 'energy'),
                    'tLM_short': ('mean', )}
FEATURES_SET_MIX = {'color': ('mean', 'std', 'energy', 'median'),
                    'tLM': ('mean', 'std')}
# Default parameter configuration
SEGM_PARAMS = {
    'name': 'imgDisk',
    'nb_classes': 3,
    'img_type': '2d_rgb',
    'slic_size': 35,
    'slic_regul': 0.2,
    # 'spacing': (12, 1, 1),
    'features': FEATURES_SET_COLOR,
    'estim_model': 'GMM',
    'pca_coef': None,
    'gc_regul': 2.0,
    'gc_edge_type': 'model',
    'gc_use_trans': False,
}
PATH_IMAGES = os.path.join(tl_data.update_path('data_images'), 'drosophila_disc')
# PATH_IMAGES = tl_data.update_path(os.path.join('data_images', 'langerhans_islets'))
PATH_RESULTS = tl_data.update_path('results', absolute=True)
NAME_EXPERIMENT = 'experiment_segm-unSupervised'
SEGM_PARAMS.update({
    # 'path_train_list': os.path.join(PATH_IMAGES, 'list_imaginal-disks.csv'),
    'path_train_list': '',
    'path_predict_imgs': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    # 'path_predict_imgs': '',
    'path_out': PATH_RESULTS,
})


def arg_parse_params(params):
    """ argument parser from cmd

    SEE: https://docs.python.org/3/library/argparse.html
    :return dict:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--path_train_list', type=str, required=False,
                        help='path to the list of image',
                        default=params['path_train_list'])
    parser.add_argument('-i', '--path_predict_imgs', type=str, required=False,
                        help='path to folder & name pattern with new image',
                        default=params['path_predict_imgs'])
    parser.add_argument('-o', '--path_out', type=str, required=False,
                        help='path to the output directory',
                        default=params['path_out'])
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='name of the experiment', default=params['name'])
    parser.add_argument('-cfg', '--path_config', type=str, required=False,
                        help='path to the segmentation config', default='')
    parser.add_argument('--img_type', type=str, required=False,
                        default=params['img_type'], choices=TYPES_LOAD_IMAGE,
                        help='type of image to be loaded')
    parser.add_argument('--nb_classes', type=int, required=False,
                        help='number of classes for segmentation',
                        default=params.get('nb_classes', 2))
    parser.add_argument('--nb_workers', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    parser.add_argument('--visual', required=False, action='store_true',
                        help='export debug visualisations', default=False)
    parser.add_argument('--unique', required=False, action='store_true',
                        help='each experiment has uniques stamp',
                        default=EACH_UNIQUE_EXPERIMENT)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %r', args)
    for k in (k for k in args if 'path' in k):
        if args[k] == '' or args[k] == 'none':
            continue
        args[k] = tl_data.update_path(args[k])
        p = os.path.dirname(args[k]) if k == 'path_predict_imgs' else args[k]
        assert os.path.exists(p), 'missing: (%s) "%s"' % (k, p)
    # args['visual'] = bool(args['visual'])
    # if the config path is set load the it otherwise use default
    if os.path.isfile(args.get('path_config', '')):
        config = tl_expt.load_config_yaml(args['path_config'])
        params.update(config)
    params.update(args)
    return params


def load_image(path_img, img_type=TYPES_LOAD_IMAGE[0]):
    """ load image and annotation according chosen type

    :param str path_img:
    :param str img_type:
    :return ndarray:
    """
    path_img = tl_data.update_path(path_img)
    assert os.path.isfile(path_img), 'missing: "%s"' % path_img
    if img_type == '2d_split':
        img, _ = tl_data.load_img_double_band_split(path_img)
        assert img.ndim == 2, 'image dims: %r' % img.shape
        # img = np.rollaxis(np.tile(img, (3, 1, 1)), 0, 3)
        # if img.max() > 1:
        #     img = (img / 255.)
    elif img_type == '2d_rgb':
        img, _ = tl_data.load_image_2d(path_img)
        # if img.max() > 1:
        #     img = (img / 255.)
    elif img_type == '2d_segm':
        img, _ = tl_data.load_image_2d(path_img)
        if img.ndim == 3:
            img = img[:, :, 0]
        if ANNOT_RELABEL_SEQUENCE:
            img, _, _ = segmentation.relabel_sequential(img)
    else:
        logging.error('not supported loading img_type: %s', img_type)
        img = None
    return img


def load_model(path_model):
    """ load exported segmentation model

    :param str path_model:
    :return (obj, obj, obj, {}, list(str)):
    """
    logging.info('loading dumped model "%s"', path_model)
    with open(path_model, 'rb') as f:
        dict_data = pickle.load(f)
    # npz_file = np.load(path_model)
    model = dict_data['model']
    params = dict_data['params']
    feature_names = dict_data['feature_names']
    return model, params, feature_names


def save_model(path_model, model, params=None, feature_names=None):
    """ save model on specific destination

    :param str path_model:
    :param obj scaler:
    :param obj pca:
    :param obj model:
    :param dict params:
    :param list(str) feature_names:
    """
    logging.info('save (dump) model to "%s"', path_model)
    # np.savez_compressed(path_model, scaler=scaler, pca=pca,
    #              model=model, params=params, feature_names=feature_names)
    dict_data = dict(model=model, params=params,
                     feature_names=feature_names)
    with open(path_model, 'wb') as f:
        pickle.dump(dict_data, f)


def parse_imgs_idx_path(imgs_idx_path):
    """ general parser for splitting all possible input combination

    :param imgs_idx_path: set of image index and path
    :return (int, str): split index and name
    """
    if isinstance(imgs_idx_path, tuple):
        idx, path_img = imgs_idx_path
    elif isinstance(imgs_idx_path, str):
        idx, path_img = None, imgs_idx_path
    else:
        logging.error('not valid imgs_idx_path -> "%r"', imgs_idx_path)
        idx, path_img = None, ''
    return idx, path_img


def get_idx_name(idx, path_img):
    """ create string identifier for particular image

    :param int idx: image index
    :param str path_img: image path
    :return str: identifier
    """
    im_name = os.path.splitext(os.path.basename(path_img))[0]
    if idx is not None:
        return '%04d_%s' % (idx, im_name)
    else:
        return im_name


def export_visual(idx_name, img, segm, debug_visual=None,
                  path_out=None, path_visu=None):
    """ export visualisations

    :param str idx_name:
    :param ndarray img: input image
    :param ndarray segm: resulting segmentation
    :param debug_visual: dictionary with debug images
    :param str path_out: path to dir with segmentation
    :param str path_visu: path to dir with debug images
    """
    logging.info('export results and visualization...')
    if set(np.unique(segm)) <= {0, 1}:
        segm *= 255

    path_img = os.path.join(path_out, str(idx_name) + '.png')
    logging.debug('exporting segmentation: %s', path_img)
    im_seg = Image.fromarray(segm.astype(np.uint8))
    im_seg.convert('L').save(path_img)
    # io.imsave(path_img, segm)

    if path_visu is not None and os.path.isdir(path_visu):
        path_fig = os.path.join(path_visu, str(idx_name) + '.png')
        logging.debug('exporting segmentation results: %s', path_fig)
        fig = tl_visu.figure_image_segm_results(img, segm)
        fig.savefig(path_fig)
        plt.close(fig)

    if path_visu is not None and os.path.isdir(path_visu) \
            and debug_visual is not None:
        path_fig = os.path.join(path_visu, str(idx_name) + '_debug.png')
        logging.debug('exporting (debug) visualization: %s', path_fig)
        fig = tl_visu.figure_segm_graphcut_debug(debug_visual)
        fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


def segment_image_independent(img_idx_path, params, path_out, path_visu=None,
                              show_debug_imgs=SHOW_DEBUG_IMAGES):
    """ segment image indecently (estimate model just for this model)

    :param (int, str) img_idx_path:
    :param dict params: segmentation parameters
    :param str path_out: path to dir with segmentation
    :param str path_visu: path to dir with debug images
    :return (str, ndarray):
    """
    idx, path_img = parse_imgs_idx_path(img_idx_path)
    logging.debug('segmenting image: "%s"', path_img)
    idx_name = get_idx_name(idx, path_img)
    img = load_image(path_img, params['img_type'])

    path_img = os.path.join(params['path_exp'], FOLDER_IMAGE, idx_name + '.png')
    tl_data.io_imsave(path_img, img.astype(np.uint8))

    debug_visual = dict() if show_debug_imgs else None
    try:
        segm, segm_soft = seg_pipe.pipe_color2d_slic_features_model_graphcut(
            img, nb_classes=params['nb_classes'],
            sp_size=params['slic_size'], sp_regul=params['slic_regul'],
            dict_features=params['features'], estim_model=params['estim_model'],
            pca_coef=params['pca_coef'], gc_regul=params['gc_regul'],
            gc_edge_type=params['gc_edge_type'],
            debug_visual=debug_visual)
        path_npz = os.path.join(path_out, idx_name + '.npz')
        np.savez_compressed(path_npz, segm_soft)
    except Exception:
        logging.exception('pipe_color2d_slic_features_model_graphcut(...)')
        segm = np.zeros(img.shape[:2])

    boundary_size = int(params['slic_size'] * 3)
    segm = seg_lbs.assume_bg_on_boundary(segm, bg_label=0,
                                         boundary_size=boundary_size)

    export_visual(idx_name, img, segm, debug_visual, path_out, path_visu)

    # gc.collect(), time.sleep(1)
    return idx_name, segm


def segment_image_model(imgs_idx_path, params, model, path_out=None,
                        path_visu=None, show_debug_imgs=SHOW_DEBUG_IMAGES):
    """ segment image with already estimated model

    :param (int, str) imgs_idx_path:
    :param dict params: segmentation parameters
    :param obj scaler:
    :param obj pca:
    :param obj model:
    :param str path_out: path to dir with segmentation
    :param str path_visu: path to dir with debug images
    :param bool show_debug_imgs: whether show debug images
    :return (str, ndarray):
    """
    idx, path_img = parse_imgs_idx_path(imgs_idx_path)
    logging.debug('segmenting image: "%s"', path_img)
    idx_name = get_idx_name(idx, path_img)
    img = load_image(path_img, params['img_type'])

    path_img = os.path.join(params['path_exp'], FOLDER_IMAGE, idx_name + '.png')
    tl_data.io_imsave(path_img, img.astype(np.uint8))

    debug_visual = dict() if show_debug_imgs else None

    try:
        segm, segm_soft = seg_pipe.segment_color2d_slic_features_model_graphcut(
            img, model, sp_size=params['slic_size'], sp_regul=params['slic_regul'],
            dict_features=params['features'], gc_regul=params['gc_regul'],
            gc_edge_type=params['gc_edge_type'],
            debug_visual=debug_visual)
        path_npz = os.path.join(path_out, idx_name + '.npz')
        np.savez_compressed(path_npz, segm_soft)
    except Exception:
        logging.exception('segment_color2d_slic_features_model_graphcut(...)')
        segm = np.zeros(img.shape[:2])

    boundary_size = int(np.sqrt(np.prod(segm.shape)) * 0.01)
    segm = seg_lbs.assume_bg_on_boundary(segm, bg_label=0,
                                         boundary_size=boundary_size)

    export_visual(idx_name, img, segm, debug_visual, path_out, path_visu)

    # gc.collect(), time.sleep(1)
    return idx_name, segm


def compare_segms_metric_ars(dict_segm_a, dict_segm_b, suffix=''):
    """ compute ARS for each pair of segmentation

    :param {str: ndarray} dict_segm_a:
    :param {str: ndarray} dict_segm_b:
    :param str suffix:
    :return DF:
    """
    df_ars = pd.DataFrame()
    for n in dict_segm_a:
        if n not in dict_segm_b:
            logging.warning('particular key "%s" is missing in dictionary', n)
            continue
        y_a = dict_segm_a[n].ravel()
        y_b = dict_segm_b[n].ravel()
        dict_ars = {'image': n,
                    'ARS' + suffix: metrics.adjusted_rand_score(y_a, y_b)}
        df_ars = df_ars.append(dict_ars, ignore_index=True)
    df_ars.set_index(['image'], inplace=True)
    return df_ars


def experiment_single_gmm(params, paths_img, path_out, path_visu,
                          show_debug_imgs=SHOW_DEBUG_IMAGES):
    imgs_idx_path = list(zip([None] * len(paths_img), paths_img))
    logging.info('Perform image segmentation as single image in each time')
    _wrapper_segment = partial(segment_image_independent, params=params,
                               path_out=path_out, path_visu=path_visu,
                               show_debug_imgs=show_debug_imgs)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_segment, imgs_idx_path,
                                          nb_workers=params['nb_workers'],
                                          desc='experiment single GMM')
    # dict_segms_gmm = {}
    # for name, segm in iterate:
    #     dict_segms_gmm[name] = segm
    dict_segms_gmm = dict(iterate)
    gc.collect()
    time.sleep(1)
    return dict_segms_gmm


def experiment_group_gmm(params, paths_img, path_out, path_visu,
                         show_debug_imgs=SHOW_DEBUG_IMAGES):
    logging.info('load all images')
    list_images = [load_image(path_img, params['img_type'])
                   for path_img in paths_img]
    imgs_idx_path = list(zip([None] * len(paths_img), paths_img))
    logging.info('Estimate image segmentation from whole sequence of images')
    params['path_model'] = os.path.join(params['path_exp'], NAME_DUMP_MODEL)
    if os.path.isfile(params['path_model']) and not FORCE_RECOMP_DATA:
        model, _, _ = load_model(params['path_model'])
    else:
        model, _ = seg_pipe.estim_model_classes_group(
            list_images, nb_classes=params['nb_classes'],
            dict_features=params['features'], sp_size=params['slic_size'],
            sp_regul=params['slic_regul'], pca_coef=params['pca_coef'],
            model_type=params['estim_model'])
        save_model(params['path_model'], model)

    logging.info('Perform image segmentation from group model')
    _wrapper_segment = partial(segment_image_model, params=params, model=model,
                               path_out=path_out, path_visu=path_visu,
                               show_debug_imgs=show_debug_imgs)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_segment, imgs_idx_path,
                                          nb_workers=params['nb_workers'],
                                          desc='experiment group GMM')
    # dict_segms_group = {}
    # for name, segm in iterate:
    #     dict_segms_group[name] = segm
    dict_segms_group = dict(iterate)
    gc.collect()
    time.sleep(1)
    return dict_segms_group


def load_path_images(params):
    if os.path.isfile(params.get('path_train_list', '')):
        logging.info('loading images from CSV: %s', params['path_train_list'])
        df_paths = pd.read_csv(params['path_train_list'], index_col=0)
        paths_img = df_paths['path_image'].tolist()
    elif 'path_predict_imgs' in params:
        logging.info('loading images from path: %s', params['path_predict_imgs'])
        paths_img = glob.glob(params['path_predict_imgs'])
        if not paths_img:
            logging.warning('no images found on given path...')
    else:
        logging.warning('no images to load!')
        paths_img = []
    return paths_img


def write_skip_file(path_dir):
    assert os.path.isdir(path_dir), 'missing: %s' % path_dir
    with open(os.path.join(path_dir, 'RESULTS'), 'w') as fp:
        fp.write('This particular experiment was skipped by user option.')


def main(params):
    """ the main body containgn two approches:
    1) segment each image indecently
    2) estimate model over whole image sequence and estimate

    :param dict params:
    :return dict:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    show_visual = params.get('visual', False)

    reload_dir_config = os.path.isfile(params['path_config']) or FORCE_RELOAD
    stamp_unique = params.get('unique', EACH_UNIQUE_EXPERIMENT)
    params = tl_expt.create_experiment_folder(params, dir_name=NAME_EXPERIMENT,
                                              stamp_unique=stamp_unique,
                                              skip_load=reload_dir_config)
    tl_expt.set_experiment_logger(params['path_exp'])
    logging.info(tl_expt.string_dict(params, desc='PARAMETERS'))
    tl_expt.create_subfolders(params['path_exp'], LIST_FOLDERS_BASE)
    if show_visual:
        tl_expt.create_subfolders(params['path_exp'], LIST_FOLDERS_DEBUG)

    paths_img = load_path_images(params)
    assert paths_img, 'missing images'

    def _path_expt(n):
        return os.path.join(params['path_exp'], n)

    # Segment as single model per image
    path_visu = _path_expt(FOLDER_SEGM_GMM_VISU) if show_visual else None
    dict_segms_gmm = experiment_single_gmm(params, paths_img,
                                           _path_expt(FOLDER_SEGM_GMM),
                                           path_visu,
                                           show_debug_imgs=show_visual)
    gc.collect()
    time.sleep(1)

    # Segment as model ober set of images
    if params.get('run_groupGMM', False):
        path_visu = _path_expt(FOLDER_SEGM_GROUP_VISU) if show_visual else None
        dict_segms_group = experiment_group_gmm(params, paths_img,
                                                _path_expt(FOLDER_SEGM_GROUP),
                                                path_visu,
                                                show_debug_imgs=show_visual)
    else:
        write_skip_file(_path_expt(FOLDER_SEGM_GROUP))
        # write_skip_file(_path_expt(FOLDER_SEGM_GROUP_VISU))
        dict_segms_group = None

    if dict_segms_group is not None:
        df_ars = compare_segms_metric_ars(dict_segms_gmm, dict_segms_group,
                                          suffix='_gmm-group')
        df_ars.to_csv(_path_expt(NAME_CSV_ARS_CORES))
        logging.info(df_ars.describe())

    return params


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params(SEGM_PARAMS)

    params = main(params)

    logging.info('DONE')
