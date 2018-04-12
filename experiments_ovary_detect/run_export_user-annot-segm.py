"""
This script is a simple one purpose script to visualise the segmented
individual eggs compare to user annotation. Generally it can serve as showing
two different eggs segmentation for the same image

SAMPLE run:
>> python run_export_user-annot-segm.py \
    -imgs "~/Medical-drosophila/ovary_selected_slices/png2/*.png"
    -segs "~/Medical-drosophila/RESULTS/ovary_centers_detect/ellipse/*.png"
    -centers "~/Medical-drosophila/RESULTS/ovary_centers_detect/centre/*.csv"
    -info "~/Medical-drosophila/info_ovary.txt"
    -out "~/Medical-drosophila/RESULTS/ovary_centers_detect/ellipse_user-annot"

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import traceback
import logging
import argparse
import multiprocessing as mproc
from functools import partial


import matplotlib
if os.environ.get('DISPLAY', '') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_data
import imsegm.utils.experiments as tl_expt
import imsegm.utils.drawing as tl_visu
import imsegm.annotation as seg_annot

NB_THREADS = max(1, int(mproc.cpu_count() * 0.8))
PATH_IMAGES = tl_data.update_path(os.path.join('data_images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_data.update_path('results', absolute=True)
PARAMS = {
    'path_images': os.path.join(PATH_IMAGES, 'image', '*.jpg'),
    'path_segms': os.path.join(PATH_IMAGES, 'annot_eggs', '*.png'),
    'path_centers': os.path.join(PATH_IMAGES, 'center_levels', '*.csv'),
    'path_infofile': os.path.join(PATH_IMAGES, 'info_ovary_images.txt'),
    'path_output': os.path.join(PATH_RESULTS, 'export_user-annot-segm'),
}
COLOR_ANNOT = '#ff6100'
COLOR_SEGM = '#00ff00'
# subfugure size for visualisations
FIGURE_SIZE = 12


def arg_parse_params(params):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return ({str: str}, int):
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=params['path_images'])
    parser.add_argument('-segs', '--path_segms', type=str, required=False,
                        help='path to directory & name pattern for segmentation',
                        default=params['path_segms'])
    parser.add_argument('-centers', '--path_centers', type=str, required=False,
                        help='path to directory & name pattern for centres',
                        default=params['path_centers'])
    parser.add_argument('-info', '--path_infofile', type=str, required=False,
                        help='path to the global information file',
                        default=params['path_infofile'])
    parser.add_argument('--stages', type=int, required=False, nargs='+',
                        help='stage indexes', default=[1, 2, 3, 4, 5])
    parser.add_argument('-out', '--path_output', type=str, required=False,
                        help='path to the output directory',
                        default=params['path_output'])
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    arg_params = vars(parser.parse_args())
    params.update(arg_params)
    for k in (k for k in params if 'path' in k):
        params[k] = tl_data.update_path(params[k], absolute=True)
    logging.info('ARG PARAMETERS: \n %s', repr(params))
    return params


def figure_draw_img_centre_segm(fig, img, centres, segm,
                                subfig_size=FIGURE_SIZE):
    """ add to a figure drawing of center
    in case no figure exists, create new one

    :param obj fig:
    :param ndarray img:
    :param [[int]] centres:
    :param ndarray segm:
    :param int subfig_size:
    :return obj:
    """
    if fig is None:
        norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))
        fig, ax = plt.subplots(figsize=(norm_size[::-1] * subfig_size))
        ax.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    else:
        ax = fig.gca()
    ax.contour(segm, levels=np.unique(segm), colors=COLOR_SEGM, linewidths=(3,))
    # ax.plot(centres[:, 0], centres[:, 1], 'o', color='b')
    ax.scatter(centres[:, 0], centres[:, 1], s=500, c=COLOR_SEGM)
    return fig


# def draw_figure_annot_segm(fig, img, annot, subfig_size=FIGURE_SIZE):
#     if fig is None:
#         norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))
#         fig, ax = plt.subplots(figsize=(norm_size[::-1] * subfig_size))
#         ax.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
#     else:
#         ax = fig.gca()
#     ax.contour(annot, levels=np.unique(annot), colors=COLOR_ANNOT,
#                linewidths=(3,))
#     return fig


def figure_draw_annot_csv(fig, img, row_slice, subfig_size=FIGURE_SIZE):
    """ draw from expert annotation stored in info file

    :param obj fig:
    :param ndarray img: backround image
    :param row_slice: line from info file containing annotation
    :param int subfig_size:
    :return obj:
    """
    if fig is None:
        norm_size = np.array(img.shape[:2]) / float(np.max(img.shape))
        fig, ax = plt.subplots(figsize=(norm_size[::-1] * subfig_size))
        ax.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
    else:
        ax = fig.gca()

    pos_ant, pos_lat, pos_post = tl_visu.parse_annot_rectangles(row_slice)

    list_masks = tl_visu.draw_eggs_rectangle(img.shape[:2],
                                             pos_ant, pos_lat, pos_post)
    for mask in list_masks:
        ax.contour(mask, colors=COLOR_ANNOT, linewidths=(3,))

    return fig


# def export_figure_v0(n_img, path_annot, path_out):
#     try:
#         fig, img = load_draw_img_segm_centre(n_img)
#
#         annot = np.array(Image.open(os.path.join(path_annot, n_img + '.png')))
#         fig = draw_figure_annot_segm(fig, img, annot)
#
#         figure_image_adjustment(fig, img)
#         fig.savefig(os.path.join(path_out, n_img + '_segm_user-auto.png'))
#         plt.close(fig)
#     except Exception:
#         print 'error for:', n_img
#         traceback.print_exc()


def export_figure(idx_row, df_slices_info, path_out):
    """ load image, segmentation and csv with centres
    1) draw figure with image, segmentation and csv
    2) draw expety annotation
    3) expert figure

    :param idx_row:
    :param df_slices_info:
    :param path_out:
    """
    _, row = idx_row
    img_name = os.path.splitext(os.path.basename(row['path_image']))[0]

    try:
        if img_name not in df_slices_info.index:
            logging.debug('missing image in annotation - "%s"', img_name)
            return

        img = tl_data.io_imread(row['path_image'])
        segm =tl_data.io_imread(row['path_segm'])
        df = pd.read_csv(os.path.join(row['path_centers']), index_col=0)
        centres = df[['X', 'Y']].values

        fig = figure_draw_img_centre_segm(None, img, centres, segm)

        row_slice = df_slices_info.loc[img_name]
        fig = figure_draw_annot_csv(fig, img, row_slice)

        tl_visu.figure_image_adjustment(fig, img.shape)
        fig.savefig(os.path.join(path_out, img_name + '.png'))
        plt.close(fig)
    except Exception:
        logging.error('failed for: %s', img_name)
        logging.error(traceback.format_exc())


def main(params):
    df_paths = tl_data.find_files_match_names_across_dirs([params['path_images'],
                                                           params['path_segms'],
                                                           params['path_centers']])
    df_paths.columns = ['path_image', 'path_segm', 'path_centers']
    df_paths.index = range(1, len(df_paths) + 1)

    if not os.path.exists(params['path_output']):
        assert os.path.exists(os.path.dirname(params['path_output'])), \
            'missing folder: "%s"' % os.path.dirname(params['path_output'])
        os.mkdir(params['path_output'])

    df_slices_info = seg_annot.load_info_group_by_slices(params['path_infofile'],
                                                         params['stages'])
    _wrapper_export = partial(export_figure, df_slices_info=df_slices_info,
                              path_out=params['path_output'])
    iterate = tl_expt.WrapExecuteSequence(_wrapper_export, df_paths.iterrows(),
                                          nb_jobs=params['nb_jobs'])
    list(iterate)
    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = arg_parse_params(PARAMS)
    main(params)
