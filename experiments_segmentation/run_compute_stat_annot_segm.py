"""
Compute segmentation statistic against given annotation
Specify the segmentation and annotation folder and optionaly the image folder

>> python run_compute_stat_annot_segm.py \
    -a "data_images/drosophila_ovary_slice/annot_struct/*.png" \
    -s "results/experiment_segm-supervise_ovary/*.png" \
    -i "data_images/drosophila_ovary_slice/image/*.jpg" \
    -o results/evaluation --drop_labels -1 --overlap 0.2 --visual

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import traceback
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import relabel_sequential

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utils.data_io as tl_data
import imsegm.utils.experiments as tl_expt
import imsegm.utils.drawing as tl_visu
import imsegm.labeling as seg_lbs
import imsegm.classification as seg_clf

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
NAME_CVS_OVERALL = 'STATISTIC__%s___Overall.csv'
NAME_CVS_PER_IMAGE = 'STATISTIC__%s___per-Image.csv'
PATH_IMAGES = os.path.join(tl_data.update_path('data_images'),
                           'drosophila_ovary_slice')
PATH_RESULTS = tl_data.update_path('results', absolute=True)
SUFFIX_VISUAL = '___STAT-visual'
PATHS = {
    'annot': os.path.join(PATH_IMAGES, 'annot_struct', '*.png'),
    'segm': os.path.join(PATH_IMAGES, 'segm', '*.png'),
    'image': None,
    'output': os.path.join(PATH_RESULTS, 'stat_annot-segm'),
}


def aparse_params(dict_paths):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return ({str: str}, obj):
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--path_annot', type=str, required=True,
                        help='path to directory with annotations & name pattern',
                        default=dict_paths['annot'])
    parser.add_argument('-s', '--path_segm', type=str, required=True,
                        help='path to directory & name pattern for segmentation',
                        default=dict_paths['segm'])
    parser.add_argument('-i', '--path_image', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=dict_paths['image'])
    parser.add_argument('-o', '--path_output', type=str, required=False,
                        help='path to the output directory',
                        default=dict_paths['output'])
    parser.add_argument('--drop_labels', type=int, required=False, nargs='*',
                        help='list of skipped labels from statistic')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    parser.add_argument('--overlap', type=float, required=False,
                        help='alpha for segmentation', default=0.2)
    parser.add_argument('--relabel', required=False, action='store_true',
                        help='relabel to find label relations', default=False)
    parser.add_argument('--visual', required=False, action='store_true',
                        help='export visualisations', default=False)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    if not isinstance(args['path_image'], str) \
            or args['path_image'].lower() == 'none':
        args['path_image'] = None
    dict_paths = {k.split('_')[-1]:
                      os.path.join(tl_data.update_path(os.path.dirname(args[k])),
                                   os.path.basename(args[k]))
                  for k in args if k.startswith('path_') and args[k] is not None}
    for k in dict_paths:
        assert os.path.isdir(os.path.dirname(dict_paths[k])), \
            'missing: (%s) "%s"' % (k, os.path.dirname(dict_paths[k]))
    if args['drop_labels'] is None:
        args['drop_labels'] = []
    return dict_paths, args


def fill_lut(lut, segm, offset=0):
    uq_lbs = np.unique(lut).tolist()
    for i, lb in enumerate(lut[1:]):
        j = i + 1 + offset
        if lb == 0 and j in segm:
            lut[j] = max(uq_lbs) + 1
            uq_lbs += [lut[j]]
    return lut


def export_visual(n_annot_seg_img, path_out, segm_alpha=1.):
    """ given visualisation of segmented image and annotation

    :param {str: ...} df_row:
    :param str path_out: path to the visualisation directory
    :param [int] drop_labels: whether skip some labels
    """
    name, annot, segm, img = n_annot_seg_img
    # relabel for simpler visualisations of class differences
    if np.sum(annot < 0) > 0:
        annot[annot < 0] = -1
        _, lut, _ = relabel_sequential(annot + 1)
        lut = fill_lut(lut, segm, offset=1)
        annot = lut[annot.astype(int) + 1] - 1
        segm = lut[segm.astype(int) + 1] - 1
    else:
        annot, lut, _ = relabel_sequential(annot)
        lut = fill_lut(lut, segm, offset=0)
        segm = lut[segm.astype(int)]

    # normalise alpha in range (0, 1)
    segm_alpha = tl_visu.norm_aplha(segm_alpha)

    fig = tl_visu.figure_overlap_annot_segm_image(annot, segm, img,
                                                  drop_labels=[-1],
                                                  segm_alpha=segm_alpha)
    logging.debug('>> exporting -> %s', name)
    fig.savefig(os.path.join(path_out, '%s.png' % name))
    plt.close(fig)


def wrapper_relabel_segm(annot_segm):
    annot, segm = annot_segm
    try:
        segm = seg_lbs.relabel_max_overlap_unique(annot, segm)
    except Exception:
        logging.error(traceback.format_exc())
    return segm


def main(dict_paths, visual=True, drop_labels=None, relabel=True,
         segm_alpha=1., nb_jobs=NB_THREADS):
    """ main evaluation

    :param {str: str} dict_paths:
    :param int nb_jobs: number of thred running in parallel
    :param bool relabel: whether relabel segmentation as sequential
    """
    logging.info('running...')
    if not os.path.isdir(dict_paths['output']):
        assert os.path.isdir(os.path.dirname(dict_paths['output'])), \
            'missing folder: %s' % dict_paths['output']
        os.mkdir(dict_paths['output'])

    name = os.path.basename(os.path.dirname(dict_paths['segm']))
    list_dirs = [dict_paths['annot'], dict_paths['segm']]
    if dict_paths.get('image', '') != '':
        list_dirs.append(dict_paths['image'])
    df_paths = tl_data.find_files_match_names_across_dirs(list_dirs)
    path_csv = os.path.join(dict_paths['output'], NAME_CVS_PER_IMAGE % name)
    df_paths.to_csv(path_csv)

    assert len(df_paths) > 0, 'nothing to compare'

    annots, _ = tl_data.load_images_list(df_paths['path_1'].values.tolist())
    segms, names = tl_data.load_images_list(df_paths['path_2'].values.tolist())
    logging.info('loaded %i annots and %i segms', len(annots), len(segms))

    if drop_labels is not None:
        annots = [np.array(annot, dtype=float) for annot in annots]
        for lb in drop_labels:
            for i, annot in enumerate(annots):
                annots[i][annot == lb] = np.nan
    annots = [np.nan_to_num(annot + 1).astype(int) - 1 for annot in annots]
    segms = [seg.astype(int) for seg in segms]

    if relabel:
        logging.info('relabel annotations and segmentations')
        if drop_labels is None:
            annots = [relabel_sequential(annot)[0] for annot in annots]
        iterate = tl_expt.WrapExecuteSequence(wrapper_relabel_segm,
                                              zip(annots, segms),
                                              nb_jobs=nb_jobs, ordered=True,
                                              desc='relabeling')
        segms = list(iterate)

    logging.info('compute statistic per image')
    path_csv = os.path.join(dict_paths['output'], NAME_CVS_PER_IMAGE % name)
    logging.debug('export to "%s"', path_csv)
    df_stat = seg_clf.compute_stat_per_image(segms, annots, names, nb_jobs,
                                             drop_labels=[-1])
    df_stat.to_csv(path_csv)

    logging.info('summarise statistic')
    path_csv = os.path.join(dict_paths['output'], NAME_CVS_OVERALL % name)
    logging.debug('export to "%s"', path_csv)
    df_desc = df_stat.describe()
    df_desc = df_desc.append(pd.Series(df_stat.median(), name='median'))
    logging.info(df_desc.T[['count', 'mean', 'std', 'median']])
    df_desc.to_csv(path_csv)

    if visual:
        images = [None] * len(annots)
        if 'path_3' in df_paths:
            images, _ = tl_data.load_images_list(df_paths['path_3'].values)
        path_visu = os.path.join(dict_paths['output'],
                                 '%s%s' % (name, SUFFIX_VISUAL))
        if not os.path.isdir(path_visu):
            os.mkdir(path_visu)
        # for idx, row in df_paths.iterrows():
        #     export_visual(row, path_visu)
        _wrapper_visual = partial(export_visual, path_out=path_visu,
                                  segm_alpha=segm_alpha)
        it_values = zip(names, annots, segms, images)
        iterate = tl_expt.WrapExecuteSequence(_wrapper_visual, it_values,
                                              desc='visualisations',
                                              nb_jobs=nb_jobs)
        list(iterate)

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dict_paths, args = aparse_params(PATHS)
    main(dict_paths, nb_jobs=args['nb_jobs'], visual=args['visual'],
         drop_labels=args['drop_labels'], relabel=args['relabel'],
         segm_alpha=args['overlap'])
