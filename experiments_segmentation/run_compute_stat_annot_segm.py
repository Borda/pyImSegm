"""
Compute segmentation statistic against given annotation
Specify the segmentation and annotation folder and optionaly the image folder

Sample usage::

    python run_compute_stat_annot_segm.py \
        -a "data-images/drosophila_ovary_slice/annot_struct/*.png" \
        -s "results/experiment_segm-supervise_ovary/*.png" \
        -i "data-images/drosophila_ovary_slice/image/*.jpg" \
        -o results/evaluation --drop_labels -1 --overlap 0.2 --visual

Copyright (C) 2016-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import argparse
import logging
import os
import sys
from functools import partial

import matplotlib

if os.environ.get('DISPLAY', '') == '' and matplotlib.rcParams['backend'] != 'agg':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import relabel_sequential

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.experiments as tl_expt
import imsegm.utilities.drawing as tl_visu
import imsegm.classification as seg_clf

NB_WORKERS = tl_expt.nb_workers(0.9)
NAME_CVS_OVERALL = 'STATISTIC__%s___Overall.csv'
NAME_CVS_PER_IMAGE = 'STATISTIC__%s___per-Image.csv'
PATH_IMAGES = os.path.join(tl_data.update_path('data-images'),
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
    parser.add_argument('--nb_workers', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_WORKERS)
    parser.add_argument('--overlap', type=float, required=False,
                        help='alpha for segmentation', default=0.2)
    parser.add_argument('--relabel', required=False, action='store_true',
                        help='relabel to find label relations', default=False)
    parser.add_argument('--visual', required=False, action='store_true',
                        help='export visualisations', default=False)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %r', args)
    if not isinstance(args['path_image'], str) or args['path_image'].lower() == 'none':
        args['path_image'] = None

    _fn_path = lambda k: os.path.join(tl_data.update_path(os.path.dirname(args[k])),
                                      os.path.basename(args[k]))
    dict_paths = {k.split('_')[-1]: _fn_path(k)
                  for k in args if k.startswith('path_') and args[k] is not None}
    for k in dict_paths:
        assert os.path.isdir(os.path.dirname(dict_paths[k])), \
            'missing: (%s) "%s"' % (k, os.path.dirname(dict_paths[k]))
    if not args['drop_labels']:
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


def export_visual(name, annot, segm, img, path_out, drop_labels, segm_alpha=1.):
    """ given visualisation of segmented image and annotation

    :param dict df_row:
    :param str path_out: path to the visualisation directory
    :param [int] drop_labels: whether skip some labels
    """
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
                                                  drop_labels=drop_labels,
                                                  segm_alpha=segm_alpha)
    logging.debug('>> exporting -> %s', name)
    fig.savefig(os.path.join(path_out, '%s.png' % name))
    plt.close(fig)


def stat_single_set(idx_row, drop_labels=None, relabel=False, path_visu='',
                    segm_alpha=1.):
    _, row = idx_row
    path_annot = row['path_1']
    path_segm = row['path_2']
    path_img = row['path_3'] if 'path_3' in row else None

    annot, _ = tl_data.load_image(path_annot)
    segm, name = tl_data.load_image(path_segm)

    if drop_labels is not None:
        annot = np.array(annot, dtype=float)
        for lb in drop_labels:
            annot[annot == lb] = np.nan
    annot = np.nan_to_num(annot + 1).astype(int) - 1

    dc_stat = seg_clf.compute_classif_stat_segm_annot((annot, segm, name),
                                                      drop_labels=[-1],
                                                      relabel=relabel)

    if os.path.isdir(path_visu):
        img, _ = tl_data.load_image(path_img)
        export_visual(name, annot, segm, img, path_visu, drop_labels=[-1],
                      segm_alpha=segm_alpha)

    return dc_stat


def main(dict_paths, visual=True, drop_labels=None, relabel=True,
         segm_alpha=1., nb_workers=NB_WORKERS):
    """ main evaluation

    :param {str: str} dict_paths:
    :param int nb_workers: number of thred running in parallel
    :param bool relabel: whether relabel segmentation as sequential
    """
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
    logging.info('found %i pairs', len(df_paths))
    df_paths.to_csv(path_csv)

    assert not df_paths.empty, 'nothing to compare'

    name_seg_dir = os.path.basename(os.path.dirname(dict_paths['segm']))
    path_visu = os.path.join(dict_paths['output'], name_seg_dir + SUFFIX_VISUAL)
    if visual and not os.path.isdir(path_visu):
        os.mkdir(path_visu)
    elif not visual:
        path_visu = ''

    logging.info('compute statistic per image')
    _wrapper_stat = partial(stat_single_set, drop_labels=drop_labels,
                            relabel=relabel, path_visu=path_visu,
                            segm_alpha=segm_alpha)
    iterate = tl_expt.WrapExecuteSequence(_wrapper_stat, df_paths.iterrows(),
                                          desc='compute statistic',
                                          nb_workers=nb_workers)
    list_stats = list(iterate)
    df_stat = pd.DataFrame(list_stats)

    path_csv = os.path.join(dict_paths['output'], NAME_CVS_PER_IMAGE % name)
    logging.debug('export to "%s"', path_csv)
    df_stat.to_csv(path_csv)

    logging.info('summarise statistic')
    path_csv = os.path.join(dict_paths['output'], NAME_CVS_OVERALL % name)
    logging.debug('export to "%s"', path_csv)
    df_desc = df_stat.describe()
    df_desc = df_desc.append(pd.Series(df_stat.median(), name='median'))
    logging.info(df_desc.T[['count', 'mean', 'std', 'median']])
    df_desc.to_csv(path_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    dict_paths, args = aparse_params(PATHS)
    main(dict_paths, nb_workers=args['nb_workers'], visual=args['visual'],
         drop_labels=args['drop_labels'], relabel=args['relabel'],
         segm_alpha=args['overlap'])

    logging.info('DONE')
