"""
With two given folder find image match and compute segmentation statistic

>> python run_compute_stat_annot_segm.py \
    -annot "images/drosophila_ovary_slice/annot_struct/*.png" \
    -segm "results/experiment_segm-supervise_ovary/*.png" \
    -img "images/drosophila_ovary_slice/image/*.jpg" \
    -out results/evaluation

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import traceback
import multiprocessing as mproc
from functools import partial

import tqdm
from skimage.segmentation import relabel_sequential

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import segmentation.utils.data_io as tl_io
import segmentation.utils.drawing as seg_visu
import segmentation.labeling as seg_lbs
import segmentation.classification as seg_clf

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
NAME_CVS_OVERALL = 'segm-STATISTIC_%s_stat-overall.csv'
NAME_CVS_PER_IMAGE = 'segm-STATISTIC_%s_stat-per-images.csv'
PATH_IMAGES = tl_io.update_path(os.path.join('images', 'drosophila_ovary_slice'))
PATH_RESULTS = tl_io.update_path('results', absolute=True)
PATHS = {
    'annot': os.path.join(PATH_IMAGES, 'annot_struct', '*.png'),
    'segm': os.path.join(PATH_IMAGES, 'segm', '*.png'),
    'image': None,
    'output': os.path.join(PATH_RESULTS, 'stat_annot-segm'),
}


def aparse_params(dict_paths):
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return: {str: str}, int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-annot', '--path_annot', type=str, required=False,
                        help='path to directory with annotations & name pattern',
                        default=dict_paths['annot'])
    parser.add_argument('-segm', '--path_segm', type=str, required=False,
                        help='path to directory & name pattern for segmentation',
                        default=dict_paths['segm'])
    parser.add_argument('-imgs', '--path_image', type=str, required=False,
                        help='path to directory & name pattern for images',
                        default=dict_paths['image'])
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to the output directory',
                        default=dict_paths['output'])
    args = parser.parse_args()
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    dict_paths = {
        'annot': tl_io.update_path(args.path_annot),
        'segm': tl_io.update_path(args.path_segm),
        'image': '',
        'output': tl_io.update_path(args.path_out),
    }
    if isinstance(args.path_image, str) and args.path_image.lower() != 'none':
        dict_paths['image'] = tl_io.update_path(args.path_image)
    for k in dict_paths:
        if dict_paths[k] == '' or k == 'output':
            continue
        p = os.path.dirname(dict_paths[k]) if '*' in dict_paths[k] else dict_paths[k]
        assert os.path.exists(p), 'missing (%s) "%s"' % (k, p)
    return dict_paths, args


def export_visual(df_row, path_out, relabel=True):
    """ given visualisation of segmented image and annotation

    :param {str: ...} df_row:
    :param str path_out: path to the visualisation directory
    :param bool relabel: whether relabel segmentation as sequential
    """
    annot, _ = tl_io.load_image_2d(df_row['path_1'])
    segm, _ = tl_io.load_image_2d(df_row['path_2'])
    img = None
    if 'path_3' in df_row:
        img, _ = tl_io.load_image_2d(df_row['path_3'])
    if relabel:
        annot = relabel_sequential(annot)[0]
        segm = seg_lbs.relabel_max_overlap_unique(annot, segm)
    fig = seg_visu.figure_overlap_annot_segm_image(annot, segm, img)
    name = os.path.splitext(os.path.basename(df_row['path_1']))[0]
    logging.debug('>> exporting -> %s', name)
    fig.savefig(os.path.join(path_out, '%s.png' % name))


def wrapper_relabel_segm(annot_segm):
    annot, segm = annot_segm
    try:
        segm = seg_lbs.relabel_max_overlap_unique(annot, segm)
    except Exception:
        logging.error(traceback.format_exc())
    return segm


def main(dict_paths, nb_jobs=NB_THREADS, relabel=True):
    """ main evaluation

    :param {str: str} dict_paths:
    :param int nb_jobs: number of thred running in parallel
    :param bool relabel: whether relabel segmentation as sequential
    """
    logging.info('running...')
    if not os.path.isdir(dict_paths['output']):
        assert os.path.isdir(os.path.dirname(dict_paths['output'])), \
            'missing %s' % dict_paths['output']
        os.mkdir(dict_paths['output'])

    name = os.path.basename(os.path.dirname(dict_paths['segm']))
    list_dirs = [dict_paths['annot'], dict_paths['segm']]
    if dict_paths['image'] != '':
        list_dirs.append(dict_paths['image'])
    df_paths = tl_io.find_files_match_names_across_dirs(list_dirs)
    path_csv = os.path.join(dict_paths['output'], NAME_CVS_PER_IMAGE % name)
    df_paths.to_csv(path_csv)

    annots, _ = tl_io.load_images_list(df_paths['path_1'].values.tolist())
    segms, names = tl_io.load_images_list(df_paths['path_2'].values.tolist())
    logging.info('loaded %i annots and %i segms', len(annots), len(segms))

    if relabel:
        mproc_pool = mproc.Pool(nb_jobs)
        annots = [relabel_sequential(annot)[0] for annot in annots]
        segms = mproc_pool.map(wrapper_relabel_segm, zip(annots, segms))
        mproc_pool.close()
        mproc_pool.join()

    path_csv = os.path.join(dict_paths['output'], NAME_CVS_PER_IMAGE % name)
    logging.debug('export to "%s"', path_csv)
    df_stat = seg_clf.compute_stat_per_image(segms, annots, names, nb_jobs)
    df_stat.to_csv(path_csv)

    path_csv = os.path.join(dict_paths['output'], NAME_CVS_OVERALL % name)
    logging.debug('export to "%s"', path_csv)
    df_desc = df_stat.describe()
    logging.info(df_desc.T[['count', 'mean', 'std']])
    df_desc.to_csv(path_csv)

    path_visu = os.path.join(dict_paths['output'], '%s__visual' % name)
    if not os.path.isdir(path_visu):
        os.mkdir(path_visu)
    # for idx, row in df_paths.iterrows():
    #     export_visual(row, path_visu)
    tqdm_bar = tqdm.tqdm(total=len(df_paths))
    wrapper_visual = partial(export_visual, path_out=path_visu)
    mproc_pool = mproc.Pool(nb_jobs)
    for _ in mproc_pool.imap_unordered(wrapper_visual,
                                       (row for idx, row in df_paths.iterrows())):
        tqdm_bar.update()
    mproc_pool.close()
    mproc_pool.join()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dict_paths, args = aparse_params(PATHS)
    main(dict_paths)
