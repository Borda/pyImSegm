"""
The clustering is already part of the center prediction scipt.
The path to the image and segmentation serves just for visualisation,
for the own clustering they are not needed.

Copyright (C) 2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import json
import multiprocessing as mproc
from functools import partial

import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import cluster

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pylab as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import segmentation.utils.data_io as tl_io
import segmentation.utils.experiments as tl_expt
import segmentation.utils.drawing as tl_visu
import run_center_candidate_training as run_train
# import run_center_prediction as run_pred

# Set experiment folders
FOLDER_CENTER = 'centers'
FOLDER_CLUSTER_VISUAL = 'centers_clustering'
LIST_SUBDIRS = [FOLDER_CENTER, FOLDER_CLUSTER_VISUAL]

IMAGE_EXTENSIONS = ['.png', '.jpg']
# subfigure size for visualisations
MAX_FIGURE_SIZE = 12
FOLDER_EXPERIMENT = 'detect-centers-predict_%s'

# The asumtion is that the max distance is about 3 * sampling distance
CLUSTER_PARAMS = {
    'DBSCAN_max_dist': 50,
    'DBSCAN_min_samples': 1,
}
PARAMS = run_train.CENTER_PARAMS
PARAMS.update(CLUSTER_PARAMS)
PARAMS.update({
    'path_expt': os.path.join(PARAMS['path_output'],
                              FOLDER_EXPERIMENT % PARAMS['name']),
    'path_images': os.path.join(run_train.PATH_IMAGES, 'image', '*.jpg'),
    'path_segms': os.path.join(run_train.PATH_IMAGES, 'segm', '*.png'),
    'path_centers': os.path.join(PARAMS['path_output'],
                                 FOLDER_EXPERIMENT % PARAMS['name'],
                                 'candidates', '*.csv')
})


def cluster_center_candidates(points, max_dist=100, min_samples=1):
    """ cluster center candidates by given desity clustering

    :param [[float]] points:
    :param float max_dist:
    :param int min_samples:
    :return: ndarray, [int]
    """
    points = np.array(points)
    if len(points) == 0:
        return points, []
    dbscan = cluster.DBSCAN(eps=max_dist, min_samples=min_samples)
    dbscan.fit(points)
    labels = dbscan.labels_.copy()

    centers = []
    for i in range(max(labels) + 1):
        clust = points[labels == i]
        if len(clust) > 0:
            center = np.mean(clust, axis=0)
            centers.append(center)

    return np.array(centers), labels


def export_draw_image_centers_clusters(path_out, name, img, centres, points=None,
                                       clust_labels=None, segm=None, fig_posix='',
                                       max_fig_size=MAX_FIGURE_SIZE):
    """ draw visualisation of clustered center candidates and export it

    :param str path_out:
    :param str name:
    :param ndarray img:
    :param centres:
    :param [[float]] points:
    :param [int] clust_labels:
    :param ndarray segm:
    :param str fig_posix:
    :param int max_fig_size:
    """
    # if the output dos nor exist, leave
    if not os.path.isdir(path_out): return

    size = None
    if img is not None:
        size = np.array(img.shape[:2][::-1], dtype=float)
    elif segm is not None:
        size = np.array(segm.shape[:2][::-1], dtype=float)

    if size is not None:
        fig_size = (size / size.max() * max_fig_size)
    else:
        fig_size = (max_fig_size, max_fig_size)

    fig, ax = plt.subplots(figsize=fig_size)
    if img.ndim == 3:
        img = img[:, :, 0]
    tl_visu.draw_image_clusters_centers(ax, img, centres, points, clust_labels, segm)

    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(path_out, name + fig_posix + '.png'))
    plt.close(fig)


def cluster_points_draw_export(dict_row, params, path_out=None):
    """ cluster points into centers and export visualisations

    :param {} dict_row:
    :param {str: ...} params:
    :param str path_out:
    :return:
    """
    assert all(n in dict_row for n in ['path_points', 'path_image', 'path_segm']), \
        'missing some required fields: %s' % repr(dict_row)
    name = os.path.splitext(os.path.basename(dict_row['path_points']))[0]
    points = tl_io.load_landmarks_csv(dict_row['path_points'])
    if len(points) == 0:
        logging.debug('no points to cluster for "%s"', name)
    points = tl_io.swap_coord_x_y(points)

    centres, clust_labels = cluster_center_candidates(points,
                                      max_dist=params['DBSCAN_max_dist'],
                                      min_samples=params['DBSCAN_min_samples'])
    path_csv = os.path.join(path_out, FOLDER_CENTER, name + '.csv')
    tl_io.save_landmarks_csv(path_csv, tl_io.swap_coord_x_y(centres))

    path_visu = os.path.join(path_out, FOLDER_CLUSTER_VISUAL)

    img, segm = None, None
    if dict_row['path_image'] is not None and os.path.isfile(dict_row['path_image']):
        img = np.array(Image.open(dict_row['path_image']))
    if dict_row['path_segm'] is not None and os.path.isfile(dict_row['path_segm']):
        segm = np.array(Image.open(dict_row['path_segm']))

    export_draw_image_centers_clusters(path_visu, name, img, centres,
                                       points, clust_labels, segm)
    dict_row.update({'image': name,
                     'path_centers': path_csv,
                     'nb_centres': len(centres)})
    return dict_row


# def load_centers_images_segm(path_pattern_csv, path_images, path_segms):
#     list_csv = sorted(glob.glob(path_pattern_csv))
#     logging.info('found %i csv files', len(list_csv))
#     # filter only csv files win specific format
#     # list_csv = [p for p in list_csv
#     #                 if re.match(PATTERN_NAME_CSV_CENTERS, os.path.basename(p)) is not None]
#     # logging.info('filtered to %i center files', len(list_csv))
#
#     def add_img_path(name, key, path_dir):
#         for im_ext in IMAGE_EXTENSIONS:
#             path_img = os.path.join(path_dir, name + im_ext)
#             if os.path.exists(path_img):
#                 d[key] = path_img
#                 break
#             else:
#                 d[key] = None
#
#     df_paths = pd.DataFrame()
#     for path_csv in list_csv:
#         d = {'path_points': path_csv}
#         name = os.path.splitext(os.path.basename(path_csv))[0]
#         add_img_path(name, 'path_image', os.path.dirname(path_images))
#         add_img_path(name, 'path_segm', os.path.dirname(path_segms))
#         df_paths = df_paths.append(d, ignore_index=True)
#     return df_paths


def main(params):
    """ PIPELINE candidate clustering

    :param {str: any} params:
    """
    logging.info('running...')

    with open(os.path.join(params['path_expt'],
                           'config_clustering.json'), 'w') as fp:
        json.dump(params, fp)

    tl_expt.create_subfolders(params['path_expt'], LIST_SUBDIRS)

    list_paths = [params[k] for k in ['path_images', 'path_segms', 'path_centers']]
    df_paths = tl_io.find_files_match_names_across_dirs(list_paths)
    df_paths.columns = ['path_image', 'path_segm', 'path_points']
    df_paths.index = range(1, len(df_paths) + 1)
    path_cover = os.path.join(params['path_expt'], run_train.NAME_CSV_TRIPLES)
    df_paths.to_csv(path_cover)

    logging.info('run clustering...')
    df_paths_new = pd.DataFrame()
    tqdm_bar = tqdm.tqdm(total=len(df_paths))
    if params['nb_jobs'] > 1:
        wrapper_clustering = partial(cluster_points_draw_export, params=params,
                                     path_out=params['path_expt'])
        pool = mproc.Pool(params['nb_jobs'])
        for dict_center in pool.imap_unordered(wrapper_clustering,
                                   (dict(row) for idx, row in df_paths.iterrows())):
            df_paths_new = df_paths_new.append(dict_center, ignore_index=True)
            tqdm_bar.update()
        pool.close()
        pool.join()
    else:
        for dict_row in (dict(row) for idx, row in df_paths.iterrows()):
            dict_center = cluster_points_draw_export(dict_row, params,
                                                     params['path_expt'])
            df_paths_new = df_paths_new.append(dict_center, ignore_index=True)
            tqdm_bar.update()

    df_paths_new.set_index('image', inplace=True)
    df_paths_new.to_csv(path_cover)

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = run_train.arg_parse_params(PARAMS)
    main(params)
