"""
Simple GUI tool for annotation and correction center candidates
Whether path is set, it load the whole table with ovary info and for each image
while loading estimate egg mask (annotation for anterior, posterior and latitude)
and automatically edit False Positive

The navigation  over images is using arrows (forward - right, backward - left)
Contour represents the reconstructed ellipse for annotated eggs from table
Explanation for colour of potential centers:
 * yellow, the original center point assumed by algo.
 * orange, mark fo false positive (clock on yellow point)
 * white, mark for false negative (click to free space)

Clicking on the image plane by left button change label or ad new point
and click by right middle remove actual point and right button changes the state
change to not changed

>> sudo apt-get install python-gtk2-dev
>> python gui_annot_center_correction.py \
    -imgs "~/Medical-drosophila/ovary_all_slices/png/*.png" \
    -csv "~/Medical-drosophila/TEMPORARY_OVARY/detect_ovary_centers_detect/insitu*.csv" \
    -info ~/Medical-drosophila/ovary_image_info_for_prague_short.csv

Copyright (C) 2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import ndimage, spatial
# from planar import line as pl_line
import matplotlib
matplotlib.use('GTKAgg')  # or >> matplotlib.rcsetup.all_backends


# http://matplotlib.org/users/navigation_toolbar.html
import gtk
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import imsegm.utilities.data_io as tl_data
import imsegm.utilities.drawing as tl_visu

PATH_BASE = tl_data.update_path(os.path.join('data_images', 'drosophila_ovary_slice'))
PATH_IMAGES = os.path.join(PATH_BASE, 'image', '*.jpg')
PATH_CSV = os.path.join(PATH_BASE, 'center_levels', '*.csv')
NAME_INFO_SHORT = 'ovary_image_info.csv'

POSIX_CSV_LABEL = '_labeled'
COLUMNS_POSITION = ['ant_x', 'ant_y', 'post_x', 'post_y', 'lat_x', 'lat_y']

DICT_LIMIT_CORRECT = 10
DICT_LIMIT_REMOVE = 30
COLOR_FALSE_POSITIVE = '#FF5733'
COLOR_FALSE_NEGATIVE = 'w'

POINT_MARKERS = [
    {'change': 0, 'label': 1, 'marker': 'o', 'color': 'y'},
    {'change': 0, 'label': 0, 'marker': 'x', 'color': 'y'},
    {'change': 1, 'label': 1, 'marker': 'o', 'color': COLOR_FALSE_NEGATIVE},
    {'change': 1, 'label': 0, 'marker': 'o', 'color': COLOR_FALSE_POSITIVE},
]

df_center_labeled, fig = None, None

# TODO: add - swapping group of points not only one by one


def arg_parse_params():
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', '--path_images', type=str, required=False,
                        help='path to dir and image pattern', default=PATH_IMAGES)
    parser.add_argument('-csv', '--path_csv', type=str, required=False,
                        help='path to the CSV directory', default=PATH_CSV)
    parser.add_argument('-info', '--path_info', type=str, required=False,
                        help='path to file with complete info', default=None)
    params = vars(parser.parse_args())
    for k in (k for k in params if 'path' in k):
        if params[k] is None:
            continue
        params[k] = os.path.abspath(os.path.expanduser(params[k]))
        p = os.path.dirname(params[k]) if '*' in params[k] else params[k]
        assert os.path.exists(p), 'missing: %s' % p
    logging.info('ARG PARAMETERS: \n %s', repr(params))
    return params


def load_paths_image_csv(params, skip_csv=POSIX_CSV_LABEL):
    """ loading content of two folder and specific pattern to obtain list
    of images and csv with centers, then it find the intersection between them
    according their unique names

    :param {str: str} params:
    :param str skip_csv: pattern in csv name that skips the file
    :return [(str, str)]:
    """
    logging.debug('loading pairs for %s and %s', params['path_csv'],
                  params['path_images'])
    get_name = lambda p: os.path.splitext(os.path.basename(p))[0]
    list_csv = glob.glob(params['path_csv'])
    list_names = [get_name(p) for p in list_csv]
    # skip al names that contains given posix
    list_names = [n for n in list_names if skip_csv not in n]
    # filter to have just paths with the  right names
    list_imgs = sorted([p for p in glob.glob(params['path_images'])
                        if get_name(p) in list_names])
    # update list of names
    list_names = [get_name(p) for p in list_imgs]
    # filter to have just paths with the  right names
    list_csv = sorted([p for p in list_csv if get_name(p) in list_names])
    assert len(list_imgs) == len(list_csv), \
        'the number of images (%i) and csv (%i) has to be same' % \
        (len(list_imgs), len(list_csv))
    list_join_img_csv = zip(list_imgs, list_csv)
    assert all(get_name(p1) == get_name(p2) for p1, p2 in list_join_img_csv), \
        'names has to be same for %s' % repr(list_join_img_csv)
    return list_join_img_csv


def set_false_positive(df_points, mask_eggs):
    for idx, row in df_points.iterrows():
        pos = row[['X', 'Y']].values
        label = mask_eggs[pos[1], pos[0]]
        if label == 0:
            df_points.xs(idx)['label'] = 0
            df_points.xs(idx)['change'] = 1
    return df_points


def set_false_negative(df_points, mask_eggs):
    points = df_points[['X', 'Y']].as_matrix().astype(int)
    for lb in (lb for lb in np.unique(mask_eggs) if lb != 0):
        mask = (mask_eggs == lb)
        labels = mask[points[:, 1], points[:, 0]]
        if sum(labels) == 0:
            pos = ndimage.measurements.center_of_mass(mask)
            df_points = df_points.append(
                {'X': pos[1], 'Y': pos[0], 'label': 1, 'change': 1},
                ignore_index=True)
    return df_points


def load_csv_center_label(path_csv, mask_eggs=None):
    """ load already edited ces with point, wheter it not exists create it

    :param str path_csv:
    :return DF: DF<x, y, label, change>
    """
    path_csv_labeled = path_csv.replace('.csv', POSIX_CSV_LABEL + '.csv')
    if os.path.isfile(path_csv_labeled):
        df_points = pd.read_csv(path_csv_labeled, index_col=0)
    else:
        df_points = pd.read_csv(path_csv, index_col=0)
        df_points['label'] = np.ones((len(df_points), ))
        df_points['change'] = np.zeros((len(df_points), ))

    # some automatic correction according info file
    if mask_eggs is not None:
        # df_points = set_false_positive(df_points, mask_eggs)
        df_points = set_false_negative(df_points, mask_eggs)

    # df_labeled.to_csv(path_csv_labeled)
    return df_points


def export_corrections():
    """ export corrected centers """
    global df_center_labeled, paths_img_csv, actual_idx
    _, path_csv = paths_img_csv[actual_idx]
    path_csv_labeled = path_csv.replace('.csv', POSIX_CSV_LABEL + '.csv')
    df_center_labeled.to_csv(path_csv_labeled)


def estimate_eggs_from_info(path_img):
    """ finds all eggs for particular slice and mask them by ellipse annotated
    by ant, post and lat in the all info table

    :param str path_img:
    :return ndarray:
    """
    global df_info_all, img, fig
    if df_info_all is None:
        return None
    name_img = os.path.basename(path_img).replace('.png', '.tif')
    mask = (df_info_all['image_path'] == name_img)
    name_stack = df_info_all[mask]['stack_path'].values[0]
    df_stack = df_info_all[df_info_all['stack_path'] == name_stack]

    dict_slice = {col: df_stack[col].values.tolist() for col in COLUMNS_POSITION}
    pos_ant = np.array(zip(dict_slice['ant_x'], dict_slice['ant_y']))
    pos_lat = np.array(zip(dict_slice['lat_x'], dict_slice['lat_y']))
    pos_post = np.array(zip(dict_slice['post_x'], dict_slice['post_y']))

    mask_eggs = tl_visu.draw_eggs_ellipse(img.shape[:2], pos_ant, pos_lat, pos_post)

    return mask_eggs


def canvas_load_image_centers():
    """ load image nad csv with centers and update canvas """
    global paths_img_csv, actual_idx, df_center_labeled, img, mask_eggs
    path_img, path_csv = paths_img_csv[actual_idx]
    logging.info('loading image (%i/%i): "%s"', actual_idx + 1, len(paths_img_csv),
                 os.path.splitext(os.path.basename(path_img))[0])

    img = plt.imread(path_img)
    mask_eggs = estimate_eggs_from_info(path_img)
    df_center_labeled = load_csv_center_label(path_csv, mask_eggs)

    canvas_update_image_centers()


def canvas_update_image_centers(marker_schema=POINT_MARKERS):
    """ according corredted points and loaded image update canvas """
    global fig, df_center_labeled, img, mask_eggs

    fig.clf()
    fig.gca().imshow(img)
    if mask_eggs is not None:
        fig.gca().contour(mask_eggs, colors='c', linestyles='dotted')

    for dict_marker in marker_schema:
        filter_label = (df_center_labeled['change'] == dict_marker['change'])
        filter_change = (df_center_labeled['label'] == dict_marker['label'])
        df_points = df_center_labeled[filter_label & filter_change]
        fig.gca().plot(df_points['X'].tolist(), df_points['Y'].tolist(),
                       dict_marker['marker'], color=dict_marker['color'])

    fig.gca().set_xlim([0, img.shape[1]])
    fig.gca().set_ylim([img.shape[0], 0])
    fig.gca().axes.get_xaxis().set_ticklabels([])
    fig.gca().axes.get_yaxis().set_ticklabels([])

    fig.canvas.draw()


def add_point_correction(x, y, changing=1, limit_dist=DICT_LIMIT_CORRECT):
    """ take list of all points and estimate the nearest, whether it is closer
    then a threshold correct this point othervise add new one

    :param float x:
    :param float y:
    :param int changing:
    :param int limit_dist:
    """
    global df_center_labeled
    points = df_center_labeled[['X', 'Y']].as_matrix()
    dists = spatial.distance.cdist(np.array(points), np.array([[x, y]]),
                                   metric='euclidean')
    if np.min(dists) < limit_dist:
        # corect a point
        idx = np.argmin(dists)
        label = df_center_labeled.xs(idx)['label']
        df_center_labeled.xs(idx)['label'] = int(not label)
        df_center_labeled.xs(idx)['change'] = changing
    else:
        # add new point
        df_center_labeled = df_center_labeled.append(
            {'X': x, 'Y': y, 'label': 1, 'change': 1}, ignore_index=True)

    canvas_update_image_centers()


def remove_point(x, y, limit_dist=DICT_LIMIT_REMOVE):
    """ take list of all points and estimate the nearest, whether it is closer
    then a threshold remove it and reindex list

    :param float x:
    :param float y:
    :param int limit_dist:
    """
    global df_center_labeled
    points = df_center_labeled[['X', 'Y']].as_matrix()
    dists = spatial.distance.cdist(np.array(points), np.array([[x, y]]),
                                   metric='euclidean')
    if np.min(dists) < limit_dist:
        idx = np.argmin(dists)
        df_center_labeled.drop(idx, inplace=True)
        df_center_labeled.reset_index(drop=True, inplace=True)

    canvas_update_image_centers()


def onclick(event):
    """ register event on click left and right button

    :param event:
    """
    if event.xdata is None or event.ydata is None:
        logging.warning('click out of image bounds')
        return
    logging.debug('button=%d, xdata=%f, ydata=%f', event.button,
                  event.xdata, event.ydata)
    if event.button == 1:  # left click
        add_point_correction(event.xdata, event.ydata)
    elif event.button == 3:  # left click
        add_point_correction(event.xdata, event.ydata, changing=0)
    elif event.button == 2:  # right click
        remove_point(event.xdata, event.ydata)
    # fig.gca().plot(event.xdata, event.ydata, '.')
    # fig.canvas.draw()


def onkey_release(widget, event, data=None):
    """ register key press for arrows to move back and forward

    :param widget:
    :param event:
    :param data:
    """
    global paths_img_csv, actual_idx
    key = gtk.gdk.keyval_name(event.keyval)
    logging.debug('pressed key: %s', key)

    if key == 'Right':
        export_corrections()
        actual_idx += 1
    elif key == 'Left':
        export_corrections()
        actual_idx -= 1

    if actual_idx < 0:
        logging.info('you are on the beginning')
        actual_idx = 0
    elif actual_idx == len(paths_img_csv):
        logging.info('you reach the End')
        actual_idx = len(paths_img_csv) - 1

    canvas_load_image_centers()


def main(params):
    global fig, paths_img_csv, actual_idx, df_info_all
    win = gtk.Window()
    win.set_default_size(600, 400)
    win.set_title('Annotation (correction) egg centers')
    win.connect('destroy', lambda x: gtk.main_quit())

    fig = Figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    canvas = FigureCanvasGTKAgg(fig)  # a gtk.DrawingArea
    win.add(canvas)

    actual_idx = 0
    paths_img_csv = load_paths_image_csv(params)
    logging.info('loaded %i pairs (image & centers)', len(paths_img_csv))
    assert paths_img_csv, 'missing paths image - csv'

    if params['path_info'] is not None and os.path.isfile(params['path_info']):
        df_info_all = pd.read_csv(params['path_info'], sep='\t', index_col=0)
    else:
        df_info_all = None
    logging.info('loaded complete info')

    canvas_load_image_centers()

    fig.canvas.mpl_connect('button_press_event', onclick)
    win.connect('key-release-event', onkey_release)

    win.show_all()
    gtk.main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running GUI...')
    params = arg_parse_params()
    main(params)
    logging.info('DONE')
