"""
Framework for handling input/output

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import re
import glob
import logging
import warnings

import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
from PIL import Image
# import libtiff, nibabel
from scipy import ndimage
from skimage import exposure, io, color, measure
import nibabel

import segmentation.utils.read_zvi as read_zvi

COLUMNS_COORDS = ['X', 'Y']
DEFAULT_PATTERN_SET_LIST_FILE = '*.txt'


def update_path(path_file, lim_depth=5, absolute=True):
    """ bubble in the folder tree up intil it found desired file 
    otherwise return original one
    
    :param str path_file: path to the input file / folder
    :param int lim_depth: maximal depth for going up
    :return str: path to output file / folder

    >>> path = 'sample_file.test'
    >>> f = open(path, 'w')
    >>> update_path(path, absolute=False)
    'sample_file.test'
    >>> os.remove(path)
    """
    if path_file.startswith('/'):
        return path_file
    elif path_file.startswith('~'):
        path_file = os.path.expanduser(path_file)
    else:
        for _ in range(lim_depth):
            if os.path.exists(path_file): break
            path_file = os.path.join('..', path_file)
    if absolute:
        path_file = os.path.abspath(path_file)
    return path_file


def swap_coord_x_y(points):
    """ swap X and Y coordinates in vector of possitions

    :param [[int, int]] points:
    :return [[int, int]]:

    >>> swap_coord_x_y(np.array([[1, 2], [2, 4], [5, 6]]))
    [[2, 1], [4, 2], [6, 5]]
    """
    points = np.array(points)
    if len(points) == 0:
        return points.tolist()
    assert points.shape[1] == 2
    points_new = points[:, [1, 0]]
    return points_new.tolist()


def load_landmarks_txt(path_file):
    """ load the landmarks from a given file of TXT type and return array
    
    :param str path_file: name of the input file(whole path)
    :return ndarray: array of landmarks of size <nbLandmarks> x 2

    >>> lnds = np.array([[1, 2], [2, 4], [5, 6]])
    >>> fp = save_landmarks_txt('./sample_landmarks.test', lnds)
    >>> fp
    './sample_landmarks.txt'
    >>> lnds_new = load_landmarks_txt(fp)
    >>> np.array_equal(lnds, lnds_new)
    True
    >>> os.remove(fp)
    """
    path_file = os.path.abspath(os.path.expanduser(path_file))
    assert os.path.exists(path_file), 'missing "%s"' % path_file
    # load input file
    with open(path_file, 'r') as f:
        lines = f.readlines()
        
    landmarks = list()
    for line in lines[2:]:
        # logging.debug(line)
        match_obj = re.match('(.*) (.*)', line)
        vals = match_obj.groups()
        vals = [int(float(i)) for i in vals]
        # logging.debug(' load_landmarks_txt: ' + repr(vals))
        landmarks.append(vals)
    logging.debug(' load_landmarks_txt (%i): \n%s',
                  len(landmarks), repr(landmarks))
    return landmarks


def load_landmarks_csv(path_file):
    """ load the landmarks from a given file of TXT type and return array
    
    :param str path_file: name of the input file(whole path)
    :return ndarray: array of landmarks of size <nbLandmarks> x 2

    >>> lnds = np.array([[1, 2], [2, 4], [5, 6]])
    >>> fp = save_landmarks_csv('./sample_landmarks.test', lnds)
    >>> fp
    './sample_landmarks.csv'
    >>> lnds_new = load_landmarks_csv(fp)
    >>> np.array_equal(lnds, lnds_new)
    True
    >>> os.remove(fp)
    """
    path_file = os.path.abspath(os.path.expanduser(path_file))
    assert os.path.exists(path_file), 'missing "%s"' % path_file
    df = pd.DataFrame.from_csv(path_file)
    landmarks = df[COLUMNS_COORDS].as_matrix().tolist()
    logging.debug(' load_landmarks_csv (%i): \n%s', len(landmarks),
                  repr(np.asarray(landmarks).astype(int).tolist()))
    return landmarks


# def load_landmarks_elastix(path_file):
#     """ load the landmarks from a given file of TXT type and return array
#
#     :param: path_file: str, name of the input file(whole path)
#     :return: array of landmarks of size <nbLandmarks> x 2
#     """
#     path_file = os.path.abspath(os.path.expanduser(path_file))
#     assert os.path.exists(path_file), 'missing "%s"' % path_file
#     # load input file
#     with open(path_file, "r") as f:
#         lines = f.readlines()
#
#     landmarks = list()
#     # parse values
#     for line in lines:
#         match_obj = re.match('(.*) OutputPoint = \[(.*)(.*) \]\t;(.*)' , line)
#         elems = match_obj.groups()
#         logging.debug(' load_landmarks_elastix: -> {} {}',
#                       repr(elems[1]), repr(elems[2]))
#         lnd = [float(elems[1]), float(elems[2])]
#         landmarks.append(lnd)
#     return landmarks


def save_landmarks_txt(path_file, landmarks):
    """ save the landmarks into a given file of TXT type
    
    :param str path_file: name of the input file(whole path)
    :param landmarks: array of landmarks of size nb_landmarks x 2
    :return str: path to output file
    """
    assert os.path.exists(os.path.dirname(path_file)), \
        'missing "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.txt'
    logging.info(' save_landmarks_txt: -> creating TXT file: %s', path_file)
    # create the results file in TXT
    with open(path_file, 'w') as f:
        f.write('point\n')
        f.write('%i\n' % len(landmarks))
        for el in landmarks:
            f.write("{} {}\n".format(int(el[0]), int(el[1])))
    return path_file


def save_landmarks_csv(path_file, landmarks, dtype=float):
    """ save the landmarks into a given file of CSV type
    
    :param str path_file: fName is name of the input file(whole path)
    :param [[int, int ]] landmarks: array of landmarks of size nb_landmarks x 2
    :return str: path to output file
    """
    assert os.path.exists(os.path.dirname(path_file)), \
        'missing "%s"' % os.path.dirname(path_file)
    path_file = os.path.splitext(path_file)[0] + '.csv'
    logging.debug(' save_landmarks_csv: -> creating CSV file: %s' % path_file)
    # create the results file in CSV
    if len(landmarks) == 0:
        logging.warning('empty set of landmarks')
        landmarks = np.zeros((0, 2))
    df = pd.DataFrame(np.array(landmarks, dtype=dtype), columns=COLUMNS_COORDS)
    df.to_csv(path_file)
    return path_file


def scale_image_vals_in_range(img, im_range=1.):
    """ scale image values in given range

    :param ndarray img: input image
    :param im_range: range to scale image values (1. or 255)
    :return ndarray:

    >>> np.random.seed(0)
    >>> img = np.random.randint(10, 255, (25, 30))
    >>> im = scale_image_vals_in_range(img)
    >>> im.min()
    0.0
    >>> im.max()
    1.0
    """
    img = (img - np.min(img)) / float(np.max(img) - np.min(img))
    if im_range == 255:
        img = (img * im_range).astype(np.uint8)
    return img


def scale_image_intensity(img, im_range=1., quantiles=(2, 98)):
    """ scale image values with in give quntile range to filter some outlaiers

    :param ndarray img: input image
    :param im_range: range to scale image values (1. or 255)
    :param (int, int) quantiles: scale image values in certain quantile range
    :return ndarray:

    >>> np.random.seed(0)
    >>> img = np.random.randint(10, 255, (25, 30))
    >>> im = scale_image_intensity(img)
    >>> im.min()
    0.0
    >>> im.max()
    1.0
    """
    p_low = np.percentile(img, quantiles[0])
    p_high = np.percentile(img, quantiles[1])
    img = exposure.rescale_intensity(img.astype(float),
                                     in_range=(p_low, p_high),
                                     out_range='float')
    if im_range == 255:
        img = (img * im_range).astype(np.uint8)
    return img


# def convert_tiff_2_ndarray(im, im_range=255):
#     img = np.empty(im.shape)
#     for i in range(img.shape[0]):
#         img[i, :, :] = np.array(im[i])
#     img = scale_image_intensity(img, im_range)
#     return img


def load_image_2d(path_img):
    """ loading any supported image type

    :param str path_img: path to the input image
    :return str, ndarray: image name, image as matrix

    PNG image
    >>> img_name = 'testing_image'
    >>> img = np.random.randint(0, 255, size=(20, 20, 3))
    >>> path_img = export_image(os.path.join('.', img_name), img,
    ...                         stretch_range=False)
    >>> path_img
    './testing_image.png'
    >>> os.path.exists(path_img)
    True
    >>> img_new, _ = load_image_2d(path_img)
    >>> np.array_equal(img, img_new)
    True
    >>> io.imsave(path_img, np.random.random((50, 65, 4)))
    >>> img_new, _ = load_image_2d(path_img)
    >>> img_new.shape
    (50, 65, 3)
    >>> Image.fromarray(np.random.randint(0, 2, (65, 50)), mode='1').save(path_img)
    >>> img_new, _ = load_image_2d(path_img)
    >>> img_new.shape
    (65, 50)
    >>> os.remove(path_img)

    TIFF image
    >>> img_name = 'testing_image'
    >>> img = np.random.randint(0, 255, size=(5, 20, 20))
    >>> path_img = export_image(os.path.join('.', img_name), img,
    ...                         stretch_range=False)
    >>> path_img
    './testing_image.tiff'
    >>> os.path.exists(path_img)
    True
    >>> img_new, _ = load_image_2d(path_img)
    >>> img_new.shape
    (5, 20, 20)
    >>> np.array_equal(img, img_new)
    True
    >>> os.remove(path_img)
    """
    assert os.path.exists(path_img), path_img
    n_img, img_ext = os.path.splitext(os.path.basename(path_img))

    if img_ext in ['.tif', '.tiff']:
        img = io.imread(path_img)
        # DEPRECATED
        # im = libtiff.TiffFile().get_tiff_array()
        # img = np.empty(im.shape)
        # for i in range(img.shape[0]):
        #     img[i, :, :] = im[i]
        # img = np.array(img.tolist())
    else:
        # img = io.imread(path_img)
        im = Image.open(path_img)
        if im.mode == '1':
            im = im.convert('L')
        img = np.asarray(im)
        # in case of png and alpha channel cut it out...
        if img.ndim == 3 and img.shape[-1] > 3:
            img = img[:, :, :3]
    # if bool_val and img.max() > 0:
    #     img = (img / float(img.max()))
    return img, n_img


def export_image(path_img, img, stretch_range=True):
    """ export an image with given path and optional pattern for image name

    :param str path_img: path to the output image
    :param ndarray img: image np.array<height, width>
    :param bool stretch_range:
    :return str: path to the image

    Image - PNG
    >>> np.random.seed(0)
    >>> img = np.random.random([5, 10])
    >>> path_img = export_image(os.path.join('.', 'testing-image'), img)
    >>> path_img
    './testing-image.png'
    >>> os.path.exists(path_img)
    True
    >>> im, name = load_image_2d(path_img)
    >>> im.shape
    (5, 10)
    >>> im
    array([[143, 186, 157, 141, 110, 168, 114, 232, 251,  99],
           [206, 137, 148, 241,  18,  22,   5, 216, 202, 226],
           [255, 208, 120, 203,  30, 166,  37, 246, 135, 108],
           [ 68, 201, 118, 148,   4, 160, 159, 160, 245, 177],
           [ 93, 113, 181,  15, 173, 174,  54,  33,  82,  94]], dtype=uint8)
    >>> os.remove(path_img)

    Image - TIFF
    >>> img = np.random.random([5, 20, 25])
    >>> path_img = export_image(os.path.join('.', 'testing-image'), img)
    >>> path_img
    './testing-image.tiff'
    >>> os.path.exists(path_img)
    True
    >>> im, name = load_image_2d(path_img)
    >>> im.shape
    (5, 20, 25)
    >>> os.remove(path_img)
    """
    assert img.ndim >= 2, 'wrong image dim: %s' % repr(img.shape)
    if not os.path.isdir(os.path.dirname(path_img)):
        return ''
    logging.debug(' .. saving image %s with %s to "%s"', repr(img.shape),
                  repr(np.unique(img)), path_img)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3):
        if stretch_range and img.max() > 0:
            img = img / float(img.max()) * 255
        # io.imsave(path_img, im_norm)
        path_img = os.path.splitext(path_img)[0] + '.png'
        Image.fromarray(img.astype(np.uint8)).save(path_img)
    elif img.ndim == 3:
        if stretch_range and img.max() > 0:
            img = img / float(img.max()) * 255 ** 2
        path_img = os.path.splitext(path_img)[0] + '.tiff'
        io.imsave(path_img, img)
        # tif = libtiff.TIFF.open(path_img, mode='w')
        # tif.write_image(img_clip.astype(np.uint16))
    else:
        logging.warning('not supported image format: %s', repr(img.shape))
    return path_img


def load_params_from_txt(path_file):
    """ parse the parameter file which was coded by repr function
    
    :param str path_file: path to file with parameters
    :return {str: ...}:

    >>> p = {'abc': 123}
    >>> path_file = './sample_config.txt'
    >>> with open(path_file, 'w') as fp:
    ...     lines = ['"{}" : {},'.format(k, p[k]) for k in p]
    ...     _= fp.write(os.linesep.join(lines))
    >>> p2 = load_params_from_txt(path_file)
    >>> p2
    {'abc': '123'}
    >>> os.remove(path_file)
    """
    assert os.path.isfile(path_file), 'missing %s' % path_file
    with open(path_file, "r") as f:
        lines = f.readlines()

    # parse all lines and add then into a dictionary
    params = {}
    for line in lines:
        match_obj = re.match(r'"(.*)"\s*:\s* (.*),\s*', line)
        # logging.debug(' CLUSTER_PARAMS reading line: {}'.format(line))
        if match_obj is not None:
            key, val = match_obj.groups()
            val = val.replace('\t', '')
            logging.debug(' load_params_from_file: {} -> {}'.format(key, val))
            params[key] = val
    return params


def convert_img_2_nifti_gray(path_img_in, path_out):
    """ converting standard image to Nifti format

    :param str path_img_in: path to input image
    :param str path_out: path to output directory
    :return str: path to output image

    >>> np.random.seed(0)
    >>> img = np.random.random((150, 125))
    >>> p_in = './test_sample_image.png'
    >>> io.imsave(p_in, img)
    >>> p_out = convert_img_2_nifti_gray(p_in, '.')
    >>> p_out
    'test_sample_image.nii'
    >>> os.remove(p_out)
    >>> os.remove(p_in)
    """
    assert os.path.exists(path_img_in), 'missing input: %s' % path_img_in
    assert os.path.exists(path_out), 'missing output: %s' % path_out
    name_img_out = os.path.splitext(os.path.basename(path_img_in))[0] + '.nii'
    path_img_out = os.path.join(os.path.dirname(path_out), name_img_out)
    logging.debug('Convert image to Nifti format "%s" ->  "%s"',
                  path_img_in, path_img_out)

    # img = Image.open(imgIn).convert('LA')
    img = io.imread(path_img_in)
    img = color.rgb2gray(img)

    img = np.swapaxes(img, 1, 0)
    nim = nibabel.Nifti1Pair(img, np.eye(4))
    nibabel.save(nim, path_img_out)

    # for k in nim.header.keys():
    #     print('{:20s}: \t{}'.format(k, nim.header[k]))
    return path_img_out


def convert_img_2_nifti_rgb(path_img_in, path_out):
    """ converting standard image to Nifti format

    :param str path_img_in: path to input image
    :param str path_out: path to output directory
    :return str: path to output image

    >>> np.random.seed(0)
    >>> p_in = './test_sample_image.png'
    >>> io.imsave(p_in, np.random.random((150, 125, 3)))
    >>> p_nifty = convert_img_2_nifti_rgb(p_in, '.')
    >>> p_nifty
    'test_sample_image.nii'
    >>> os.remove(p_nifty)
    >>> os.remove(p_in)
    """
    assert os.path.exists(path_img_in), 'missing input: %s' % path_img_in
    assert os.path.exists(path_out), 'missing output: %s' % path_out
    name_img_out = os.path.splitext(os.path.basename(path_img_in))[0] + '.nii'
    path_img_out = os.path.join(os.path.dirname(path_out), name_img_out)
    logging.debug('Convert image to Nifti format "%s" ->  "%s"',
                  path_img_in, path_img_out)

    # img = Image.open(pImgIn)
    img = io.imread(path_img_in)
    dims = img.shape

    img = img.reshape([dims[0], dims[1], 1, dims[2], 1])
    img = np.swapaxes(np.swapaxes(img, 0, 3), 1, 4)

    nim = nibabel.Nifti1Pair(img, np.eye(4))
    nibabel.save(nim, path_img_out)

    # for k in nim.header.keys():
    #     print('{:20s}: \t{}'.format(k, nim.header[k]))
    return path_img_out


def convert_nifti_2_img(path_img_in, path_img_out):
    """ given input and output path convert from nifti to image

    :param str path_img_in: path to input image
    :param str path_img_out: path to output image
    :return str: path to output image

    >>> np.random.seed(0)
    >>> p_in = './test_sample_image.png'
    >>> io.imsave(p_in, np.random.random((150, 125, 3)))
    >>> p_nifty = convert_img_2_nifti_rgb(p_in, '.')
    >>> p_nifty
    'test_sample_image.nii'
    >>> p_img = convert_nifti_2_img(p_nifty, './test_sample_image.jpg')
    >>> p_img
    './test_sample_image.jpg'
    >>> os.remove(p_nifty)
    >>> os.remove(p_img)
    >>> os.remove(p_in)
    """
    assert os.path.exists(path_img_in), 'missing input: %s' % path_img_in
    assert os.path.exists(os.path.dirname(path_img_out)), \
        'missing output: %s' % os.path.dirname(path_img_out)

    nim = nibabel.load(path_img_in)

    if len(nim.get_data().shape) > 2: # colour
        img = np.swapaxes(np.swapaxes(nim.get_data(), 0, 3), 1, 4)
        dims = img.shape
        img = img.reshape([dims[0], dims[1], dims[3]])
    else:  # gray
        # img = nim.get_data()
        img = np.swapaxes(nim.get_data(), 1, 0)

    if img.max() > 1:
        img = img / 255.

    io.imsave(path_img_out, img)
    return path_img_out


# def convertTransform_mhd2txt(matlab, libs, transMHD, txtOut, fLog=None):
#     logging.debug('Convert MHD transform to txt format "{}"'.format(transMHD))
#     fileMHD = os.path.splitext(os.path.basename(transMHD))[0]
#     pathMHD = os.path.dirname(transMHD)
#     # matlab -nodisplay -nosplash -nodesktop -r "addpath('../scripts/Matlab');
#            [D P]=mhd_read('results.mhd'); im=repmat(double(D)./255, [1 1 3]);
#            imwrite(im, 'results.jpg'); exit;"
#     cmdMatlab = """{} -nodisplay -nosplash -nodesktop -r \"addpath('{}'); \
#                 [D P]=mhd_read('{}.mhd', '{}'); \
#                 dlmwrite('{}', D); \
#                 exit;\" """.format(matlab, libs, fileMHD, pathMHD, txtOut)
#     toolsTest.runCommandLine(cmdMatlab, fLog)


def load_image_tiff_volume(path_img, im_range=None):
    """ loading TIFF image

    :param str path_img: path to the input image
    :param float im_range: range to scale image values (1. or 255)
    :return ndarray:

    >>> p_img = os.path.join(update_path('images'), 'drosophila_ovary_3D',
    ...                      'AU10-13_f0011.tif')
    >>> img = load_image_tiff_volume(p_img)
    >>> img.shape
    (30, 323, 512)
    >>> p_img = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_slice', 'image', 'insitu7545.tif')
    >>> img = load_image_tiff_volume(p_img)
    >>> img.shape
    (647, 1024, 3)
    """
    path_img = update_path(path_img)
    assert os.path.exists(path_img), 'given image "%s" not exist!' % path_img

    img = io.imread(path_img)

    # import libtiff
    # tif = libtiff.TIFF.open(path_img, mode='r')
    # img = []
    # # to read all images in a TIFF file:
    # for im in tif.iter_images():
    #     img.append(im)
    # tif.close()
    # img = np.array(img)

    # special case of loading 2d tiff
    if img.ndim == 4:
        if img.shape[1] == 3:
            img = img[:, 0, ...]
        else:
            img = img[..., 0]
        # rotate for RGB image
        if img.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)

    logging.debug('image %s values (%d - %d)',
                  repr(img.shape), img.min(), img.max())
    if im_range is not None:
        img = scale_image_intensity(img, im_range)
    return img


def load_tiff_volume_split_double_band(path_img, im_range=None):
    """ load TIFF volume  assuming that there are two bands in zip style:
    c1, c2, c1, c2, c1, ...
    and split each odd index belong to one of two bands

    :param str path_img: path to the input image
    :param float im_range: range to scale image values (1. or 255)
    :return ndarray, ndarray:

    >>> p_img = os.path.join(update_path('images'), 'drosophila_ovary_3D',
    ...                      'AU10-13_f0011.tif')
    >>> img_b1, img_b2 = load_tiff_volume_split_double_band(p_img)
    >>> img_b1.shape
    (15, 323, 512)
    >>> img_b2.shape
    (15, 323, 512)
    >>> p_img = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_slice', 'image', 'insitu7545.tif')
    >>> img_b1, img_b2 = load_tiff_volume_split_double_band(p_img)
    >>> img_b1.shape
    (1, 647, 1024)
    >>> img_b2.shape
    (1, 647, 1024)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img = load_image_tiff_volume(path_img, im_range)

    if img.shape[2] == 3:
        img_b1 = img[np.newaxis, ..., 0]
        img_b2 = img[np.newaxis, ..., 1]
    elif img.shape[0] == 3:
        img_b1 = img[np.newaxis, 0, ...]
        img_b2 = img[np.newaxis, 1, ...]
    else:  # true volume
        img_b1 = np.array(img[0::2])
        img_b2 = np.array(img[1::2])
        if len(img_b2) == 0:
            # loading also 2d images with rgb bands
            assert img_b1.ndim == 4, 'image is not RGB'
            img_b2 = np.array([img_b1[0, :, :, 1]])
            img_b1 = np.array([img_b1[0, :, :, 0]])
    assert img_b1.shape[0] == img_b2.shape[0], \
        'not equal slice number for %s and %s' \
        % (repr(img_b1.shape), repr(img_b2.shape))
    return img_b1, img_b2


def load_zvi_volume_double_band_split(path_img):
    """ loading zvi image and split by bands

    :param str path_img: path to the image
    :return ndarray, ndarray:

    >>> p_img = os.path.join(update_path('images'),
    ...                      'others', 'sample.zvi')
    >>> img_b1, img_b2 = load_zvi_volume_double_band_split(p_img)
    >>> img_b1.shape
    (2, 488, 648)
    """
    assert os.path.isfile(path_img), 'missing: %s' % path_img
    img = read_zvi.load_image(path_img)
    nb_half = img.shape[0] / 2
    img_b1 = img[:int(nb_half)]
    img_b2 = img[int(nb_half):]
    return img_b1, img_b2


def load_img_double_band_split(path_img, im_range=1., quantiles=(2, 98)):
    """ load image and split channels

    :param str path_img: path to the image
    :param float im_range: range to scale image values (1. or 255)
    :param (int, int) quantiles: scale image values in certain quantile range
    :return:

    >>> p_imgs = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_slice', 'image')
    >>> p_img = os.path.join(p_imgs, 'insitu7545.jpg')
    >>> img_b1, img_b2 = load_img_double_band_split(p_img)
    >>> img_b1.shape
    (647, 1024)
    >>> p_img = os.path.join(p_imgs, 'insitu7545.tif')
    >>> img_b1, img_b2 = load_img_double_band_split(p_img)
    >>> img_b1.shape
    (647, 1024)
    >>> p_img = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_3D', 'AU10-13_f0011.tif')
    >>> img_b1, img_b2 = load_img_double_band_split(p_img)
    >>> img_b1.shape
    (15, 323, 512)
    """
    assert os.path.isfile(path_img), 'missing: %s' % path_img
    file_posix = os.path.splitext(os.path.basename(path_img))[1]
    if file_posix == '.zvi':
        img_b1, img_b2 = load_zvi_volume_double_band_split(path_img)
    elif file_posix in ['.tif', '.tiff']:
        img_b1, img_b2 = load_tiff_volume_split_double_band(path_img)
    else:  # assuming PNG
        img = np.array(Image.open(path_img))
        img_b1 = img[..., 0]
        img_b2 = img[..., 1]
    # in case thete is just single clice work with it as 2D image
    img_b1 = img_b1[0, ...] if img_b1.shape[0] == 1 else img_b1
    img_b2 = img_b2[0, ...] if img_b2.shape[0] == 1 else img_b2
    # scale values
    if im_range is not None:
        img_b1 = scale_image_intensity(img_b1, im_range, quantiles)
        img_b2 = scale_image_intensity(img_b2, im_range, quantiles)
    return img_b1, img_b2


def scale_image_size(path_img, size, path_out=None):
    """ load image - scale image - export image on the same path

    :param str path_img: path to the image
    :param [int, int] size: new image size
    :param str path_out: path to output image, if none overwrite the input
    :return str: path to output image

    >>> np.random.seed(0)
    >>> path_in = './test_sample_image.png'
    >>> io.imsave(path_in, np.random.random((150, 125, 3)))
    >>> path_out = scale_image_size(path_in, [75, 50])
    >>> Image.open(path_out).size
    (75, 50)
    >>> os.remove(path_out)
    """
    if not path_out:
        path_out = path_img
    logging.debug('Image scaling %s -> %s"', path_img, path_out)
    img = Image.open(path_img)
    img = img.resize(size, Image.ANTIALIAS)
    img.save(path_out)
    return path_out


def load_complete_image_folder(path_dir, img_name_pattern='*.png',
                               nb_sample=None, im_range=255, skip=()):
    """ load complete image folder with specific name pattern

    :param str path_dir: loading dictionary
    :param str img_name_pattern: image name pattern
    :param int nb_sample: load just some subset of images
    :param im_range: range to scale image values (1. or 255)
    :param [str] skip: skip some prticular images by name
    :return:

    >>> p_imgs = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_slice', 'image')
    >>> l_imgs, l_names = load_complete_image_folder(p_imgs, '*.jpg')
    >>> len(l_imgs)
    5
    >>> l_names
    ['insitu4174', 'insitu4358', 'insitu7331', 'insitu7544', 'insitu7545']
    """
    path_imgs = glob.glob(os.path.join(path_dir, img_name_pattern))
    for s in skip:
        path_imgs = [p for p in path_imgs if s not in os.path.basename(p)]
    path_imgs = sorted(path_imgs)[:nb_sample]
    logging.debug('found following images (%i): %s', len(path_imgs), path_imgs)
    return load_images_list(path_imgs, im_range)


def load_images_list(path_imgs, im_range=255):
    """ load list of images together with image names

    :param [str] path_imgs: paths to input images
    :param im_range: range to scale image values (1. or 255)
    :return [ndarray], [str]:

    >>> np.random.seed(0)
    >>> path_in = './test_sample_image.png'
    >>> io.imsave(path_in, np.random.random((150, 125, 3)))
    >>> l_imgs, l_names = load_images_list([path_in, './test_sample.img'])
    >>> l_names
    ['test_sample_image']
    >>> [img.shape for img in l_imgs]
    [(150, 125, 3)]
    >>> [img.dtype for img in l_imgs]
    [dtype('uint8')]
    >>> os.remove(path_in)
    >>> path_in = './test_sample_image.tif'
    >>> io.imsave(path_in, np.random.random((150, 125, 3)))
    >>> l_imgs, l_names = load_images_list([path_in, './test_sample.img'])
    >>> l_names
    ['test_sample_image']
    >>> os.remove(path_in)
    """
    list_images, list_names = [], []
    for i, path_im in enumerate(path_imgs):
        path_im = os.path.abspath(os.path.expanduser(path_im))
        if path_im is None or not os.path.exists(path_im):
            logging.debug('particular image is missing "%s"', path_im)
            continue
        logging.debug('loading image "{}"'.format(path_im))
        if 'tif' in os.path.splitext(path_im)[1]:
            img = load_image_tiff_volume(path_im, im_range)
            im = img[..., 0]
        else:
            im, _ = load_image_2d(path_im)
        # logging.debug('image dims: {}'.format(im.shape))
        list_images.append(im)
        list_names.append(os.path.splitext(os.path.basename(path_im))[0])
    return list_images, list_names


# def load_list_names(path_list):
#     assert os.path.exists(path_list), '%s' % path_list
#     df = pd.DataFrame.from_csv(path_list, index_col=False, header=None)
#     assert len(df.columns) == 1  # assume just single column
#     list_names = df.as_matrix()[:, 0].tolist()
#     return list_names


def merge_image_channels(img_ch1, img_ch2, img_ch3=None):
    """ merge 2 or 3 image channels into single image

    :param ndarray img_ch1: image channel
    :param ndarray img_ch2: image channel
    :param ndarray img_ch3: image channel
    :return ndarray:

    >>> np.random.seed(0)
    >>> merge_image_channels(np.random.random((150, 125)),
    ...                      np.random.random((150, 125))).shape
    (150, 125, 3)
    >>> merge_image_channels(np.random.random((150, 125)),
    ...                      np.random.random((150, 125)),
    ...                      np.random.random((150, 125))).shape
    (150, 125, 3)
    """
    assert img_ch1.ndim == 2, \
        'image as to strictly 2D and single channel, got %s' \
        % repr(img_ch1.shape)
    assert img_ch1.shape == img_ch2.shape, \
        'channel dimension has to match: %s vs %s' \
        % (repr(img_ch1.shape), repr(img_ch2.shape))
    if img_ch3 is None:
        img_ch3 = np.zeros(img_ch1.shape)
    else:
        assert img_ch1.shape == img_ch3.shape, \
            'channel dimension has to match: %s vs %s' \
            % (repr(img_ch1.shape), repr(img_ch3.shape))
    img_rgb = np.rollaxis(np.array([img_ch1, img_ch2, img_ch3]), 0, 3)
    return img_rgb


def find_files_match_names_across_dirs(list_path_pattern, drop_none=True):
    """ walk over dir with images and segmentation and pair those with the same
    name and if the folder with centers exists also add to each par a center
    NOTE: returns just paths

    :param [str] list_path_pattern: list of paths with image name patterns
    :param bool drop_none: drop if there are some none - missing values in rows
    :return: DF<path_1, path_2, ...>

    >>> def mpath(d, n):
    ...     p = os.path.join(update_path('images'),
    ...                      'drosophila_ovary_slice', d, n)
    ...     return p
    >>> df = find_files_match_names_across_dirs([mpath('image', '*.jpg'),
    ...                                          mpath('segm', '*.png'),
    ...                                          mpath('center_levels', '*.csv')])
    >>> len(df) > 0
    True
    >>> df.columns.tolist()
    ['path_1', 'path_2', 'path_3']
    >>> df = find_files_match_names_across_dirs([mpath('image', '*.png'),
    ...                                          mpath('segm', '*.jpg'),
    ...                                          mpath('center_levels', '*.csv')])
    >>> len(df)
    0
    """
    list_path_pattern = [pp for pp in list_path_pattern if pp is not None]
    assert len(list_path_pattern) > 1, 'at least 2 paths required'
    for p in list_path_pattern:
        assert os.path.exists(os.path.dirname(p)), \
            'missing "%s"' % os.path.dirname(p)

    def get_name(path, pattern='*'):
        name = os.path.splitext(os.path.basename(path))[0]
        for s in pattern.split('*'):
            name = name.replace(s, '')
        return name

    def get_paths_names(path_pattern):
        paths_ = glob.glob(path_pattern)
        if len(paths_) == 0:
            return [None], [None]
        names_ = [get_name(p, os.path.basename(path_pattern)) for p in paths_]
        return paths_, names_

    logging.info('find match files...')
    paths_0, names_0 = get_paths_names(list_path_pattern[0])
    list_paths = [paths_0]

    for path_pattern_n in list_path_pattern[1:]:
        paths_n = [None] * len(paths_0)
        name_pattern = os.path.basename(path_pattern_n)
        list_files = glob.glob(path_pattern_n)
        logging.debug('found %i files in %s', len(list_files), path_pattern_n)
        for i, path_n in enumerate(list_files):
            name_n = get_name(path_n, name_pattern)
            if name_n in names_0:
                idx = names_0.index(name_n)
                paths_n[idx] = path_n
        list_paths.append(paths_n)

    col_names = ['path_%i' % (i + 1) for i in range(len(list_paths))]
    df_paths = pd.DataFrame(list(zip(*list_paths)), columns=col_names)

    # filter None
    if drop_none:
        df_paths.dropna(inplace=True)
    return df_paths


def get_background_color(image):
    """ extract background color as median along image boundaries

    :param image:
    :return:

    >>> img = np.zeros((5, 15), dtype=int)
    >>> img[:4, 3:9] = 1
    >>> img
    array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> get_background_color(img)
    0
    >>> get_background_color(np.ones((5, 15, 3), dtype=int))
    array([1, 1, 1])
    >>> get_background_color(np.ones((5, 15, 3, 1), dtype=int))
    array(0)
    """
    if image.ndim == 2:
        bg_pixels = np.hstack([image[0, :], image[:, 0],
                               image[-1, :], image[:, -1]])
        bg_color = np.argmax(np.bincount(bg_pixels))
    elif image.ndim == 3:
        bg_pixels = np.vstack([image[0, :, ...], image[:, 0, ...],
                               image[-1, :, ...], image[:, -1, ...]])
        bg_color = np.median(bg_pixels, axis=0)
    else:
        logging.error('not supported image dim: %s' % repr(image.shape))
        bg_color = np.array(0)
    bg_color = bg_color.astype(image.dtype)
    return bg_color


def add_padding(img_size, padding, min_row, min_col, max_row, max_col):
    """ add some padding but still be inside image

    :param (int, int) img_size:
    :param int padding: set padding around segmented object
    :param int min_row: setting top left corner of bounding box
    :param int min_col: setting top left corner of bounding box
    :param int max_row: setting bottom right corner of bounding box
    :param int max_col: setting bottom right corner of bounding box
    :return: int, int, int, int

    >>> add_padding((50, 50), 5, 15, 25, 35, 55)
    (10, 20, 40, 50)
    """
    min_row = max(0, min_row - padding)
    min_col = max(0, min_col - padding)
    max_row = min(img_size[0], max_row + padding)
    max_col = min(img_size[1], max_col + padding)
    return min_row, min_col, max_row, max_col


def cut_object(img, mask, padding, use_mask=False, bg_color=None):
    """ cut an object fro image according binary object segmentation

    :param ndarray img:
    :param ndarray mask:
    :param int padding: set padding around segmented object
    :param use_mask: fill BG values also outside the mask
    :param bg_color: set as default values outside bounding box
    :return:

    >>> img = np.ones((10, 20), dtype=int)
    >>> img[3:7, 4:16] = 2
    >>> mask = np.zeros((10, 20), dtype=int)
    >>> mask[4:6, 5:15] = 1
    >>> cut_object(img, mask, 2)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    >>> cut_object(img, mask, 2, use_mask=True)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
           [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    """
    assert mask.shape[:2] == img.shape[:2]

    prop = measure.regionprops(mask.astype(int))[0]
    bg_pixels = np.hstack([mask[0, :], mask[:, 0], mask[-1, :], mask[:, -1]])
    bg_mask = np.argmax(np.bincount(bg_pixels))

    if bg_color is None:
        bg_color = get_background_color(img)

    rotate = np.rad2deg(prop.orientation)
    shift = prop.centroid - (np.array(mask.shape) / 2.)
    shift = np.append(shift, np.zeros(img.ndim - mask.ndim))

    mask = ndimage.interpolation.shift(mask, -shift[:mask.ndim], order=0)
    mask = ndimage.rotate(mask, -rotate, order=0, mode='constant',
                          cval=np.nan)

    img_cut = ndimage.interpolation.shift(img, -shift[:img.ndim], order=0)
    img_cut = ndimage.rotate(img_cut, -rotate, order=0, mode='constant',
                             cval=np.nan)
    img_cut[np.isnan(mask), ...] = bg_color
    mask[np.isnan(mask)] = bg_mask

    prop = measure.regionprops(mask.astype(int))[0]
    min_row, min_col, max_row, max_col = add_padding(img_cut.shape, padding,
                                                     *prop.bbox)
    img_cut = img_cut[min_row:max_row, min_col:max_col, ...]

    if use_mask:
        use_mask = mask[min_row:max_row, min_col:max_col, ...].astype(bool)
        img_cut[~use_mask, ...] = bg_color

    return img_cut
