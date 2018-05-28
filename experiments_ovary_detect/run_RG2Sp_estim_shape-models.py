"""
Create the Ray shape model fro Region Growing segmentation

SAMPLE run:
>> python run_RG2Sp_estim-shape-models.py \
    -annot "~/Medical-drosophila/mask_2d_slice_complete_ind_egg/*.png" \
    -out data -nb 15

Copyright (C) 2016-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import pickle
import argparse

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root
import imsegm.utils.data_io as tl_data
import imsegm.region_growing as tl_rg

PATH_DATA = tl_data.update_path('data', absolute=True)
PATH_IMAGES = os.path.join(tl_data.update_path('data_images'), 'drosophila_ovary_slice')
PATH_ANNOT = os.path.join(PATH_IMAGES, 'annot_eggs', '*.png')
RAY_STEP = 10
# names of default files for models
NAME_CSV_RAY_ALL = 'egg_ray_shapes.csv'
NAME_PKL_MODEL_SINGLE = 'RG2SP_single-model.pkl'
NAME_PKL_MODEL_MIXTURE = 'RG2SP_mixture-model.pkl'
NAME_NPZ_MODEL_SINGLE = 'RG2SP_single-model.npz'
NAME_NPZ_MODEL_MIXTURE = 'RG2SP_mixture-model.npz'


def arg_parse_params():
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: str}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-annot', '--path_annot', type=str, required=False,
                        help='path to directory & name pattern for annotations',
                        default=PATH_ANNOT)
    parser.add_argument('-out', '--path_out', type=str, required=False,
                        help='path to the output directory', default=PATH_DATA)
    parser.add_argument('-nb', '--nb_comp', type=int, required=False,
                        help='number of component in Mixture model', default=2)
    params = vars(parser.parse_args())
    for k in (k for k in params if 'path' in k):
        params[k] = tl_data.update_path(params[k], absolute=True)
        p = os.path.dirname(params[k]) if k == 'path_annot' else params[k]
        assert os.path.exists(p), 'missing: %s' % p
    # load saved configuration
    logging.info('ARG PARAMETERS: \n %s', repr(params))
    return params


def main(path_annot, path_out, nb_comp=5):
    list_paths = sorted(glob.glob(path_annot))
    print ('nb images:', len(list_paths), 'SAMPLES:',
           [os.path.basename(p) for p in list_paths[:5]])
    list_segms = []
    for path_seg in list_paths:
        seg = tl_data.io_imread(path_seg)
        list_segms.append(seg)

    list_rays, _ = tl_rg.compute_object_shapes(list_segms, ray_step=RAY_STEP,
                                               interp_order='spline',
                                               smooth_coef=1)
    logging.info('nb eggs: %i, nb rays: %i', len(list_rays), len(list_rays[0]))

    x_axis = np.linspace(0, 360, len(list_rays[0]), endpoint=False)
    df = pd.DataFrame(np.array(list_rays), columns=x_axis.astype(int))
    path_csv = os.path.join(path_out, NAME_CSV_RAY_ALL)
    logging.info('exporting all Rays: %s', path_csv)
    df.to_csv(path_csv)

    # SINGLE MODEL
    model, list_cdf = tl_rg.transform_rays_model_cdf_mixture(list_rays, 1)
    cdf = np.array(np.array(list_cdf))

    # path_model = os.path.join(path_out, NAME_NPZ_MODEL_SINGLE)
    # logging.info('exporting model: %s', path_model)
    # np.savez(path_model, name='cdf', cdfs=cdf, mix_model=model)
    path_model = os.path.join(path_out, NAME_PKL_MODEL_SINGLE)
    logging.info('exporting model: %s', path_model)
    with open(path_model, 'wb') as fp:
        pickle.dump({'name': 'cdf',
                     'cdfs': cdf,
                     'mix_model': model}, fp)

    # MIXTURE MODEL
    model, list_mean_cdf = tl_rg.transform_rays_model_sets_mean_cdf_mixture(
        list_rays, nb_comp)

    # path_model = os.path.join(path_out, NAME_NPZ_MODEL_MIXTURE)
    # logging.info('exporting model: %s', path_model)
    # np.savez(path_model, name='set_cdfs', cdfs=list_mean_cdf,
    #                     mix_model=model)
    path_model = os.path.join(path_out, NAME_PKL_MODEL_MIXTURE)
    logging.info('exporting model: %s', path_model)
    with open(path_model, 'wb') as fp:
        pickle.dump({'name': 'set_cdfs',
                     'cdfs': list_mean_cdf,
                     'mix_model': model}, fp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(params['path_annot'], params['path_out'], params['nb_comp'])

    logging.info('Done')
