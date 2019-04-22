"""

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange


def __cinit__():
    pass


# def getColourMeanImg2d_simple(img, seg):
#     # slow while pthon function is called inside cython code, such as "[:]"
#     nbSegments = np.max(seg) +1
#     features = np.zeros([nbSegments, 3])
#     count = np.zeros([nbSegments, 1])
#     h, w  = seg.shape
#     for x in range(w):
#         for y in range(h):
#             count[seg[x,y], 0] += 1
#             features[seg[x,y],:] += img[x, y,:]
#     features = (features / count)
#     return features


# def getColourMeanImg2dRGB(np.ndarray[np.int16_t, ndim=3] img,
#                           np.ndarray[np.int16_t, ndim=2] seg):
#     cdef:
#         int nbSegments = np.max(seg) +1
#         np.ndarray[np.float64_t, ndim=2] features = np.zeros([nbSegments, 3],
#                                                              dtype=np.float64)
#         np.ndarray[np.int32_t, ndim=2] count = np.zeros([nbSegments, 1],
#                                                         dtype=np.int32)
#         int w = seg.shape[0]
#         int h = seg.shape[1]
#         int x, y, i, idx
#
#     for x in range(w):
#         for y in range(h):
#             idx = seg[x,y]
#             count[idx, 0] += 1
#             features[idx, 0] += img[x, y, 0]
#             features[idx, 1] += img[x, y, 1]
#             features[idx, 2] += img[x, y, 2]
#     # features = features / count
#     for i in range(nbSegments):
#         if count[i, 0] == 0:
#             continue
#         features[i, 0] = features[i, 0] / count[i, 0]
#         features[i, 1] = features[i, 1] / count[i, 0]
#         features[i, 2] = features[i, 2] / count[i, 0]
#     return features


def normColorFeatures(int[:, :] seg,
                      double[:, :] features):
    cdef:
        int nb_segments = np.max(seg) + 1
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int w = seg.shape[1]
        int h = seg.shape[0]
        int z, x, y, i
    for x in range(h):
        for y in range(w):
            count[seg[x, y]] += 1
    # features = features / count
    # for z in prange(3, nogil=True):
    for z in range(3):
        for i in range(nb_segments):
            if count[i] > 0:
                features[i, z] = features[i, z] / count[i]
    return features


def computeColorImage2dMean(float[:, :, :] img,
                            int[:, :] seg):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:, :] features = np.zeros([nb_segments, 3], dtype=np.float64)
        int w = seg.shape[1]
        int h = seg.shape[0]
        int z, x, y, i
    # for z in prange(3, nogil=True):
    for z in range(3):
        for x in range(h):
            for y in range(w):
                features[seg[x, y], z] += img[x, y, z]
    # features = features / count
    features = normColorFeatures(seg, features)
    return features


def computeColorImage2dEnergy(float[:, :, :] img,
                              int[:, :] seg):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:, :] features = np.zeros([nb_segments, 3], dtype=np.float64)
        float val
        int w = seg.shape[1]
        int h = seg.shape[0]
        int z, x, y, i
    for z in prange(3, nogil=True):
        for x in range(h):
            for y in range(w):
                val = img[x, y, z]
                features[seg[x, y], z] += val * val
    # features = features / count
    features = normColorFeatures(seg, features)
    return features


def computeColorImage2dVariance(float[:, :, :] img,
                                int[:, :] seg,
                                float[:, :] mean):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:, :] features = np.zeros([nb_segments, 3], dtype=np.float64)
        int w = seg.shape[1]
        int h = seg.shape[0]
        int z, x, y, i
        float v
    for z in prange(3, nogil=True):
        for x in range(h):
            for y in range(w):
                v = img[x, y, z] - mean[seg[x, y], z]
                features[seg[x, y], z] += v * v
    # features = features / count
    features = normColorFeatures(seg, features)
    return features


def computeGrayImage3dMean(float[:, :, :] img,
                           int[:, :, :] seg):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:] features = np.zeros(nb_segments, dtype=np.float64)
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int d = seg.shape[0]
        int w = seg.shape[2]
        int h = seg.shape[1]
        int z, x, y, i, idx
    for z in prange(d, nogil=True):
        for x in range(h):
            for y in range(w):
                idx = seg[z, x, y]
                count[idx] += 1
                features[idx] += img[z, x, y]
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features


def computeGrayImage3dEnergy(float[:, :, :] img,
                             int[:, :, :] seg):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:] features = np.zeros(nb_segments, dtype=np.float64)
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int d = seg.shape[0]
        int w = seg.shape[2]
        int h = seg.shape[1]
        int z, x, y, i, idx
    for z in prange(d, nogil=True):
        for x in range(h):
            for y in range(w):
                idx = seg[z, x, y]
                count[idx] += 1
                features[idx] += img[z, x, y] * img[z, x, y]
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features


def computeGrayImage3dVariance(float[:, :, :] img,
                               int[:, :, :] seg,
                               float[:] mean):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:] features = np.zeros(nb_segments, dtype=np.float64)
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int d = seg.shape[0]
        int w = seg.shape[2]
        int h = seg.shape[1]
        int z, x, y, i, idx
        float v
    for z in prange(d, nogil=True):
        for x in range(h):
            for y in range(w):
                idx = seg[z, x, y]
                count[idx] += 1
                v = img[z, x, y] - mean[idx]
                features[idx] += v * v
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features


def computeLabelHistogram2d(short[:, :] segm_select,
                            short[:, :] struc_elem,
                            int nb_labels):
    cdef:
        long[:] hist = np.zeros(nb_labels, dtype=np.int64)
        int w = segm_select.shape[1]
        int h = segm_select.shape[0]

    for x in range(h):
        for y in range(w):
            if segm_select[x, y] >= 0 and struc_elem[x, y] == 1:
                hist[segm_select[x, y]] += 1
    return hist


def computeRayFeaturesBinary2d(char[:, :] seg_binary,
                               int[:] position,
                               float angle_step,
                               int edge):
    # NOTE: for the edges: 'up' == 1 and 'down' == -1
    cdef:
        float[:] angles = np.arange(0, 360, angle_step, dtype=np.float32)
        float[:] ray_dist = np.ones(len(angles), dtype=np.float32) * -1
        int w = seg_binary.shape[1]
        int h = seg_binary.shape[0]
        char last, actual
        int i, segm_diag
        float ang, rad, grad_max, diff_x, diff_y
        float[2] pos, grad

    # in case the position is inside the border label
    if seg_binary[position[0], position[1]] and edge == 1:
        return np.zeros(len(angles), dtype=np.float32)
    segm_diag = int(np.sqrt((w * w) + (h * h)))

    # iterate over all angles in radians
    for i, rad in enumerate([np.deg2rad(ang) for ang in angles]):
        pos[0], pos[1] = position[0], position[1]
        grad = [np.sin(rad), np.cos(rad)]
        grad_max = max(abs(grad[0]), abs(grad[1]))
        grad[0] /= grad_max
        grad[1] /= grad_max
        last = seg_binary[position[0], position[1]]
        for _ in range(segm_diag):
            pos[0] += grad[0]
            pos[1] += grad[1]
            if pos[0] < 0 or round(pos[0]) >= h or pos[1] < 0 or round(pos[1]) >= w:
                break
            actual = seg_binary[int(round(pos[0])), int(round(pos[1]))]
            if (edge == 1 and actual) or (edge == -1 and last and not actual):
                diff_x = pos[0] - position[0]
                diff_y = pos[1] - position[1]
                ray_dist[i] = np.sqrt((diff_x * diff_x) + (diff_y * diff_y))
                break
            last = actual

    return ray_dist
