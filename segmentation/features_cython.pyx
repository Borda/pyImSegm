"""

Copyright (C) 2014-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np


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
#         int x, y, i, id
#
#     for x in range(w):
#         for y in range(h):
#             id = seg[x,y]
#             count[id, 0] += 1
#             features[id, 0] += img[x, y, 0]
#             features[id, 1] += img[x, y, 1]
#             features[id, 2] += img[x, y, 2]
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
        int w = seg.shape[0]
        int h = seg.shape[1]
        int z, x, y, i
    for x in range(w):
        for y in range(h):
            count[seg[x,y]] += 1
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
        int w = seg.shape[0]
        int h = seg.shape[1]
        int z, x, y, i, id
    # for z in prange(3, nogil=True):
    for z in range(3):
        for x in range(w):
            for y in range(h):
                features[seg[x,y], z] += img[x, y, z]
    # features = features / count
    features = normColorFeatures(seg, features)
    return features


def computeColorImage2dEnergy(float[:, :, :] img,
                              int[:, :] seg):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:, :] features = np.zeros([nb_segments, 3], dtype=np.float64)
        float val
        int w = seg.shape[0]
        int h = seg.shape[1]
        int z, x, y, i, id
    for z in prange(3, nogil=True):
        for x in range(w):
            for y in range(h):
                val = img[x, y, z]
                features[seg[x,y], z] += val * val
    # features = features / count
    features = normColorFeatures(seg, features)
    return features


def computeColorImage2dVariance(float[:, :, :] img,
                                int[:, :] seg,
                                float[:, :] mean):
    cdef:
        int nb_segments = np.max(seg) + 1
        double[:, :] features = np.zeros([nb_segments, 3], dtype=np.float64)
        int w = seg.shape[0]
        int h = seg.shape[1]
        int z, x, y, i, id
        float v
    for z in prange(3, nogil=True):
        for x in range(w):
            for y in range(h):
                v = img[x, y, z] - mean[seg[x,y], z]
                features[seg[x,y], z] += v * v
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
        int w = seg.shape[1]
        int h = seg.shape[2]
        int z, x, y, i, id
    for z in prange(d, nogil=True):
        for x in range(w):
            for y in range(h):
                id = seg[z, x, y]
                count[id] += 1
                features[id] += img[z, x, y]
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features


def computeGrayImage3dEnergy(float[:, :, :] img,
                             int[:, :, :] seg):
    cdef:
        int nb_segments = np.max(seg) +1
        double[:] features = np.zeros(nb_segments, dtype=np.float64)
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int d = seg.shape[0]
        int w = seg.shape[1]
        int h = seg.shape[2]
        int z, x, y, i, id
    for z in prange(d, nogil=True):
        for x in range(w):
            for y in range(h):
                id = seg[z, x, y]
                count[id] += 1
                features[id] += img[z, x, y] * img[z, x, y]
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features


def computeGrayImage3dVariance(float[:, :, :] img,
                               int[:, :, :] seg,
                               float[:] mean):
    cdef:
        int nb_segments = np.max(seg) +1
        double[:] features = np.zeros(nb_segments, dtype=np.float64)
        int[:] count = np.zeros(nb_segments, dtype=np.int32)
        int d = seg.shape[0]
        int w = seg.shape[1]
        int h = seg.shape[2]
        int z, x, y, i, id
        float v
    for z in prange(d, nogil=True):
        for x in range(w):
            for y in range(h):
                id = seg[z,x,y]
                count[id] += 1
                v = img[z, x, y] - mean[id]
                features[id] += v * v
    for i in prange(nb_segments, nogil=True):
        if count[i] > 0:
            features[i] = features[i] / count[i]
    # features = features / count
    return features
