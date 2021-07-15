"""
Framework for ellipse fitting

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import numpy as np
from scipy import ndimage, spatial
from skimage import morphology
from skimage.measure import fit as sk_fit

from imsegm.descriptors import compute_ray_features_segm_2d, reconstruct_ray_features_2d, reduce_close_points
from imsegm.superpixels import make_graph_segm_connect_grid2d_conn4, segment_slic_img2d, superpixel_centers
# from skimage.measure.fit import EllipseModel  # fix in future skimage>0.13.0
from imsegm.utilities.drawing import ellipse

# INIT_MASK_BORDER = 50.
#: define minimal size of estimated ellipse
MIN_ELLIPSE_DAIM = 25.
#: define maximal Figure size in larger dimension
MAX_FIGURE_SIZE = 14
# SEGM_OVERLAP = 0.5  # define transparency for overlapping two images
#: smoothing background with morphological operation
STRUC_ELEM_BG = 15
#: smoothing foreground with morphological operation
STRUC_ELEM_FG = 5


class EllipseModelSegm(sk_fit.EllipseModel):
    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    Example
    -------
    >>> from imsegm.utilities.drawing import ellipse_perimeter
    >>> params = 20, 30, 12, 16, np.deg2rad(30)
    >>> rr, cc = ellipse_perimeter(*params)
    >>> xy = np.array([rr, cc]).T
    >>> ellipse = EllipseModelSegm()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)
    array([ 19.5 ,  29.5 ,  12.45,  16.52,   0.53])
    >>> xy = EllipseModelSegm().predict_xy(np.linspace(0, 2 * np.pi, 25), params)
    >>> ellipse = EllipseModelSegm()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)
    array([ 20.  ,  30.  ,  12.  ,  16.  ,   0.52])
    >>> np.round(abs(ellipse.residuals(xy)), 5)
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    >>> ellipse.params[2] += 2
    >>> ellipse.params[3] += 2
    >>> np.round(abs(ellipse.residuals(xy)))
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
            2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    """

    def criterion(self, points, weights, labels, table_prob=(0.1, 0.9)):
        """ Determine residuals of data to model.

        :param points: points coordinates
        :param weights: weight for each point represent the region size
        :param labels: vector of labels for each point
        :param table_prob: is a vector or foreground probabilities for each class
            and being background is supplement to 1. Another option is define
            a matrix with number of columns related to number of classes and
            the first row denote probability being foreground and second being
            background
        :return:

        Example
        -------
        >>> seg = np.zeros((10, 15), dtype=int)
        >>> r, c = np.meshgrid(range(seg.shape[1]), range(seg.shape[0]))
        >>> el = EllipseModelSegm()
        >>> el.params = [4, 7, 3, 6, np.deg2rad(10)]
        >>> weights = np.ones(seg.ravel().shape)
        >>> seg[4:5, 6:8] = 1
        >>> table_prob = [[0.1, 0.9]]
        >>> el.criterion(np.array([r.ravel(), c.ravel()]).T, weights, seg.ravel(), table_prob)  # doctest: +ELLIPSIS
        87.888...
        >>> seg[2:7, 4:11] = 1
        >>> el.criterion(np.array([r.ravel(), c.ravel()]).T, weights, seg.ravel(), table_prob)  # doctest: +ELLIPSIS
        17.577...
        >>> seg[1:9, 1:14] = 1
        >>> el.criterion(np.array([r.ravel(), c.ravel()]).T, weights, seg.ravel(), table_prob)   # doctest: +ELLIPSIS
        -70.311...
        """
        if not len(points) == len(weights) == len(labels):
            raise ValueError(
                'different sizes for points %i and weights %i and labels %i' % (len(points), len(weights), len(labels))
            )
        table_prob = np.array(table_prob)
        if table_prob.ndim == 1 or table_prob.shape[0] == 1:
            if table_prob.shape[0] == 1:
                table_prob = table_prob[0]
            table_prob = np.array([table_prob, 1. - table_prob])
        if table_prob.shape[0] != 2:
            raise ValueError('table shape %r' % table_prob.shape)
        if np.max(labels) >= table_prob.shape[1]:
            raise ValueError('labels (%i) exceed the table %r' % (np.max(labels), table_prob.shape))

        r_pos, c_pos = points[:, 0], points[:, 1]
        r_org, c_org, r_rad, c_rad, phi = self.params
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        r, c = (r_pos - r_org), (c_pos - c_org)
        dist_1 = ((r * cos_phi + c * sin_phi) / r_rad)**2
        dist_2 = ((r * sin_phi - c * cos_phi) / c_rad)**2
        inside = ((dist_1 + dist_2) <= 1)

        # import matplotlib.pyplot as plt
        # plt.imshow(labels.reshape((10, 15)), interpolation='nearest')
        # plt.contour(inside.reshape((10, 15)))

        table_q = -np.log(table_prob)
        labels_in = labels[inside].astype(int)

        diff = table_q[0, labels_in] - table_q[1, labels_in]
        residual = np.sum(weights[labels_in] * diff)

        return residual


def ransac_segm(
    points,
    model_class,
    points_all,
    weights,
    labels,
    table_prob,
    min_samples,
    residual_threshold=1,
    max_trials=100,
):
    """ Fit a model to points with the RANSAC (random sample consensus).

    Parameters
    ----------
    points : [list, tuple of] (N, D) array
        Data set to which the model is fitted, where N is the number of points
        points and D the dimensionality of the points.
        If the model class requires multiple input points arrays (e.g. source
        and destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*points)``, ``residuals(*points)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each points array as
        separate arguments.
    model_class : class
        Object with the following object methods:

         * ``success = estimate(*points)``
         * ``residuals(*points)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    points_all: list
    weights: list
    labels: list
    table_prob: list
    min_samples : int float
        The minimum number of points points to fit a model to.
    residual_threshold : float
        Maximum distance for a points point to be classified as an inlier.
    max_trials : int, optional
        Maximum number of iterations for random sample selection.


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    Examples
    --------
    >>> seg = np.zeros((120, 150), dtype=int)
    >>> ell_params = 60, 75, 40, 65, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> slic, points_all, labels = get_slic_points_labels(seg, slic_size=10, slic_regul=0.3)
    >>> points = prepare_boundary_points_ray_dist(seg, [(40, 90)], 2, sel_bg=1, sel_fg=0)[0]
    >>> table_prob = [[0.01, 0.75, 0.95, 0.9], [0.99, 0.25, 0.05, 0.1]]
    >>> weights = np.bincount(slic.ravel())
    >>> ransac_model, _ = ransac_segm(
    ...     points, EllipseModelSegm, points_all, weights, labels, table_prob, 0.6, 3, max_trials=15)
    >>> np.round(ransac_model.params[:4]).astype(int)
    array([60, 75, 41, 65])
    >>> np.round(ransac_model.params[4], 1)
    0.5
    """
    best_model = None
    best_inlier_num = 0
    best_model_fit = np.inf
    best_inliers = None

    if isinstance(min_samples, float):
        if not 0 < min_samples <= 1:
            raise ValueError("`min_samples` as ration must be in range (0, 1]")
        min_samples = int(min_samples * len(points))
    if not 0 < min_samples <= len(points):
        raise ValueError("`min_samples` must be in range (0, <nb-samples>]")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    # make sure points is ndarray and not tuple/list, so it can be modified below
    points = np.array(points)

    for _ in range(max_trials):
        # choose random sample set
        random_idxs = np.random.choice(len(points), min_samples, replace=False)
        samples = points[random_idxs]
        # for d in points:
        #     samples.append(d[random_idxs])

        # estimate model for current random sample set
        model = model_class()
        success = model.estimate(samples)

        if success is not None:  # backwards compatibility
            if not success:
                continue

        model_residuals = np.abs(model.residuals(points))
        # consensus set / inliers
        model_inliers = model_residuals < residual_threshold
        model_fit = model.criterion(points_all, weights, labels, table_prob)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(model_inliers)
        if model_fit < best_model_fit:
            best_model = model
            best_model_fit = model_fit
            if sample_inlier_num > best_inlier_num:
                best_inliers = model_inliers
                best_inlier_num = sample_inlier_num

    # estimate final model using all inliers
    if best_inliers is not None:
        points = points[best_inliers]
        best_model.estimate(points)

    return best_model, best_inliers


def get_slic_points_labels(segm, img=None, slic_size=20, slic_regul=0.1):
    """ run SLIC on image or supepixels and return superpixels, their centers
    and also lebels (label from segmentation in position of superpixel centre)

    :param ndarray segm: segmentation
    :param ndarray img: input image
    :param int slic_size: superpixel size
    :param float slic_regul: regularisation in range (0, 1)
    :return tuple:
    """
    if not img:
        img = segm / float(segm.max())
    slic = segment_slic_img2d(img, sp_size=slic_size, relative_compact=slic_regul)
    slic_centers = np.array(superpixel_centers(slic)).astype(int)
    labels = segm[slic_centers[:, 0], slic_centers[:, 1]]
    return slic, slic_centers, labels


def add_overlap_ellipse(segm, ellipse_params, label, thr_overlap=1.):
    """ add to existing image ellipse with specific label
    if the new ellipse does not ouvelap with already existing object / ellipse

    :param ndarray segm: segmentation
    :param tuple ellipse_params: parameters
    :param int label: selected label
    :param float thr_overlap: relative overlap with existing objects
    :return ndarray:

    >>> seg = np.zeros((15, 20), dtype=int)
    >>> ell_params = 7, 10, 5, 8, np.deg2rad(30)
    >>> ell = add_overlap_ellipse(seg, ell_params, 1)
    >>> ell
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> ell2_params = 4, 5, 2, 3, np.deg2rad(-30)
    >>> add_overlap_ellipse(ell, ell2_params, 2)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    if not ellipse_params:
        return segm
    mask = np.zeros(segm.shape)
    c1, c2, h, w, phi = ellipse_params
    rr, cc = ellipse(int(c1), int(c2), int(h), int(w), orientation=phi, shape=segm.shape)
    mask[rr, cc] = 1

    # filter overlapping ellipses
    for lb in range(1, int(np.max(segm) + 1)):
        overlap = np.sum(np.logical_and(segm == lb, mask == 1))
        # together = np.sum(np.logical_or(segm == lb, mask == 1))
        # ratio = float(overlap) / float(together)
        sizes = [s for s in [np.sum(segm == lb), np.sum(mask == 1)] if s > 0]
        if not sizes:
            return segm
        ratio = float(overlap) / float(min(sizes))
        # if there is already ellipse with such size, return just the segment
        if ratio > thr_overlap:
            return segm
    segm[mask == 1] = label
    return segm


def prepare_boundary_points_ray_join(
    seg,
    centers,
    close_points=5,
    min_diam=MIN_ELLIPSE_DAIM,
    sel_bg=STRUC_ELEM_BG,
    sel_fg=STRUC_ELEM_FG,
):
    """ extract some point around foreground boundaries

    :param ndarray seg: input segmentation
    :param [(int, int)] centers: list of centers
    :param float close_points: remove closest point then a given threshold
    :param int min_diam: minimal size of expected objest
    :param int sel_bg: smoothing background with morphological operation
    :param int sel_fg: smoothing foreground with morphological operation
    :return [ndarray]:

    >>> seg = np.zeros((10, 20), dtype=int)
    >>> ell_params = 5, 10, 4, 6, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> pts = prepare_boundary_points_ray_join(seg, [(4, 9)], 5., 3, sel_bg=1, sel_fg=0)
    >>> np.round(pts).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[[4.0, 16.0],
      [7.0, 10.0],
      [9.0, 5.0],
      [4.0, 16.0],
      [7.0, 10.0]]]
    """
    seg_bg, seg_fg = split_segm_background_foreground(seg, sel_bg, sel_fg)

    points_centers = []
    for center in centers:
        ray_bg = compute_ray_features_segm_2d(seg_bg, center)
        ray_bg[ray_bg < min_diam] = min_diam
        points_bg = reconstruct_ray_features_2d(center, ray_bg)
        points_bg = reduce_close_points(points_bg, close_points)

        ray_fc = compute_ray_features_segm_2d(seg_fg, center, edge='down')
        ray_fc[ray_fc < min_diam] = min_diam
        points_fc = reconstruct_ray_features_2d(center, ray_fc)
        points_fc = reduce_close_points(points_fc, close_points)

        points_both = np.vstack((points_bg, points_fc))
        points_centers.append(points_both)
    return points_centers


def split_segm_background_foreground(seg, sel_bg=STRUC_ELEM_BG, sel_fg=STRUC_ELEM_FG):
    """ smoothing segmentation with morphological operation

    :param ndarray seg: input segmentation
    :param int|float sel_bg: smoothing background with morphological operation
    :param int sel_fg: smoothing foreground with morphological operation
    :return tuple(ndarray,ndarray):

    >>> seg = np.zeros((10, 20), dtype=int)
    >>> ell_params = 5, 10, 4, 6, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> seg_bg, seg_fc = split_segm_background_foreground(seg, 1.5, 0)
    >>> seg_bg.astype(int)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    >>> seg_fc.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    seg_bg = (seg > 0)
    seg_bg = 1 - ndimage.morphology.binary_fill_holes(seg_bg)
    if sel_bg > 0:
        seg_bg = morphology.opening(seg_bg, morphology.disk(sel_bg))

    seg_fg = (seg == 1)
    if sel_fg > 0:
        seg_fg = morphology.opening(seg_fg, morphology.disk(sel_fg))
    return seg_bg, seg_fg


def prepare_boundary_points_ray_edge(
    seg,
    centers,
    close_points=5,
    min_diam=MIN_ELLIPSE_DAIM,
    sel_bg=STRUC_ELEM_BG,
    sel_fg=STRUC_ELEM_FG,
):
    """ extract some point around foreground boundaries

    :param ndarray seg: input segmentation
    :param [(int, int)] centers: list of centers
    :param float close_points: remove closest point then a given threshold
    :param int min_diam: minimal size of expected objest
    :param int sel_bg: smoothing background with morphological operation
    :param int sel_fg: smoothing foreground with morphological operation
    :return [ndarray]:

    >>> seg = np.zeros((10, 20), dtype=int)
    >>> ell_params = 5, 10, 4, 6, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> pts = prepare_boundary_points_ray_edge(seg, [(4, 9)], 2.5, 3, sel_bg=1, sel_fg=0)
    >>> np.round(pts).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[[4.0, 16.0],
      [7.0, 15.0],
      [9.0, 5.0],
      [4.0, 5.0],
      [1.0, 7.0],
      [0.0, 14.0]]]
    """
    seg_bg, seg_fc = split_segm_background_foreground(seg, sel_bg, sel_fg)

    points_centers = []
    for center in centers:
        ray_bg = compute_ray_features_segm_2d(seg_bg, center)

        ray_fc = compute_ray_features_segm_2d(seg_fc, center, edge='down')

        # replace not found (-1) by large values
        rays = np.array([ray_bg, ray_fc], dtype=float)
        rays[rays < 0] = np.inf
        rays[rays < min_diam] = min_diam
        # take the smallest from both
        ray_close = np.min(rays, axis=0)
        points_close = reconstruct_ray_features_2d(center, ray_close)
        points_close = reduce_close_points(points_close, close_points)

        points_centers.append(points_close)
    return points_centers


def prepare_boundary_points_ray_mean(
    seg,
    centers,
    close_points=5,
    min_diam=MIN_ELLIPSE_DAIM,
    sel_bg=STRUC_ELEM_BG,
    sel_fg=STRUC_ELEM_FG,
):
    """ extract some point around foreground boundaries

    :param ndarray seg: input segmentation
    :param [(int, int)] centers: list of centers
    :param float close_points: remove closest point then a given threshold
    :param int min_diam: minimal size of expected objest
    :param int sel_bg: smoothing background with morphological operation
    :param int sel_fg: smoothing foreground with morphological operation
    :return [ndarray]:

    >>> seg = np.zeros((10, 20), dtype=int)
    >>> ell_params = 5, 10, 4, 6, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> pts = prepare_boundary_points_ray_mean(seg, [(4, 9)], 2.5, 3, sel_bg=1, sel_fg=0)
    >>> np.round(pts).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[[4.0, 16.0],
      [7.0, 15.0],
      [9.0, 5.0],
      [4.0, 5.0],
      [1.0, 7.0],
      [0.0, 14.0]]]
    """
    seg_bg, seg_fc = split_segm_background_foreground(seg, sel_bg, sel_fg)

    points_centers = []
    for center in centers:
        ray_bg = compute_ray_features_segm_2d(seg_bg, center)

        ray_fc = compute_ray_features_segm_2d(seg_fc, center, edge='down')

        # replace not found (-1) by large values
        rays = np.array([ray_bg, ray_fc], dtype=float)
        rays[rays < 0] = np.inf
        rays[rays < min_diam] = min_diam

        # take the smalles from both
        ray_min = np.min(rays, axis=0)
        ray_mean = np.mean(rays, axis=0)
        ray_mean[np.isinf(ray_mean)] = ray_min[np.isinf(ray_mean)]

        points_close = reconstruct_ray_features_2d(center, ray_mean)
        points_close = reduce_close_points(points_close, close_points)

        points_centers.append(points_close)
    return points_centers


def prepare_boundary_points_ray_dist(seg, centers, close_points=1, sel_bg=STRUC_ELEM_BG, sel_fg=STRUC_ELEM_FG):
    """ extract some point around foreground boundaries

    :param ndarray seg: input segmentation
    :param [(int, int)] centers: list of centers
    :param float close_points: remove closest point then a given threshold
    :param int sel_bg: smoothing background with morphological operation
    :param int sel_fg: smoothing foreground with morphological operation
    :return [ndarray]:

    >>> seg = np.zeros((10, 20), dtype=int)
    >>> ell_params = 5, 10, 4, 6, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> pts = prepare_boundary_points_ray_dist(seg, [(4, 9)], 2, sel_bg=0, sel_fg=0)
    >>> np.round(pts, 2).tolist()  # doctest: +NORMALIZE_WHITESPACE
    [[[4.0, 16.0],
      [6.8, 15.0],
      [9.0, 5.5],
      [4.35, 5.0],
      [1.0, 6.9],
      [1.0, 9.26],
      [0.0, 11.31],
      [0.5, 14.0],
      [1.45, 16.0]]]
    """
    seg_bg, _ = split_segm_background_foreground(seg, sel_bg, sel_fg)

    points = []
    for center in centers:
        ray = compute_ray_features_segm_2d(seg_bg, center)
        points_bg = reconstruct_ray_features_2d(center, ray, 0)
        points_bg = reduce_close_points(points_bg, close_points)

        points += points_bg.tolist()
    points = np.array(points)
    # remove all very small negative valeue, probaly by rounding
    points[(points < 0) & (points > -1e-3)] = 0.

    dists = spatial.distance.cdist(points, centers, metric='euclidean')
    close_center = np.argmin(dists, axis=1)

    points_centers = []
    for i in range(close_center.max() + 1):
        pts = points[close_center == i]
        points_centers.append(pts)
    return points_centers


def filter_boundary_points(segm, slic):
    slic_centers = np.array(superpixel_centers(slic)).astype(int)
    labels = segm[slic_centers[:, 0], slic_centers[:, 1]]

    vertices, edges = make_graph_segm_connect_grid2d_conn4(slic)
    nb_labels = labels.max() + 1

    neighbour_labels = np.zeros((len(vertices), nb_labels))
    for e1, e2 in edges:
        # print e1, labels[e2], e2, labels[e1]
        neighbour_labels[e1, labels[e2]] += 1
        neighbour_labels[e2, labels[e1]] += 1
    sums = np.tile(np.sum(neighbour_labels, axis=1), (nb_labels, 1)).T
    neighbour_labels = neighbour_labels / sums

    # border point nex to foreground
    filter_bg = np.logical_and(labels == 0, neighbour_labels[:, 0] < 1)
    # fulicul cells next to background
    filter_fc = np.logical_and(labels == 1, neighbour_labels[:, 0] > 0)
    points = slic_centers[np.logical_or(filter_bg, filter_fc)]

    return points


def prepare_boundary_points_close(seg, centers, sp_size=25, relative_compact=0.3):
    """ extract some point around foreground boundaries

    :param ndarray seg: input segmentation
    :param [(int, int)] centers: list of centers
    :param int sp_size: superpixel size
    :return [ndarray]:

    >>> seg = np.zeros((100, 200), dtype=int)
    >>> ell_params = 50, 100, 40, 60, np.deg2rad(30)
    >>> seg = add_overlap_ellipse(seg, ell_params, 1)
    >>> pts = prepare_boundary_points_close(seg, [(40, 90)])
    >>> pts  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [array([[  6,  85],
            [  8, 150],
            ...
            [ 92, 118]])]
    """
    slic = segment_slic_img2d(seg / float(seg.max()), sp_size=sp_size, relative_compact=relative_compact)
    points_all = filter_boundary_points(seg, slic)

    dists = spatial.distance.cdist(points_all, centers, metric='euclidean')
    close_center = np.argmin(dists, axis=1)

    points_centers = []
    for i in range(int(close_center.max() + 1)):
        points = points_all[close_center == i]
        points_centers.append(points)
    return points_centers


# def find_dist_hist_local_minim(dists, nb_bins=25, gauss_sigma=1):
#     hist, bin = np.histogram(dists, bins=nb_bins)
#     hist = ndimage.filters.gaussian_filter1d(hist, sigma=gauss_sigma)
#     # bins = (bin[1:] + bin[:-1]) / 2.
#     # idxs = peakutils.indexes(-hist, thres=0, min_dist=1)
#     coord = feature.peak_local_max(-hist, min_distance=4).tolist() + [
#         [len(hist) - 1]]
#     thr_dist = bin[coord[0][0]]
#     return thr_dist

# def prepare_boundary_points_dist(seg, centers, sp_size=25, rltv_compact=0.3):
#     """ extract some point around foreground boundaries
#
#     :param ndarray seg: input segmentation
#     :param [(int, int)] centers: list of centers
#     :return [ndarray]:
#
#     >>> seg = np.zeros((100, 200), dtype=int)
#     >>> ell_params = 50, 100, 40, 60, 30
#     >>> seg = add_overlap_ellipse(seg, ell_params, 1)
#     >>> pts = prepare_boundary_points_dist(seg, [(40, 90)])
#     >>> sorted(np.round(pts).tolist())  # doctest: +NORMALIZE_WHITESPACE
#     [[[8, 63], [5, 79], [6, 97], [7, 117], [19, 73], [19, 85], [19, 95],
#      [19, 107], [21, 119], [24, 62], [28, 129], [33, 51], [46, 47],
#      [60, 50], [70, 60], [74, 71], [80, 81], [83, 93]]]
#     """
#     slic = seg_spx.segment_slic_img2d(seg / float(seg.max()), sp_size=sp_size,
#                                      relatv_compact=rltv_compact)
#     points_all = filter_boundary_points(seg, slic)
#
#     dists = spatial.distance.cdist(points_all, centers, metric='euclidean')
#     close_center = np.argmin(dists, axis=1)
#     dist_min = np.min(dists, axis=1)
#
#     points_centers = []
#     for i in range(int(close_center.max() + 1)):
#         dist_thr = find_dist_hist_local_minim(dist_min[close_center == i])
#         points = points_all[np.logical_and(close_center == i,
#                                        dist_min <= dist_thr)]
#         points_centers.append(points)
#     return points_centers
