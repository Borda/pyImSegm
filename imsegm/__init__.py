"""
Using the try/except import since the init is called in setup  to get pkg info
before satisfying install requirements

"""

try:
    import imsegm.utilities
    imsegm.utilities
except ImportError:
    import traceback
    traceback.print_exc()

__version__ = '0.1.7'
__author__ = 'Jiri Borovec'
__author_email__ = 'jiri.borovec@fel.cvut.cz'
__license__ = 'BSD 3-clause'
__homepage__ = 'https://borda.github.io/pyImSegm'
__copyright__ = 'Copyright (c) 2014-2019, Jiri Borovec.'
__doc__ = """# Image segmentation - general superpixel segmentation & region growing

This package is aiming at (un/semi)supervised segmentation on superpixels with
 computing some basic colour and texture features. This general segmentation
 can be followed by an object centre detection and proximate ellipse fitting
 to expected object boundaries. Last included method is region growing
 with learned shape prior also running on superpixel grid.
The package contains several low-level Cython implementation to speed up some
 feature extraction methods.
Overall the project/repository contains example codes with visualisation
 in ipython notebooks and experiments required for replicating all published results.

## Superpixel segmentation with GraphCut regularisation
Image segmentation is widely used as an initial phase of many image processing
 tasks in computer vision and image analysis. Many recent segmentation methods
 use superpixels because they reduce the size of the segmentation problem
 by order of magnitude. Also, features on superpixels are much more robust
 than features on pixels only. We use spatial regularisation on superpixels
 to make segmented regions more compact. The segmentation pipeline comprises
 (i) computation of superpixels; (ii) extraction of descriptors such as colour
 and texture; (iii) soft classification, using a standard classifier for supervised
 learning, or the Gaussian Mixture Model for unsupervised learning; (iv) final
 segmentation using Graph Cut. We use this segmentation pipeline on real-world
 applications in medical imaging. We also show that unsupervised segmentation
 is sufficient for some situations, and provides similar results to those obtained
 using trained segmentation.

## Object centre detection and Ellipse approximation
An image processing pipeline to detect and localize Drosophila egg chambers that
 consists of the following steps: (i) superpixel-based image segmentation
 into relevant tissue classes (see above); (ii) detection of egg center candidates
 using label histograms and ray features; (iii) clustering of center candidates and;
 (iv) area-based maximum likelihood ellipse model fitting.

## Superpixel Region Growing with Shape prior
Region growing is a classical image segmentation method based on hierarchical
 region aggregation using local similarity rules. Our proposed approach differs
 from standard region growing in three essential aspects. First, it works
 on the level of superpixels instead of pixels, which leads to a substantial speedup.
 Second, our method uses learned statistical shape properties which encourage growing
 leading to plausible shapes. In particular, we use ray features to describe
 the object boundary. Third, our method can segment multiple objects and ensure
 that the segmentations do not overlap. The problem is represented as energy
 minimisation and is solved either greedily, or iteratively using GraphCuts.

## References
* Borovec J., Svihlik J., Kybic J., Habart D. (2017). Supervised and unsupervised
 segmentation using superpixels, model estimation, and Graph Cut.
 SPIE Journal of Electronic Imaging 26(6), 061610. DOI: 10.1117/1.JEI.26.6.061610.
* Borovec J., Kybic J., Nava R. (2017) Detection and Localization of Drosophila
 Egg Chambers in Microscopy Images. In: Wang Q., Shi Y., Suk HI., Suzuki K. (eds)
 Machine Learning in Medical Imaging. MLMI 2017. LNCS, vol 10541. Springer, Cham.
 DOI: 10.1007/978-3-319-67389-9_3.
* Borovec J., Kybic J., Sugimoto, A. (2017). Region growing using superpixels
 with learned shape prior. SPIE Journal of Electronic Imaging 26(6), 061611.
 DOI: 10.1117/1.JEI.26.6.061611.
"""
