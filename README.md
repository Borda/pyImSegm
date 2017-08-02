# Image segmentation toolbox

[![Build Status](https://travis-ci.com/Borda/pyImSegm.svg?token=HksCAm7DV2pJNEbsGJH2&branch=master)](https://travis-ci.com/Borda/pyImSegm)
[![codecov](https://codecov.io/gh/Borda/pyImSegm/branch/master/graph/badge.svg?token=BCvf6F5sFP)](https://codecov.io/gh/Borda/pyImSegm)
[![Run Status](https://api.shippable.com/projects/5962ea48a125960700c197f8/badge?branch=master)](https://app.shippable.com/github/Borda/pyImSegm)
[![Coverage Badge](https://api.shippable.com/projects/5962ea48a125960700c197f8/coverageBadge?branch=master)](https://app.shippable.com/github/Borda/pyImSegm)
[![CircleCI](https://circleci.com/gh/Borda/pyImSegm.svg?style=svg&circle-token=a30180a28ae7e490c0c0829d1549fcec9a5c59d0)](https://circleci.com/gh/Borda/pyImSegm)

## Segmentation methods

### Superpixel segmentation with GraphCut regularisation

Image  segmentation  is  widely  used  as  an  initial  phase  of  many  image  processing  tasks  in  computer vision and image analysis.  Many recent segmentation methods use superpixels, because they reduce the size of the segmentation problem by an order of magnitude.   In addition,  features on superpixels are much more robust than features  on  pixels  only.   We  use  spatial  regularization  on  superpixels  to  make  segmented  regions  more  compact. The segmentation pipeline comprises:  (i) computation of superpixels; (ii) extraction of descriptors such as color and texture;  (iii) soft classification,  using a standard classifier for supervised learning,  or the Gaussian Mixture Model for unsupervised learning; (iv) final segmentation using Graph Cut.  We use this segmentation pipeline on four real-world applications in medical imaging. We also show that unsupervised segmentation is sufficient for some situations, and provides similar results to those obtained using trained segmentation.

### Object centre detection and Ellipse approximation

We present an image processing pipeline to detect and localize Drosophila egg chambers that consists of the following steps: (i) superpixel-based image segmentation into relevant tissue classes; (ii) detection of egg center candidates using label histograms and ray features; (iii) clustering of center candidates and; (iv) area-based maximum likelihood ellipse model fitting.

### Superpixel Region Growing 

Region growing is a classical image segmentation method based on hierarchical region aggregation using local similarity rules. Our proposed method differs from classical region growing in three important aspects. First, it works on the level of superpixels instead of pixels, which leads to a substantial speedup. Second, our method uses learned statistical shape properties which encourage growing leading to plausible shapes. In particular, we use ray features to describe the object boundary. Third, our method can segment multiple objects and ensure that the segmentations do not overlap. The problem is represented as an energy minimization and is solved either greedily, or iteratively using GraphCuts.

### References

^[1] Borovec, J., Kybic, J., & Nava, R. (2017). Detection and localization of Drosophila egg chambers in microscopy images. In 8th International Workshop on Machine Learning in Medical Imaging. Quebec: Springer. Retrieved from ftp://cmp.felk.cvut.cz/pub/cmp/articles/borovec/Borovec-MLMI207.pdf

## Installation and configuration

### Configure local environment

Create your own local environment, for more see the [User Guide](https://pip.pypa.io/en/latest/user_guide.html), and install dependencies requirements.txt contains list of packages and can be installed as
```
@duda:~$ cd pyImSegm  
@duda:~/pyImSegm$ virtualenv env
@duda:~/pyImSegm$ source env/bin/activate  
(env)@duda:~/pyImSegm$ pip install -r requirements.txt  
(env)@duda:~/pyImSegm$ python ...
```
and in the end terminating...
```
(env)@duda:~/pyImSegm$ deactivate
```
