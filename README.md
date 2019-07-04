# Image segmentation toolbox

[![Build Status](https://travis-ci.org/Borda/pyImSegm.svg?branch=master)](https://travis-ci.org/Borda/pyImSegm)
[![codecov](https://codecov.io/gh/Borda/pyImSegm/branch/master/graph/badge.svg?token=BCvf6F5sFP)](https://codecov.io/gh/Borda/pyImSegm)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/48b7976bbe9d42bc8452f6f9e573ee70)](https://www.codacy.com/app/Borda/pyImSegm?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/pyImSegm&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/Borda/pyImSegm.svg?style=svg&circle-token=a30180a28ae7e490c0c0829d1549fcec9a5c59d0)](https://circleci.com/gh/Borda/pyImSegm)
[![Run Status](https://api.shippable.com/projects/5962ea48a125960700c197f8/badge?branch=master)](https://app.shippable.com/github/Borda/pyImSegm)
[![CodeFactor](https://www.codefactor.io/repository/github/borda/pyimsegm/badge)](https://www.codefactor.io/repository/github/borda/pyimsegm)
[![Documentation Status](https://readthedocs.org/projects/pyimsegm/badge/?version=latest)](https://pyimsegm.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/pyImSegm/community.svg)](https://gitter.im/pyImSegm/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<!--
[![Coverage Badge](https://api.shippable.com/projects/5962ea48a125960700c197f8/coverageBadge?branch=master)](https://app.shippable.com/github/Borda/pyImSegm)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Borda/pyImSegm/master?filepath=notebooks)
-->

---

## Superpixel segmentation with GraphCut regularisation

Image segmentation is widely used as an initial phase of many image processing tasks in computer vision and image analysis. Many recent segmentation methods use superpixels because they reduce the size of the segmentation problem by order of magnitude. Also,  features on superpixels are much more robust than features on pixels only. We use spatial regularisation on superpixels to make segmented regions more compact. The segmentation pipeline comprises (i) computation of superpixels; (ii) extraction of descriptors such as colour and texture; (iii) soft classification, using a standard classifier for supervised learning, or the Gaussian Mixture Model for unsupervised learning; (iv) final segmentation using Graph Cut. We use this segmentation pipeline on real-world applications in medical imaging (see a sample [images](./data_images)). We also show that [unsupervised segmentation](notebooks/segment-2d_slic-fts-clust-gc.ipynb) is sufficient for some situations, and provides similar results to those obtained using [trained segmentation](notebooks/segment-2d_slic-fts-classif-gc.ipynb).

![schema](figures/schema_slic-fts-clf-gc.jpg)

**Sample ipython notebooks:**
* [Supervised segmentation](notebooks/segment-2d_slic-fts-classif-gc.ipynb) requires training annotation
* [Unsupervised segmentation](notebooks/segment-2d_slic-fts-clust-gc.ipynb) just asks for expected number of classes
* **partially annotated images** with missing annotation is marked by a negative number

**Illustration**

![input image](figures/insitu7545.jpg)
![segmentation](figures/insitu7545_gc.png)

Reference: _Borovec J., Svihlik J., Kybic J., Habart D. (2017). **Supervised and unsupervised segmentation using superpixels, model estimation, and Graph Cut.** In: Journal of Electronic Imaging._


## Object centre detection and Ellipse approximation

An image processing pipeline to detect and localize Drosophila egg chambers that consists of the following steps: (i) superpixel-based image segmentation into relevant tissue classes (see above); (ii) detection of egg center candidates using label histograms and ray features; (iii) clustering of center candidates and; (iv) area-based maximum likelihood ellipse model fitting. See our [Poster](http://cmp.felk.cvut.cz/~borovji3/documents/poster-MLMI2017.compressed.pdf) related to this work.

**Sample ipython notebooks:**
* [Center detection](notebooks/egg-center_candidates-clustering.ipynb) consists of center candidate training and prediction, and candidate clustering.
* [Ellipse fitting](notebooks/egg-detect_ellipse-fitting.ipynb) with given estimated center structure segmentation.

**Illustration**

![estimated centres](figures/insitu7545_center_clusters.jpg)
![ellipse fitting](figures/insitu7545_ellipse_fit.png)

Reference: _Borovec J., Kybic J., Nava R. (2017) **Detection and Localization of Drosophila Egg Chambers in Microscopy Images.** In: Machine Learning in Medical Imaging._

## Superpixel Region Growing with Shape prior

Region growing is a classical image segmentation method based on hierarchical region aggregation using local similarity rules. Our proposed approach differs from standard region growing in three essential aspects. First, it works on the level of superpixels instead of pixels, which leads to a substantial speedup. Second, our method uses learned statistical shape properties which encourage growing leading to plausible shapes. In particular, we use ray features to describe the object boundary. Third, our method can segment multiple objects and ensure that the segmentations do not overlap. The problem is represented as energy minimisation and is solved either greedily, or iteratively using GraphCuts.

**Sample ipython notebooks:**
* [General GraphCut](notebooks/egg_segment_graphcut.ipynb) from given centers and initial structure segmentation.
* [Shape modeling](notebooks/RG2Sp_shape-models.ipynb) estimation from training examples.
* [Region growing](notebooks/RG2Sp_region-growing.ipynb) from given centers and initial structure segmentation with shape models.

**Illustration**

![growing RG](figures/insitu7545_RG2SP_animation.gif)
![ellipse fitting](figures/insitu7545_RG2SP-gc-mm.png)

Reference: _Borovec J., Kybic J., Sugimoto, A. (2017). **Region growing using superpixels with learned shape prior.** In: Journal of Electronic Imaging._

---

## Installation and configuration

**Configure local environment**

Create your own local environment, for more see the [User Guide](https://pip.pypa.io/en/latest/user_guide.html), and install dependencies requirements.txt contains list of packages and can be installed as
```bash
@duda:~$ cd pyImSegm  
@duda:~/pyImSegm$ virtualenv env
@duda:~/pyImSegm$ source env/bin/activate  
(env)@duda:~/pyImSegm$ pip install -r requirements.txt  
(env)@duda:~/pyImSegm$ python ...
```
and in the end terminating...
```bash
(env)@duda:~/pyImSegm$ deactivate
```

<!--
Moreover, we are using python [GraphCut wrapper](https://github.com/Borda/pyGCO) which require to be installed separately (not yet integrated in PIP)
```bash
(env)@duda:~/pyImSegm$ mkdir libs && cd libs
(env)@duda:~/pyImSegm$ git clone https://github.com/Borda/pyGCO.git
(env)@duda:~/pyImSegm$ pip install -r requirements.txt
(env)@duda:~/pyImSegm$ python setup.py install
```
-->

**Compilation**

We have implemented `cython` version of some functions, especially computing descriptors, which require to compile them before using them
```bash
python setup.py build_ext --inplace
```
If loading of compiled descriptors in `cython` fails, it is automatically swapped to use `numpy` which gives the same results, but it is significantly slower.

**Installation**

The package can be installed via pip 
```bash
pip install git+https://github.com/Borda/pyImSegm.git
```
or using `setuptools` from a local folder 
```bash
python setup.py install
```

---

## Experiments

Short description of our three sets of experiments that together compose single image processing pipeline in this order:

1. **Semantic (un/semi)supervised segmentation**
1. **Center detection and ellipse fitting**
1. **Region growing with the learned shape prior**


### Annotation tools

We introduce some useful tools for work with image annotation and segmentation.

* **Quantization:** in case you have some smooth colour labelling in your images you can remove them with following quantisation script.
    ```bash
    python handling_annotations/run_image_color_quantization.py \
        -imgs "./data_images/drosophila_ovary_slice/segm_rgb/*.png" \
        -m position -thr 0.01 --nb_workers 2
    ```
* **Paint labels:** concerting image labels into colour space and other way around.
    ```bash
    python handling_annotations/run_image_convert_label_color.py \
        -imgs "./data_images/drosophila_ovary_slice/segm/*.png" \
        -out ./data_images/drosophila_ovary_slice/segm_rgb
    ```
* **Visualisation:** having input image and its segmentation we can use simple visualisation which overlap the segmentation over input image. 
    ```bash
    python handling_annotations/run_overlap_images_segms.py \
        -imgs "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -segs ./data_images/drosophila_ovary_slice/segm \
        -out ./results/overlap_ovary_segment
    ```
* **In-painting** selected labels in segmentation.
    ```bash
    python handling_annotations/run_segm_annot_inpaint.py \
        -imgs "./data_images/drosophila_ovary_slice/segm/*.png" \
        --label 4
    ```
* **Replace labels:** change labels in input segmentation into another set of labels in 1:1 schema.
    ```bash
    python handling_annotations/run_segm_annot_relabel.py \
        -out ./results/relabel_center_levels \
        --label_old 2 3 --label_new 1 1 
    ```


### Semantic (un/semi)supervised segmentation

We utilise (un)supervised segmentation according to given training examples or some expectations.
![vusial debug](figures/visual_img_43_debug.jpg)

* Evaluate superpixels (with given SLIC parameters) quality against given segmentation. It helps to find out the best SLIC configuration.
    ```bash
    python experiments_segmentation/run_eval_superpixels.py \
        -imgs "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -segm "./data_images/drosophila_ovary_slice/annot_eggs/*.png" \
        --img_type 2d_split \
        --slic_size 20 --slic_regul 0.25 --slico
    ```
* Perform **Un-Supervised** segmentation in images given in CSV
    ```bash
    python experiments_segmentation/run_segm_slic_model_graphcut.py \
       -l ./data_images/langerhans_islets/list_lang-isl_imgs-annot.csv -i "" \
       -cfg experiments_segmentation/sample_config.yml \
       -o ./results -n langIsl --nb_classes 3 --visual --nb_workers 2
    ```
    OR specified on particular path:
    ```bash
    python experiments_segmentation/run_segm_slic_model_graphcut.py \
       -l "" -i "./data_images/langerhans_islets/image/*.jpg" \
       -cfg ./experiments_segmentation/sample_config.yml \
       -o ./results -n langIsl --nb_classes 3 --visual --nb_workers 2
    ```
    ![unsupervised](figures/imag-disk-20_gmm.jpg)
* Perform **Supervised** segmentation with afterwards evaluation.
    ```bash
    python experiments_segmentation/run_segm_slic_classif_graphcut.py \
        -l ./data_images/drosophila_ovary_slice/list_imgs-annot-struct.csv \
        -i "./data_images/drosophila_ovary_slice/image/*.jpg" \
        --path_config ./experiments_segmentation/sample_config.yml \
        -o ./results -n Ovary --img_type 2d_split --visual --nb_workers 2
    ```
    ![supervised](figures/imag-disk-20_train.jpg)
* Perform **Semi-Supervised** is using the the supervised pipeline with not fully annotated images.
* For both experiment you can evaluate segmentation results.
    ```bash
    python experiments_segmentation/run_compute-stat_annot-segm.py \
        -a "./data_images/drosophila_ovary_slice/annot_struct/*.png" \
        -s "./results/experiment_segm-supervise_ovary/*.png" \
        -i "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -o ./results/evaluation --visual
    ```
    ![vusial](figures/segm-visual_D03_sy04_100x.jpg)

The previous two (un)segmentation accept [configuration file](experiments_segmentation/sample_config.yml) (YAML) by parameter `-cfg` with some extra parameters which was not passed in arguments, for instance:
```yaml
slic_size: 35
slic_regul: 0.2
features: 
  color_hsv: ['mean', 'std', 'eng']
classif: 'SVM'
nb_classif_search: 150
gc_edge_type: 'model'
gc_regul: 3.0
run_LOO: false
run_LPO: true
cross_val: 0.1
```

### Center detection and ellipse fitting

In general, the input is a formatted list (CSV file) of input images and annotations. Another option is set by `-list none` and then the list is paired with given paths to images and annotations.

**Experiment sequence is the following:**

1. We can create the annotation completely manually or use the following script which uses annotation of individual objects and create the zones automatically.
    ```bash
    python experiments_ovary_centres/run_create_annotation.py
    ```
1. With zone annotation, we train a classifier for centre candidate prediction. The annotation can be a CSV file with annotated centres as points, and the zone of positive examples is set uniformly as the circular neighbourhood around these points. Another way (preferable) is to use an annotated image with marked zones for positive, negative and neutral examples.
    ```bash
    python experiments_ovary_centres/run_center_candidate_training.py -list none \
        -segs "./data_images/drosophila_ovary_slice/segm/*.png" \
        -imgs "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -centers "./data_images/drosophila_ovary_slice/center_levels/*.png" \
        -out ./results -n ovary
    ```
1. Having trained classifier we perform center prediction composed from two steps: i) center candidate clustering and ii) candidate clustering.
    ```bash
    python experiments_ovary_centres/run_center_prediction.py -list none \
        -segs "./data_images/drosophila_ovary_slice/segm/*.png" \
        -imgs "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -centers ./results/detect-centers-train_ovary/classifier_RandForest.pkl \
        -out ./results -n ovary
    ```
1. Assuming you have an expert annotation you can compute static such as missed eggs.
    ```bash
    python experiments_ovary_centres/run_center_evaluation.py
    ```
1. This is just cut out clustering in case you want to use different parameters.
    ```bash
    python experiments_ovary_centres/run_center_clustering.py \
        -segs "./data_images/drosophila_ovary_slice/segm/*.png" \
        -imgs "./data_images/drosophila_ovary_slice/image/*.jpg" \
        -centers "./results/detect-centers-train_ovary/candidates/*.csv" \
        -out ./results
    ```
1. Matching the ellipses to the user annotation.
    ```bash
    python experiments_ovary_detect/run_ellipse_annot_match.py \
        -info "~/Medical-drosophila/all_ovary_image_info_for_prague.txt" \
        -ells "~/Medical-drosophila/RESULTS/3_ellipse_ransac_crit_params/*.csv" \
        -out ~/Medical-drosophila/RESULTS
    ```
1. Cut eggs by stages and norm to mean size.
    ```bash
    python experiments_ovary_detect/run_ellipse_cut_scale.py \
        -info ~/Medical-drosophila/RESULTS/info_ovary_images_ellipses.csv \
        -imgs "~/Medical-drosophila/RESULTS/0_input_images_png/*.png" \
        -out ~/Medical-drosophila/RESULTS/images_cut_ellipse_stages
    ```
1. Rotate (swap) extracted eggs according the larger mount of mass.
    ```bash
    python experiments_ovary_detect/run_egg_swap_orientation.py \
        -imgs "~/Medical-drosophila/RESULTS/atlas_datasets/ovary_images/stage_3/*.png" \
        -out ~/Medical-drosophila/RESULTS/atlas_datasets/ovary_images/stage_3
    ```

![ellipse fitting](figures/insitu7544_ellipses.jpg)

### Region growing with a shape prior

In case you do not have estimated object centres, you can use [plugins](ij_macros) for landmarks import/export for [Fiji](http://fiji.sc/).

**Note:** install the multi-snake package which is used in multi-method segmentation experiment.
```bash
pip install --user git+https://github.com/Borda/morph-snakes.git
```

**Experiment sequence is the following:**

1. Estimating the shape model from set training images containing a single egg annotation.
    ```bash
    python experiments_ovary_detect/run_RG2Sp_estim_shape-models.py  \
        -annot "~/Medical-drosophila/egg_segmentation/mask_2d_slice_complete_ind_egg/*.png" \
        -out ./data_images -nb 15
    ```
1. Run several segmentation techniques on each image.
    ```bash
    python experiments_ovary_detect/run_ovary_egg-segmentation.py  \
        -list ./data_images/drosophila_ovary_slice/list_imgs-segm-center-points.csv \
        -out ./results -n ovary_image --nb_workers 1 \
        -m ellipse_moments \
           ellipse_ransac_mmt \
           ellipse_ransac_crit \
           GC_pixels-large \
           GC_pixels-shape \
           GC_slic-large \
           GC_slic-shape \
           rg2sp_greedy-mixture \
           rg2sp_GC-mixture \
           watershed_morph
    ```
1. Evaluate your segmentation ./results to expert annotation.
    ```bash
    python experiments_ovary_detect/run_ovary_segm_evaluation.py --visual
    ```
1. In the end, cut individual segmented objects comes as minimal bounding box.
    ```bash
    python experiments_ovary_detect/run_cut_segmented_objects.py \
        -annot "./data_images/drosophila_ovary_slice/annot_eggs/*.png" \
        -img "./data_images/drosophila_ovary_slice/segm/*.png" \
        -out ./results/cut_images --padding 50
    ```
1. Finally, performing visualisation of segmentation results together with expert annotation.
    ```bash
    python experiments_ovary_detect/run_export_user-annot-segm.py
    ```
    ![user-annnot](figures/insitu7545_user-annot-segm.jpg)

---

## References

For complete references see [BibTex](docs/references.bib).
1. Borovec J., Svihlik J., Kybic J., Habart D. (2017). **Supervised and unsupervised segmentation using superpixels, model estimation, and Graph Cut.** SPIE Journal of Electronic Imaging 26(6), 061610. [DOI: 10.1117/1.JEI.26.6.061610](http://doi.org/10.1117/1.JEI.26.6.061610).
1. Borovec J., Kybic J., Nava R. (2017) **Detection and Localization of Drosophila Egg Chambers in Microscopy Images.** In: Wang Q., Shi Y., Suk HI., Suzuki K. (eds) Machine Learning in Medical Imaging. MLMI 2017. LNCS, vol 10541. Springer, Cham. [DOI: 10.1007/978-3-319-67389-9_3](http://doi.org/10.1007/978-3-319-67389-9_3).
1. Borovec J., Kybic J., Sugimoto, A. (2017). **Region growing using superpixels with learned shape prior.** SPIE Journal of Electronic Imaging 26(6),  061611. [DOI: 10.1117/1.JEI.26.6.061611](http://doi.org/10.1117/1.JEI.26.6.061611).
