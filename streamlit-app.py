"""
Simple StreamLit app for unsupervised segmentation

>> streamlit run streamlit-app.py
"""

import matplotlib.pyplot as plt
import streamlit as st
from skimage import segmentation as ski_segm

from imsegm.pipelines import estim_model_classes_group, segment_color2d_slic_features_model_graphcut

FEATURES_SET_MIN = {
    'color': (
        'mean',
        'std',
        # 'energy',
    ),
    'tLM_short': ('mean', ),
}


def process_image(
    img_path: str = 'data-images/drosophila_disc/image/img_5.jpg',
    nb_classes: int = 2,
    spx_size: int = 30,
    spx_regul: float = 0.5,
    gc_regul: float = 0.4,
    streamlit_app: bool = False,
):
    if not img_path:
        return

    img = plt.imread(img_path)
    # if streamlit_app:
    #     st.image(img)

    debug = {}
    spx_config = dict(sp_size=spx_size, sp_regul=spx_regul, dict_features=FEATURES_SET_MIN)
    model, _ = estim_model_classes_group([img], nb_classes=nb_classes, **spx_config)
    segm, _ = segment_color2d_slic_features_model_graphcut(
        img, model, **spx_config, gc_regul=gc_regul, debug_visual=debug
    )

    print(debug.keys())
    spx_contour = ski_segm.mark_boundaries(debug['image'], debug['slic'], color=(1, 0, 0), mode='subpixel')

    fig, axarr = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
    axarr[0, 0].imshow(debug['image'])
    axarr[0, 1].imshow(spx_contour)
    axarr[0, 2].imshow(debug['slic_mean'] / 255.)
    axarr[1, 0].imshow(debug['img_graph_edges'])
    axarr[1, 1].imshow(debug['img_graph_segm'])
    axarr[1, 2].imshow(segm)
    if streamlit_app:
        st.pyplot(fig)


st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Image segmentation Demo")
nb_cls = st.sidebar.slider('number classes', min_value=2, max_value=10, value=3, step=1)
sz_spx = st.sidebar.slider('SuperPixel edge size [px]', min_value=5, max_value=200, value=35, step=10)
reg_spx = st.sidebar.slider('SuperPixel regularization', min_value=0.1, max_value=1.0, value=0.4, step=0.05)
reg_gc = st.sidebar.slider('GraphCut regularization', min_value=0.1, max_value=20., value=0.7, step=0.05)
img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg'])

# run the app
process_image(img_file, nb_classes=nb_cls, spx_size=sz_spx, spx_regul=reg_spx, streamlit_app=True)
# process_image(model)  # dry rn with locals
