import os
import yaml
import pickle

import rasterio

import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.fill import fillnodata

# from utils import *
from guided_filter import guidefilter_ite2, guidefilter


def preprocess(file, config="./config.yaml"):
    """
    This function applies preprocess of resampling and filtering on each downloaded raw Sentinel file. The preprocessed
    image is saved as an .npy file for efficient usage in regression.

    Params:
        file -- (str) File name of image to be resampled
        config -- (str) A yaml file for configuration. The key named "Preprocess" is used in this function.

    """
    with open(config, "r") as config:
        params = yaml.safe_load(config)["Lasso"]

    aoi = params["aoi"]
    polarizations = params["polarizations"]

    dir_img = params["dir_img"].format(aoi)
    dir_ready = os.path.split(params["dir_ready"])[0].format(aoi)
    
    dir_meta = params["dir_meta"].format(dir_ready, aoi)

    # Filter parameters
    r = params["kernel"]
    eps = params["eps"]
    ite = params["iteration"]

    # dst metadata
    with open(dir_meta, "rb") as dst:
        dst_meta = pickle.load(dst)
        dst_canvas = np.zeros((dst_meta["height"], dst_meta["width"]), dtype=np.float64)
        dst_transform = dst_meta["transform"]
        dst_crs = dst_meta["crs"]

    
    for i in range(len(polarizations)):
	
        with rasterio.open(os.path.join(dir_img, file)) as src:
            img = src.read(i + 1)
            src_transform = src.transform
            src_crs = src.crs


        # Filter
        print("start filtering")
        I = np.where(np.isnan(img), -99, img)
        out_img = guidefilter_ite2(I, r, eps, ite)
        # out_img = guidefilter(I, I, r, eps)
        out_img = np.where(np.isnan(img), img, out_img)
        del I, img




        # # Write preprocessed images into narray
        polarization = polarizations[i]
        dir_out = params["dir_ready"].format(aoi, aoi, polarization)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)


        out_name = "{}_{}.npy".format(file.split(".")[0], polarization)

        np.save(os.path.join(dir_out.format(aoi, polarization), out_name),
                reproject(
                    source=out_img,
                    destination=dst_canvas,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)[0]
                )
    print("finish: ", out_name)
