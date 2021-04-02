import os
import click
import yaml
import pickle
import numpy as np
import rasterio
from datetime import datetime

from get_preprocess import preprocess
from get_lasso import getHarmonics
from get_mosaic import get_mosaic
from utils import *


def run_it(config):

    ## Params
    with open(config, "r") as config:
        params = yaml.safe_load(config)["Lasso"]

    aoi = params["aoi"]
    polarizations = params["polarizations"]

    # Directories
    dir_raw = params["dir_raw"].format(aoi)
    dir_img = params["dir_img"].format(aoi)
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)
    dir_ready = os.path.split(params["dir_ready"])[0].format(aoi)
    dir_meta = params["dir_meta"].format(dir_ready, aoi)
    if not os.path.isdir(dir_ready):
        os.mkdir(dir_ready)
    dir_out = os.path.split(params["dir_out"])[0].format(aoi)
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    ## Merge
    print("mosaic start:", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    raw_ls = [m for m in os.listdir(dir_raw) if m.endswith(".tif") and "S1B" in m]
    raw_ls.sort()
    print(raw_ls)

    src_to_mosaic_all = []
    while len(raw_ls) > 0:
        # print(len(raw_ls))
        src_files_to_mosaic = []

        mosaic_key = raw_ls[0].split("_")[4].split("T")[0]
        # out_name = raw_ls[0]

        for raw in raw_ls.copy():

            if raw.split("_")[4].split("T")[0] == mosaic_key:
                # src = rasterio.open(raw)
                src_files_to_mosaic.append(raw)
                raw_ls.remove(raw)
            else:
                break

        src_to_mosaic_all.append(src_files_to_mosaic)

    finished = [m for m in os.listdir(dir_img) if m.endswith(".tif")]
    print(len(finished))
    for src in src_to_mosaic_all:
        # print(len(src))
        if src[0] in finished:
            src_to_mosaic_all.remove(src)

    multicore(get_mosaic, src_to_mosaic_all, 4)
    # for m in src_to_mosaic_all:
    #     get_mosaic(m)
    # # get_mosaic(src_to_mosaic_all[0])
    print("mosaic finished:", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))


    ## Resample and Filter
    print("Preprocess start:", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    file_ls = [m for m in os.listdir(dir_img) if m.endswith(".tif")]

    # save metadata
    with rasterio.open(os.path.join(dir_img, file_ls[0]))as dst:
        dst_meta = dst.meta
    file_meta = dir_meta.format(aoi)
    with open(file_meta, "wb") as file:
        pickle.dump(dst_meta, file)


    print(len(file_ls))
    finished = ["_".join(m.split("_")[:-1]) + ".tif" for m in os.listdir(params["dir_ready"].format(aoi, aoi, "VV"))]
    file_ls = [m for m in file_ls if m not in finished]
    print("Number of files left for preprocessing", len(file_ls))

    # parallelize preprocess
    multicore(preprocess, file_ls, 4)

    print("lasso start:", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    ## Get Lasso
    harPair = params["Harmonic_pairs"]
    lasso_coefs = np.memmap("tmp.dat",
                            dtype="float64",
                            mode="w+",
                            shape=(2 + harPair * 2, len(polarizations), dst_meta["height"], dst_meta["width"]))
    inds = range(dst_meta["height"])

    # get harmonics
    # getHarmonics(inds[0])
    multicore(getHarmonics, inds)

    # Write tif
    for i in range(len(lasso_coefs)):
        fn_out = os.path.split(params["dir_out"])[1].format(aoi, i)
        with rasterio.open(os.path.join(dir_out, fn_out), "w", **dst_meta) as dst:
            dst.write(lasso_coefs[i, :, :, :])

    del lasso_coefs
    print("lasso finished:", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

@click.command()
@click.option('--config', default='./config,yaml',
              help='Directory of config file')
def main(config):
    run_it(config)

if __name__ == "__main__":
    main()
