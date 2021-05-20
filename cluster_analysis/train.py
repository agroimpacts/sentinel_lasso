import os
import gc
import pickle

import random
import numpy as np
import rasterio
import pandas as pd
from sklearn.cluster import KMeans
from rasterio.windows import Window

from utils import *

def train(params):

    # parameters
    dir_data = params['dir_data']
    dir_out = params['dir_train']
    fn_catalog_train = params['catalog_train'] if os.path.isabs(params['catalog_train']) \
        else os.path.join(dir_data, params['catalog_train'])
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    ## cluster parameters
    sample_rate = params['sample_rate']
    clusters = params['cluster_number']

    # get image catalog and train
    catalog_train = pd.read_csv(fn_catalog_train)
    dirs_img = catalog_train[params['data_col']].drop_duplicates()

    # build dataset
    print("Building dataset for train")
    data = []

    if len(dirs_img) == 1:
        dir_img = dirs_img[0]
        print(dir_img)
        src = rasterio.open(dir_img)
        height = src.height
        width = src.width

        chip_size = 2000
        # if img smaller than chip_size
        if src.height < chip_size or src.width < chip_size:
            data.extend(random.sample(get_flat(src.read()).tolist(),
                                      int(src.height * src.width * sample_rate)))
        ## else take samples from sub-grids
        else:
            std_img = np.nanstd(src.read())  # std_img as baseline
            col_off = 0
            while col_off + chip_size < width:
                row_off = 0
                while row_off + chip_size < height:
                    img_chip = src.read(window=Window(col_off, row_off, chip_size, chip_size))
                    # std_chip larger than std_img will get more samples and vise versa
                    sample_rate_chip = np.nanstd(img_chip) * sample_rate / std_img

                    img_chip = np.nan_to_num(img_chip, -999) # incase there's any nan
                    try:
                        lasso = random.sample(get_flat(img_chip).tolist(),
                                          int(min(img_chip.size * sample_rate_chip, 1)))
                        data.extend(lasso)
                    except:
                        pass

                    row_off += chip_size
                col_off += chip_size


    else:

        for dir_img in dirs_img:
            # dir_img = dir_img if os.path.isabs(dir_img) else os.path.join(dir_data, dir_img)
            print(dir_img)
            src = rasterio.open(dir_img)
            lasso = random.sample(np.nan_to_num(get_flat(src.read()), -999).tolist(), int(src.height * src.width * sample_rate))
            data.extend(lasso)

    #         meta = src.meta
    data = np.array(data)
    ## Train
    print("Start train")

    model = KMeans(n_clusters=clusters)
    model.fit(data)

    pickle.dump(model, open(os.path.join(dir_out, "model.sav"), 'wb'))
    del lasso, data
    gc.collect()

    print("Finished train. The model is saved at {}".format(os.path.join(dir_out, "model.sav")))
