import os
import gc
import pickle

import random
import numpy as np
import rasterio
import pandas as pd
from sklearn.cluster import KMeans

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
    for dir_img in dirs_img:
        dir_img = dir_img if os.path.isabs(dir_img) else os.path.join(dir_data, dir_img)
        print(dir_img)
        src = rasterio.open(dir_img)
        lasso = random.sample(get_flat(src.read()).tolist(), int(src.height * src.width * sample_rate))
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
