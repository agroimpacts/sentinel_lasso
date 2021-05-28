import os
import pickle

import rasterio
from rasterio.merge import merge
import pandas as pd
import numpy as np

from utils import *

def predict(params, model=None):

    mode = params['mode']
    dir_data = params['dir_data']
    dir_model = params['dir_model']
    dir_out = params['dir_predict']
    fn_out = params['fn_predict']
    dir_catalog_predict = params['catalog_predict'] if os.path.isabs(params['catalog_predict']) \
        else os.path.join(dir_data, params['catalog_predict'])
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    # load model and predict catalog
    model = pickle.load(open(dir_model, 'rb')) if model is None else model
    catalog_predict = pd.read_csv(dir_catalog_predict)

    # predict
    if mode == 'single':
        # get merged image
        dirs_img = catalog_predict[params['data_col']].drop_duplicates()
        for dir_img in dirs_img:
            dir_img = dir_img if os.path.isabs(dir_img) \
                else os.path.join(dir_data, dir_img)
            src_img = rasterio.open(dir_img)

            meta = src_img.meta
            img = np.nan_to_num(src_img.read(), -999)
            _, h, w = img.shape
            img = get_flat(img)
            # predict and write into tif

            out = model.predict(img).reshape((h, w))

            meta.update({
                'dtype': 'int8',
                'count': 1,
            })
            fn_img = os.path.split(dir_img)[-1]
            with rasterio.open(os.path.join(dir_out, fn_out.format(fn_img)),
                               "w",
                               **meta) as dst:
                dst.write(out.astype(meta['dtype']), 1)

    elif mode == 'merge_row':
        row_ids_col = params['tilerow_index_col']
        row_ids = catalog_predict[row_ids_col].unique()
        for idx in row_ids:
            catalog = catalog_predict.query("{}=={}".format(row_ids_col, idx))
            dirs_img = catalog[params['data_col']].drop_duplicates()

            # merge images with the same row index
            lasso_for_merge =[]
            for dir_img in dirs_img:
                dir_img = dir_img if os.path.isabs(dir_img) \
                    else os.path.join(dir_data, dir_img)
                print(dir_img)
                lasso_for_merge.append(rasterio.open(dir_img))

            meta = lasso_for_merge[0].meta
            lasso_merged, out_tf = merge(lasso_for_merge)
            _, h, w = lasso_merged.shape
            lasso_merged = get_flat(lasso_merged)

            # predict and write into tif
            out = model.predict(lasso_merged).reshape((h, w))

            meta.update({
                'dtype': 'int8',
                'count': 1,
                'height': h,
                'width': w,
                'transform': out_tf
            })

            with rasterio.open(os.path.join(dir_out, fn_out.format(idx)),
                               "w",
                               **meta) as dst:
                dst.write(out.astype(meta['dtype']), 1)

            print("Finish prediction on row: {}".format(idx))

    elif mode == 'merge_all':

        # get merged image
        dirs_img = catalog_predict[params['data_col']].drop_duplicates()
        lasso_for_merge =[]
        for dir_img in dirs_img:
            dir_img = dir_img if os.path.isabs(dir_img) \
                else os.path.join(dir_data, dir_img)
            print(dir_img)
            lasso_for_merge.append(rasterio.open(dir_img))

        meta = lasso_for_merge[0].meta
        lasso_merged, out_tf = merge(lasso_for_merge)
        _, h, w = lasso_merged.shape
        lasso_merged = get_flat(lasso_merged)

        # predict and write into tif
        out = model.predict(lasso_merged).reshape((h, w))

        meta.update({
            'dtype': 'int8',
            'count': 1,
            'height': h,
            'width': w,
            'transform': out_tf
        })

        with rasterio.open(os.path.join(dir_out, fn_out.format(0)),
                           "w",
                           **meta) as dst:
            dst.write(out.astype(meta['dtype']), 1)

        print("Finish prediction on: {}".format(fn_out.format(0)))

    else:
        assert ('Mode not applicable')




