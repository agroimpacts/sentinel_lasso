import os
import rasterio
import numpy as np
from rasterio.merge import merge
from skimage.morphology import remove_small_objects, remove_small_holes

def reclass(params):
    dir_out = params['dir_predict']
    dir_reclass = params['dir_reclass']
    fn_reclass = params['fn_reclass']
    merge_result = params['merge_reclassed']
    reclass_values = input("Enter the category indices to reclass into category 1 \nSkip this if the values are set in config.yaml: ")
    # print(reclass_values)
    reclass_values = params['reclass_values'] if reclass_values=="" else reclass_values
    reclass_values_ls = reclass_values.split(",") if isinstance(reclass_values, str) \
        else reclass_values
    reclass_statement = "| ".join(["(img=={})".format(m) for m in reclass_values_ls])
    if not os.path.exists(dir_reclass):
        os.mkdir(dir_reclass)

    for_reclass = [m for m in os.listdir(dir_out) if m.endswith(".tif")]

    for data in for_reclass:
        with rasterio.open(os.path.join(dir_out, data)) as src:
            img = src.read()
            meta = src.meta

        # reclass
        out = np.where(eval(reclass_statement), 1, 0)
        arr = out > 0
        out = remove_small_holes(arr, area_threshold=3000, connectivity=2)
        out = remove_small_objects(out, min_size=2000, in_place=False, connectivity=1).astype(int)

        with rasterio.open(os.path.join(dir_reclass, (fn_reclass.format(data.split(".")[0]))), "w", **meta) as dst:
            dst.write(out)

    if merge_result:
        fn_merge = params['fn_merge']
        reclass_for_merge = [rasterio.open(os.path.join(dir_reclass, m)) for m in os.listdir(dir_reclass) if
                             m.endswith('.tif')]
        meta = reclass_for_merge[0].meta
        reclass_merged, out_tf = merge(reclass_for_merge)

        _, h, w = reclass_merged.shape
        meta.update({
            'height': h,
            'width': w,
            'transform': out_tf
        })

        arr = reclass_merged > 0
        reclass_merged = remove_small_holes(arr, area_threshold=3000, connectivity=2)
        reclass_merged = remove_small_objects(reclass_merged, min_size=2000, in_place=False, connectivity=1).astype(int)

        with rasterio.open(os.path.join(dir_reclass, fn_merge),
                           "w",
                           **meta) as dst:
            dst.write(reclass_merged)
