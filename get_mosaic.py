import os
import yaml
import rasterio
from rasterio.merge import merge
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.fill import fillnodata

buffer = 80


def get_mosaic(src_files_to_mosaic, config="./config.yaml"):

    with open(config, "r") as config:
        params = yaml.safe_load(config)["Lasso"]
    
    aoi = params["aoi"]
    dir_raw = params["dir_raw"].format(aoi)
    dir_out = params["dir_img"].format(aoi)
    out_name = src_files_to_mosaic[0]
    src_to_mosaic = [rasterio.open(os.path.join(dir_raw, m)) for m in src_files_to_mosaic]
    
    if len(src_to_mosaic) > 1:

        mosaic, out_tf = merge(src_to_mosaic)

        # see using merge function or fill nan data
        x_min, x_max, y_max, y_min = [None] * 4
        for raw in src_to_mosaic:
            tf = raw.transform
            x_min = min(x_min, tf[2]) if x_min else tf[2]
            x_max = max(x_max, x_min + tf[0] * raw.width) if x_max else x_min + tf[0] * raw.width
            y_max = max(y_max, tf[5]) if y_max else tf[5]
            y_min = min(y_max, y_max + tf[4] * raw.height) if y_min else y_max + tf[4] * raw.height

        y_min_mosaic = out_tf[5] + out_tf[4] * mosaic.shape[1]
        x_max_mosaic = out_tf[2] + out_tf[0] * mosaic.shape[2]

        if y_min == y_min_mosaic and x_max == x_max_mosaic:

            out = mosaic

        else:

            mosaic.fill(np.nan)

            for ready in src_to_mosaic:

                ready_tf = ready.transform
                ready_crs = ready.crs
                ready = ready.read()

                to_fill_pj = np.zeros(mosaic.shape)

                for i in range(1, ready.shape[0] + 1):
                    reproject(
                        source=ready[i - 1, :, :],
                        src_transform=ready_tf,
                        src_crs=ready_crs,
                        destination=to_fill_pj[i - 1, :, :],
                        dst_transform=out_tf,
                        dst_crs=ready_crs,
                        resampling=Resampling.bilinear)

                to_fill_pj = np.where(to_fill_pj == 0, np.nan, to_fill_pj)
                mosaic = np.where((mosaic == 0) | (np.isnan(mosaic)), to_fill_pj, mosaic)

            # Fill nodata
            #             out = np.zeros(mosaic.shape)
            del ready, to_fill_pj
            # out = mosaic
            
            # Fill nodata
            out = np.zeros(mosaic.shape)

            for i in range(len(mosaic)):
                toFill = mosaic[i,:,:]

                mask = np.where(np.isnan(toFill), 0, 1)
                toFill = fillnodata(toFill, mask, max_search_distance = 2)
                out[i,:,:] = toFill
            

            del toFill, mosaic

    else:
        with rasterio.open(os.path.join(dir_raw, out_name), "r") as src:
            out = src.read()
            out_tf = src.transform


    ## Clip out some pixels if valid data not covering entire region
    inds = np.argwhere(np.isnan(out[0, :, :]))

    if len(inds) > 0:

        _, h, w = out.shape
        x_min = min(inds[:, 0])
        y_min = min(inds[:, 1])

        off_left = buffer if y_min == 0 else 0
        off_right = buffer if y_min > 0 else 0
        off_up = buffer if x_min == 0 else 0
        off_down = buffer if x_min > 0 else 0

        mask = np.where(np.isnan(out[0, :, :]), np.isnan, 1)

        # only toward one side either horizontally or vertically
        mask = np.pad(mask, ((off_up, off_down), (off_left, off_right)),
                      'constant', constant_values=1)[0 + off_down: 0 + off_down + h, 0 + off_right: 0 + off_right + w]
        for i in range(len(out)):
            out[i, :, :] = np.where(mask == 1, out[i, :, :], np.nan)
    else:
        pass

    del mask

    ## Write out images
    out_meta = src_to_mosaic[0].meta.copy()
    out_meta.update({"height": out.shape[1],
                     "width": out.shape[2],
                     "transform": out_tf
                     })

    with rasterio.open(os.path.join(dir_out, out_name), "w", **out_meta) as dst:
        dst.write(out)
    print("Finished:", out_name)
