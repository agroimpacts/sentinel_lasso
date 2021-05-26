# Read in image stack, create and apply common mask band, and write out to
# geotiff
import os
import numpy as np
import rioxarray as rxr
import glob
import re

# Set-up directories
ROOT = os.path.abspath(os.getcwd())
data_path = os.path.join(ROOT, "external/data/volumetric/condensed/")
out_path = os.path.join(ROOT, "external/data/volumetric/condensed_masked/")

# read in image files
img_files = sorted(
    glob.glob(os.path.join(data_path, '*.tif'))
)

# process files
for i in img_files:
    print("Processing " + os.path.basename(i))
    stack = rxr.open_rasterio(i, masked=True).squeeze()

    # create common mask (union of all NAs in each band)
    mask = stack.mean(axis=0, skipna=False)
    mask = mask / mask
    stack_mask = (stack * mask).astype('uint8') # save as 8 integer

    # get no data value from mask, write to temporary raster
    nodata = stack_mask.min()
    temp_stack = stack_mask.rio.write_nodata(nodata)  # set NA

    # and then to new geotiff
    out_name = re.sub(".tif", "_masked.tif", os.path.basename(i))
    out_file = os.path.join(out_path, out_name)
    temp_stack.rio.to_raster(out_file)