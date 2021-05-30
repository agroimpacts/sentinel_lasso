import os
import rioxarray as rxr
import re

def mask_image(image_file, out_file=None, dtype='int16'):
    """
    Read in and mask a single input image using rioxarray, with the option to
    write masked image to disk. Mask is created from union of all bands' NA
    values.

        Params:
            image_file (str): Path to local geotiff
            out_file (str): Path/name for output geotiff (default is None)
            dtype (str): Data type for masked image (default 'int16')

        Returns:
            A masked image in memory
    """
    image = rxr.open_rasterio(image_file, masked=True).squeeze()

    # create common mask (union of all NAs in each band)
    mask = image.mean(axis=0, skipna=False)
    mask = mask / mask
    masked_image = (image * mask).astype(dtype) # save as 8 integer

    # Write to new geotiff, if output path given
    if out_file is not None:
        nodata = masked_image.min()  # no data value
        temp_image = masked_image.rio.write_nodata(nodata)  # set NA
        temp_image.rio.to_raster(out_file)

    return masked_image

def mask_images(image_files, dtype='int16'):
    """
    Read in and mask a list of input images and write to geotiff. The mask is
    created from union of all bands' NA values. The file names of the masked
    images are formed form the input names with '_masked' appended.

        Params:
            image_files (list): List of paths for local geotiffs
            dtype (str): Data type for masked image (default 'int16')

        Returns:
            The last image in the list in memory.
    """
    for image in image_files:
        image_name = os.path.basename(image)
        masked_image_name = re.sub('.tif', '_masked.tif', image_name)
        out_file = os.path.join(out_path, masked_image_name)

        print('Processing {}'.format(image_name))

        masked_image = mask_image(image, out_file, dtype)

        return masked_image
