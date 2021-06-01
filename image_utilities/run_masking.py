import yaml
import click
import sys
import os
import glob
sys.path.append('../image_utilities')

from utils import *
from mask_images import *


def execute(config_path, download, dtype):
    with open(config_path, "r") as config:
        params = yaml.safe_load(config)

    if download:
        session = boto3.session.Session(profile_name=params['AWS']['profile'])
        s3 = session.resource('s3')

        keys = list_objects(
            s3, params['AWS']['bucket'], params['AWS']['prefix'],
            params['AWS']['suffix']
        )
        local_files = download_s3_files(
            s3_resource=s3, bucket=params['AWS']['bucket'],
            keys=keys, local_dir=params['Mask']['local_dir']
        )

    else:
        # read in image files
        local_files = sorted(
            glob.glob(os.path.join(params['Mask']['local_dir'], '*.tif'))
        )

    # output directory, create if needed
    out_path = params['Mask']['output_path']
    if os.path.isdir(out_path) is False:
        os.mkdir(out_path)

    mask_images(local_files, out_path, dtype)

    # print(masked_image)

@click.command()
@click.option('--config_path', default='config.yaml',
              help='Directory of the config file')
@click.option('--download', is_flag=True, help='Download files from S3 first?')
@click.option('--dtype', default='int16',
              help='Data type for output masked image')
def main(config_path, download, dtype):
    execute(config_path, download, dtype)

if __name__=='__main__':
    main()



