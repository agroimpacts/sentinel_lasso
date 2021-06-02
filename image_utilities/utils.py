import os
import boto3


def aws_client(resource, profile='default'):
    """
    Create a boto3 client for an AWS resource.

        Params:
            resource (str): The AWS resource to connect to, e.g. 's3'
            profile (str): A named AWS cli profile defaulting to 'default'

        Returns:
            AWS client
    """
    boto3.setup_default_session(profile_name=profile)
    client = boto3.client(resource)

    return client

def aws_resource(resource, profile='default'):
    """
    Set up an AWS resource.

        Params:
            resource (str): The AWS resource to connect to, e.g. 's3'
            profile (str): A named AWS cli profile defaulting to 'default'

        Returns:
            AWS resource
    """

    if profile is not None:
        session = boto3.session.Session(profile_name=profile)
        _resource = session.resource(resource)
    else:
        _resource = boto3.resource(resource)

    return _resource

def list_objects(s3_resource, bucket, prefix, suffix=None):
    """
    Get list of keys in an S3 bucket, filtering by prefix and suffix. Function
    developed by Kaixi Zhang as part of AWS_S3 class and adapted slightly here.
    This function retrieves all matching objects, and is not subject to the 1000
    item limit.

        Params:
            s3_resource (object): A boto3 s3 resource object
            bucket (str): Name of s3 bucket to list
            prefix (str): Prefix within bucket to search
            suffix (str, list): Optional string or string list of file endings

        Returns:
            List of s3 keys
    """
    keys = []
    if s3_resource is not None:
        s3_bucket = s3_resource.Bucket(bucket)
        for obj in s3_bucket.objects.filter(Prefix=prefix):
            # if no suffix given, add all objects with the prefix
            if suffix is None:
                keys.append(str(obj.key))
            else:
                # add all objects that ends with the given suffix
                if isinstance(suffix, list):
                    for _suffix in suffix:
                        if obj.key.endswith(_suffix):
                            keys.append(str(obj.key))
                            break
                else:
                    # suffix is a single string
                    if obj.key.endswith(suffix):
                        keys.append(str(obj.key))
    else:
        print
        'Warning: please first create an s3 resource'
    return keys


# def get_matching_s3_keys(client, bucket, prefix='', suffix=''):
#     """
#     Generate the keys in an S3 bucket.
#     From https://alexwlchan.net/2017/07/listing-s3-keys/
#
#         Params:
#             client (obj): AWS S3 client
#             bucket (str): Name of the S3 bucket.
#             prefix (str): Only fetch keys that start with this prefix (optional).
#             suffix (str): Only fetch keys that end with this suffix (optional).
#     """
#     kwargs = {'Bucket': bucket, 'Prefix': prefix}
#     while True:
#         resp = client.list_objects_v2(**kwargs)
#         for obj in resp['Contents']:
#             key = obj['Key']
#             if key.endswith(suffix):
#                 yield key
#
#         try:
#             kwargs['ContinuationToken'] = resp['NextContinuationToken']
#         except KeyError:
#             break


def download_s3_files(s3_resource, bucket, keys, local_dir):
    """
    Downloads a file or list of files from S3 to a local directory.

        Params:
            s3_resource (object): A boto3 s3 resource object
            bucket (str): AWS bucket name
            keys (str, list): s3 keys for 1 (string) or more files
            local_dir (str): Local directory to save file to

        Returns:
            List of paths to local files
    """
    s3_bucket = s3_resource.Bucket(bucket)

    # convert single path string to list
    if type(keys) is str:
        print('Converting key to list')
        keys = [keys]

    local_files = []
    for key in keys:
        file_name = os.path.basename(key)
        print('Downloading {}'.format(file_name))

        local_file = os.path.join(local_dir, file_name)
        # s3client.download_file(client, bucket, key, local_file)
        s3_bucket.download_file(key, local_file)

        local_files.append(local_file)

    return local_files
