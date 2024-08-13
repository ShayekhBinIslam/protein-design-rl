import boto3
import os
import sys


def download_files_resolver(value: str, download_dir: str = "/tmp") -> str:
    """
    Downloads a file from a given URI to a specified directory.
    Currently, supports downloading from Amazon S3.

    Args:
        value (str): The URI of the file to be downloaded. Supported
            protocols are s3:// for Amazon S3.
        download_dir (str, optional): The directory where the downloaded
            file will be stored. Defaults to '/tmp'.

    Returns:
        str: Path to the downloaded file.

    Raises:
        AssertionError: If the provided download directory does not exist.
        ValueError: If the provided URI has an unsupported protocol.

    Examples:
        >>> download_files_resolver("s3://bucket-name/path/to/file.txt")
        [ConfigResolver] Downloading file from s3://bucket-name/path/to/file.txt to /tmp/file.txt:
        [/==================================================]
        "/tmp/file.txt"

        # Can be used as yaml/OmegaConf resolver. E.g.:
        py:
            import OmegaConf
            OmegaConf.register_new_resolver("download", download_files_resolver)

        yaml:
            param_path: ${download:${oc.env:ANYSCALE_ARTIFACT_STORAGE}/mp.pt}
            param_path: ${download:${oc.env:ANYSCALE_ARTIFACT_STORAGE}/mp.pt,/custom/download/path}

    """
    # Check if download path exists, otherwise create it
    os.makedirs(download_dir, exist_ok=True)
    assert os.path.isdir(download_dir), f"[ConfigResolver] Download directory {download_dir} does not exist"

    if value.startswith("s3://"):
        # Download from s3

        # Initialize the S3 client
        s3 = boto3.client('s3')

        def parse_s3_uri(uri):
            """
            Parse an S3 URI into bucket and key.
            """
            assert uri.startswith("s3://")
            parts = uri[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else None
            return bucket, key

        # Extract bucket and object key
        bucket_name, object_key = parse_s3_uri(value)

        # Generate progress bar
        meta_data = s3.head_object(Bucket=bucket_name, Key=object_key)
        total_length = int(meta_data.get('ContentLength', 0))
        downloaded = 0

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(50 * downloaded / total_length)
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
            sys.stdout.flush()

        # Define the local path where you want to save the file
        local_file_path = os.path.basename(value)

        # Download the file
        download_path = os.path.join(download_dir, local_file_path)
        print(f"[ConfigResolver] Downloading file from {value} to {download_path}: ")
        if os.path.exists(download_path):
            print(f"[ConfigResolver] File already exists at {download_path}. Skipping download.")
        else:
            s3.download_file(
                bucket_name, object_key, download_path, Callback=progress
            )
    else:
        raise ValueError(f"[ConfigResolver] Unknown download protocol: {value}")

    return download_path
