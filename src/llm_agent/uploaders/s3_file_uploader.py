import logging

from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client

log = logging.getLogger(__name__)


class UploadError(Exception):
    """Failure uploading file to S3 location."""


def upload_file(client: S3Client, file_name: str, bucket_name: str) -> str:
    """Upload a file to an S3 bucket.

    Args:
        client (S3Client): A boto3 `client.S3Client`

        file_name (str): Path to the file to upload

        bucket_name (str): The name of the bucket to which to upload the file

    Returns:
        Public path to the resource in S3

    Raises:
        UploadError: If upload fails
    """
    try:
        object_key = file_name.split("/")[-1]
        client.upload_file(file_name, bucket_name, object_key)
    except ClientError as e:
        log.exception("Error uploading file to S3.")
        raise UploadError from e

    return f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{object_key}"
