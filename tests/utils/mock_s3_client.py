from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

    MockS3Client = S3Client
else:
    MockS3Client = Mock


def mock_s3_client() -> MockS3Client:
    s3_client = Mock()
    s3_client.upload_file.return_value = None
    return s3_client
