from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from llm_agent.uploaders.s3_file_uploader import UploadError, upload_file


def test_upload_file():
    # Setup
    s3_client = Mock()
    s3_client.upload_file.return_value = None

    result = upload_file(s3_client, "/path/to/some/file_name", "bucket_name")

    assert result

    s3_client.upload_file.assert_called_once_with(
        "/path/to/some/file_name", "bucket_name", "file_name"
    )


def test_upload_file_error():
    # Setup
    s3_client = Mock()
    s3_client.upload_file.side_effect = ClientError({"error": "error"}, "error")  # pyright: ignore[reportArgumentType]

    with pytest.raises(UploadError):
        upload_file(s3_client, "file_name", "bucket_name")

    s3_client.upload_file.assert_called_once_with("file_name", "bucket_name", "file_name")
