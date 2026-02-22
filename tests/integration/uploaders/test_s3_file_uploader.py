import logging
import os
import tempfile

import boto3
import pytest

from llm_agent.uploaders.s3_file_uploader import upload_file

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@pytest.mark.integration
def test_upload_succeeds():
    # Create test file with explicit, consistent name
    test_file_name = "test_upload_file.txt"
    temp_dir = tempfile.gettempdir()
    test_file_path = os.path.join(temp_dir, test_file_name)
    test_content = "test content"

    s3_client = boto3.client("s3")  # type: ignore[reportUnknownMemberType]

    try:
        # Write test content to file
        with open(test_file_path, "w") as f:
            f.write(test_content)

        # Upload file
        result = upload_file(s3_client, test_file_path, "test-geotiff-bucket")

        # Assert the uploaded file name matches our explicit name
        assert (
            result == "https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/test_upload_file.txt"
        )
        assert result.startswith("https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/")

        # Fetch file to assert its content is as expected
        response = s3_client.get_object(Bucket="test-geotiff-bucket", Key="test_upload_file.txt")
        stored_string = response["Body"].read().decode("utf-8")
        assert stored_string == "test content"

    finally:
        # Clean up temporary file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
