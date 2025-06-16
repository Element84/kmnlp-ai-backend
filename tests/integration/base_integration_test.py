import os

import pytest
from e84_geoai_common.llm.models import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from rasterio.session import boto3  # type: ignore[reportMissingStubs]

from llm_agent.utils import DEFAULT_MODEL_NAME


@pytest.mark.integration
class BaseIntegrationTest:
    @classmethod
    def setup_class(cls) -> None:
        # Expected usage:
        #
        # Localy, AWS_PROFILE should be set. If it is set, use that value to
        # create the `Session`.
        #
        # In CI, AWS_PROFILE should not be set. Instead, the AWS
        # key/secret/session token should be set using the result of an `aws
        # sts assume-role call`
        session: boto3.Session
        if profile_name := os.getenv("AWS_PROFILE"):
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )

        client = session.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

        cls.llm = BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS[DEFAULT_MODEL_NAME],
            client=client,
        )
