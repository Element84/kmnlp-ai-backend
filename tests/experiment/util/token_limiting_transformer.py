import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import floor
from time import time

from anyio import Event, Lock, Semaphore, sleep
from e84_geoai_common.llm.core import LLMAssistantMessage, LLMInferenceConfig, LLMMessage

from llm_agent.utils.prompt_as_service import (
    PromptService,
    SimplePromptServiceTransformer,
)

log = logging.getLogger(__name__)


@dataclass()
class Usage:
    tokens: int
    time: float


class TokenLimitingTransformer(SimplePromptServiceTransformer):
    """Rate limiter for token usage as a transformer.

    This class is not (yet) intended for use outside of test contexts. Be sure
    to evaluate it for quality if you would like to move it to `src/`.

    TokenLimiterTransformer limits the amount of tokens used in API requests (to Bedrock,
    in our case) in a given minute to 100,000 tokens. That limit is half of the
    default quota to Claude Sonnet 4 -- the only model this limiter currently
    gets used with.

    Use: Initialize a `TokenLimiterTransformer` instance, and compose it with a
    `PromptService` to create a new, token-limited `PromptService`. Provide
    that service as the argument wherever you'd want to a prompt-limited
    version.

    NOTE: The `TokenLimiterTransformer` does allow overuse: parallel requests
    can (and often do) go over the quota when they resolve because they we
    don't optimistically reserve quota. This is okay (even desired) for the
    test use case, as it keeps things running quickly, if a little bursty.
    """

    _TOKEN_LIMIT_PER_MINUTE = floor(125_000 / 4)

    _AVERAGE_TOKENS_PER_REQUEST = 8_000

    _usages: deque[Usage]
    _mutex: Lock

    _semaphore: Semaphore

    def __init__(self) -> None:
        self._usages = deque()
        self._mutex = Lock()

        # NOTE: This limit actually leads to a decent amount of overuse, but we
        # don't really mind because of the headroom we've saved ourselves. To
        # avoid overuse if we want to repurpose this limiter later, reduce the
        # parallelizaion limit, or pre-track estimated tokens.
        parallelization_limit = floor(
            self._TOKEN_LIMIT_PER_MINUTE / self._AVERAGE_TOKENS_PER_REQUEST
        )
        log.info("parallelization_limit is: %d", parallelization_limit)
        self._semaphore = Semaphore(parallelization_limit)

    async def __call__(
        self, args: tuple[Sequence[LLMMessage], LLMInferenceConfig], service: PromptService
    ) -> LLMAssistantMessage:
        # Make request and track usage, limiting concurrency to avoid
        # over-using quota before any responses can be tracked.
        log.debug("starting prompt_limited")
        async with self._semaphore:
            # Wait until lock can be acquired
            log.debug("Semaphore acquired.")
            quota_ready_event = Event()
            await self._notify_when_quota_available(quota_ready_event)
            log.debug("notify when quota available resolved")
            await quota_ready_event.wait()
            log.debug("quota available event received")

            try:
                log.debug("executing prompt")
                result = await service(args)
            except Exception:
                log.exception("Exception in prompt function.")
                raise

            log.debug("tracking usage")
            await self._track_usage(result)

        log.debug("Semaphore released.")

        return result

    async def _notify_when_quota_available(self, event: Event) -> None:
        log.debug("notify when quota available called")
        while await self._quota_remaining() <= 0:
            log.debug("in waiting loop")
            now = time()
            minimum_wait_time = self._usages[0].time + 60 - now
            log.debug("Sleeping for %.2f", minimum_wait_time)
            await sleep(minimum_wait_time)

        log.debug("setting quota available event")
        event.set()

    async def _quota_remaining(self) -> int:
        """Return the quota yet to be used in the last minute."""
        async with self._mutex:
            log.debug("acquired mutex for usage calculation")
            now = time()
            beginning_of_lookback_window = now - 60

            # Drop usages older than a minute.
            while self._usages:
                if self._usages[0].time < beginning_of_lookback_window:
                    self._usages.popleft()
                else:
                    break

            # Sum token use in _usages deque.
            quota_used = sum([usage.tokens for usage in self._usages])
            log.debug("Quota used: %d", quota_used)

        log.debug("released mutex for usage calculation")

        # Calculate quota remaining.
        quota_remaining = self._TOKEN_LIMIT_PER_MINUTE - quota_used
        log.debug("Quota remaining: %d", quota_remaining)
        return quota_remaining

    async def _track_usage(self, result: LLMAssistantMessage) -> None:
        """Track the usage for a given request."""
        # Add usage information to queue.
        log.debug("tracking usage")
        metadata = result.metadata
        assert metadata is not None  # Safe because we're in a test

        tokens_used = metadata.input_tokens + metadata.output_tokens
        log.debug("Tokens used: %d", tokens_used)

        # It's tempting to make getting the time and appending to the queue
        # atomic, but it isn't necessary because we don't strictly rely on the
        # queue being ordered. If the queue is out of order, the only effect
        # will be that the older result's usage gets tracked as longer as the
        # newer message in front of it is tracked.
        now = time()
        async with self._mutex:
            log.debug("acquired mutex to track usage")
            self._usages.append(Usage(tokens=tokens_used, time=now))

        log.debug("released mutex to track usage")
