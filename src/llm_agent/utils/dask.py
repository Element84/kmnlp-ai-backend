import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, NewType

import humanize.filesize
import xarray as xr
from dask.base import is_dask_collection  # type: ignore[reportUnknownVariableType]
from dask.distributed import Future, get_client  # type: ignore[reportUnknownVariableType]
from distributed.diagnostics.progressbar import ProgressBar  # type: ignore[reportMissingTypeStubs]
from langfuse import observe  # type: ignore[reportUnknownVariableType]

if TYPE_CHECKING:
    from dask.distributed import Client

log = logging.getLogger(__name__)

MAX_ALLOWED_COMPUTE_SIZE = 5 * 10**9  # 5 GB

MaterializedDataArray = NewType("MaterializedDataArray", xr.DataArray)

# Attribute key holding the fraction (0-1) of a DataArray's lat/lon cells that remain after a
# spatial mask (e.g. a polygon filter). Lets the size check count only the meaningful data rather
# than the full, mostly-NaN bounding box.
SPATIAL_VALID_FRACTION_ATTR = "spatial_valid_fraction"

# An xr.DataArray that has been spatially masked in place (same shape, NaN outside the mask) and
# carries SPATIAL_VALID_FRACTION_ATTR in its .attrs.
SpatiallyMaskedDataArray = NewType("SpatiallyMaskedDataArray", xr.DataArray)


class DaskComputeError(Exception):
    """Failure to compute values from a dask DataArray."""


class DataSizeTooLargeError(Exception):
    """DataArray size exceeds allowed limit."""

    def __init__(self, nbytes: int, size_limit: int = MAX_ALLOWED_COMPUTE_SIZE) -> None:  # noqa: D107
        size_str = humanize.filesize.naturalsize(nbytes)
        size_limit_str = humanize.filesize.naturalsize(size_limit)
        super().__init__(
            f"Query requires processing {size_str} of data, "
            f"which exceeds the allowed limit of {size_limit_str}."
        )


def validate_data_array_size_within_limit(
    data_array: xr.DataArray, size_limit: int = MAX_ALLOWED_COMPUTE_SIZE
) -> None:
    """Check if DataArray size is within allowed limit.

    If the array carries a spatial valid-fraction attr (set by a spatial mask such as a polygon
    filter), only that fraction of its bytes counts toward the limit, so a small region carved from
    a large grid is not rejected for the size of its mostly-NaN bounding box.
    """
    valid_fraction = float(data_array.attrs.get(SPATIAL_VALID_FRACTION_ATTR, 1.0))
    effective_nbytes = round(data_array.nbytes * valid_fraction)
    if effective_nbytes > size_limit:
        raise DataSizeTooLargeError(effective_nbytes, size_limit)
    log.info("Processing DataArray of size %s", humanize.filesize.naturalsize(effective_nbytes))


class _AsyncProgressLogger(ProgressBar):  # type: ignore[reportMissingTypeStubs]
    """Async progress tracker that logs completion via the scheduler.

    Uses the same scheduler-side Progress plugin as the dask dashboard,
    which traverses the full task dependency graph for accurate counts.
    """

    def __init__(
        self,
        *,
        keys: Any,  # noqa: ANN401
        scheduler: Any = None,  # noqa: ANN401
        interval: str = "100ms",
        complete: bool = True,
        callback: Callable[[int, int], None] | None = None,  # type: ignore[reportUnknownVariableType]
    ) -> None:
        super().__init__(keys, scheduler=scheduler, interval=interval, complete=complete)  # pyright: ignore[reportUnknownMemberType]
        self.callback = callback
        self.last_total = 0

    def _draw_bar(self, remaining: int, all: int, **kwargs: object) -> None:  # type: ignore[reportMissingParameterType]  # noqa: A002, ARG002
        self.last_total = all
        log.info("Dask progress: %d/%d tasks complete", all - remaining, all)
        if self.callback:
            self.callback(all - remaining, all)

    def _draw_stop(self, remaining: int = 0, all: int = 0, **kwargs: object) -> None:  # type: ignore[reportMissingParameterType]  # noqa: A002, ARG002
        self.last_total = all or self.last_total
        log.info(
            "Dask progress: %d/%d tasks complete", self.last_total - remaining, self.last_total
        )
        if self.callback:
            self.callback(self.last_total - remaining, self.last_total)


@asynccontextmanager
async def _track_progress(
    future: Future,
    client: "Client",
    *,
    callback: Callable[[int, int], None] | None = None,
    poll_interval_sec: float = 5.0,
) -> AsyncGenerator[_AsyncProgressLogger]:
    """Context manager that tracks dask computation progress via a scheduler plugin.

    Starts an async progress listener on entry, cancels and cleans up on exit.
    After exit, ``tracker.last_total`` contains the total task count.
    """
    tracker = _AsyncProgressLogger(
        keys=[future],
        scheduler=client.scheduler.address,  # type: ignore[reportUnknownMemberType]
        interval=f"{poll_interval_sec}s",
        complete=True,
        callback=callback,
    )
    progress_task = asyncio.create_task(tracker.listen())  # type: ignore[reportUnknownMemberType]
    try:
        yield tracker
    finally:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task
        with contextlib.suppress(Exception):
            if hasattr(tracker, "comm"):
                await tracker.comm.close()  # type: ignore[reportUnknownMemberType]


@observe(as_type="chain")
async def compute_dask_array(
    data_array: xr.DataArray,
    *,
    size_limit: int = MAX_ALLOWED_COMPUTE_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_poll_interval_sec: float = 5.0,
) -> MaterializedDataArray:
    """Convert to non-lazy array using async dask client.

    This function supports cancellation - if the computation is cancelled via
    asyncio.CancelledError, the Dask cluster computation will also be cancelled
    to prevent orphaned tasks.

    - If data_array is not a dask array, return it as is.
    - If data_array is already computed, return it as is.
    - If there is no active dask client, call data_array.compute() directly.
    - If dask client is asynchronous, use await to run computation.
    - If computation does not result in a DataArray, raise an error.

    Raises:
        DaskComputeError: If computation fails for any reason.
        RuntimeError: If dask client is not async (asynchronous=False).
    """
    if not is_dask_collection(data_array) or data_array.chunks is None:
        # not a dask array
        return MaterializedDataArray(data_array)

    validate_data_array_size_within_limit(data_array, size_limit=size_limit)

    try:
        client = get_client()
    except ValueError:
        # no active Dask client, compute directly
        return MaterializedDataArray(data_array.compute())  # type: ignore[reportUnknownMemberType]

    # Validate that client is async - required for cancellation support
    if not client.asynchronous:
        msg = (
            "Dask client must be initialized with asynchronous=True for cancellable execution. "
            "Example: Client(scheduler_address, asynchronous=True)"
        )
        raise RuntimeError(msg)

    compute_output = _compute(client, data_array)

    if isinstance(compute_output, xr.DataArray):
        # already computed synchronously
        return MaterializedDataArray(compute_output)

    async with _track_progress(
        compute_output,
        client,
        callback=progress_callback,
        poll_interval_sec=progress_poll_interval_sec,
    ) as tracker:
        try:
            result = await compute_output.result()  # type: ignore[reportUnknownArgumentType]
            total = tracker.last_total
            log.info("Dask computation finished: %d/%d tasks complete", total, total)
            if progress_callback and total:
                progress_callback(total, total)
        except asyncio.CancelledError:
            log.info("Computation cancelled, cleaning up Dask future")
            await _cancel_future(compute_output)
            raise
        except Exception as e:
            log.exception("Dask computation errored")
            msg = f"Error encountered when awaiting .compute() output: {e!r}"
            raise DaskComputeError(msg) from e

    if not isinstance(result, xr.DataArray):
        msg = f"await returned unexpected type {type(result)}."  # type: ignore[reportUnknownArgumentType]
        raise DaskComputeError(msg)

    return MaterializedDataArray(result)


def _compute(client: "Client", data_array: xr.DataArray) -> MaterializedDataArray | Future:
    # client.compute()'s docs say that it returns a Future or list of Futures,
    # but its inferred return type is:
    #     Unknown |
    #     Generator[Generator[Unknown, None, None] |
    #     CoroutineType[Any, Any, Unknown] | Unknown, None, None] |
    #     CoroutineType[Any, Any, Unknown] |
    #     list[Unknown]  # noqa: ERA001
    # Hence the extra type checks below.
    compute_output = client.compute(data_array)  # type: ignore[reportUnknownMemberType]

    if isinstance(compute_output, xr.DataArray):
        return MaterializedDataArray(compute_output)

    if not isinstance(compute_output, Future):
        msg = (
            "client.compute() returned unexpected type "
            f"{type(compute_output)}."  # type: ignore[reportUnknownArgumentType]
        )
        raise DaskComputeError(msg)

    return compute_output


async def _cancel_future(future: Future) -> None:
    try:
        cancel_result = future.cancel()  # type: ignore[reportUnknownMemberType]
        # cancel() may return a coroutine if client is async
        if inspect.iscoroutine(cancel_result):  # type: ignore[reportUnknownMemberType]
            await cancel_result  # type: ignore[reportUnknownArgumentType]
    except Exception:
        # Handle exception cancelling Dask computation by logging.
        log.exception("Failed to cancel Dask future during cleanup")
