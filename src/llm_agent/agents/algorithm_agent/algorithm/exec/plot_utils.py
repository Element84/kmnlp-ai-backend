"""Low-level, spec-agnostic plot adjustment and formatting helpers.

These are pure figure/axes tweaks and label formatting utilities used by the plotting exec code;
they intentionally know nothing about the algorithm's plot specs.
"""

import logging
import re
import textwrap
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def deoverlap_x_tick_labels(fig: Figure) -> None:
    """Rotate x tick labels on any axes whose labels horizontally overlap, to keep them readable.

    Only axes with actually-overlapping, visible labels are touched (measured after a layout pass),
    so plots that fit stay horizontal, axes whose x labels are hidden (e.g. inner subplots of a grid
    with a shared x-axis) are left alone, and cartopy maps — whose lon/lat labels are gridliner
    artists rather than standard x tick labels — are unaffected.
    """
    try:
        fig.draw_without_rendering()  # type: ignore[reportUnknownMemberType]
        renderer = fig.canvas.get_renderer()  # type: ignore[reportAttributeAccessIssue]
        for ax in fig.axes:
            if not ax.get_visible() or not ax.xaxis.get_visible():  # type: ignore[reportUnknownMemberType]
                continue
            labels = [t for t in ax.get_xticklabels() if t.get_text() and t.get_visible()]  # type: ignore[reportUnknownMemberType]
            if len(labels) < 2:  # noqa: PLR2004
                continue
            boxes = sorted(
                (t.get_window_extent(renderer) for t in labels),  # type: ignore[reportUnknownMemberType]
                key=lambda b: b.x0,
            )
            overlaps = any(boxes[i].x1 + 1.0 > boxes[i + 1].x0 for i in range(len(boxes) - 1))
            if overlaps:
                for t in labels:
                    t.set_rotation(30)  # type: ignore[reportUnknownMemberType]
                    t.set_horizontalalignment("right")  # type: ignore[reportUnknownMemberType]
                    t.set_rotation_mode("anchor")  # type: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001
        # Label de-overlapping is purely cosmetic; never fail a plot over it.
        log.debug("Could not de-overlap x tick labels.", exc_info=True)


def wrap_titles_to_axes_width(fig: Figure) -> None:
    """Re-wrap each axes title to the character budget that fits its post-layout width.

    Subplot width is only known after layout runs, so a fixed character budget over- or under-wraps
    depending on plot size; measuring the rendered title and axes widths adapts the wrap (and
    un-wraps titles that comfortably fit) to the actual axes.
    """
    try:
        fig.draw_without_rendering()  # type: ignore[reportUnknownMemberType]
        renderer = fig.canvas.get_renderer()  # type: ignore[reportAttributeAccessIssue]
        for ax in fig.axes:
            title = ax.get_title()  # type: ignore[reportUnknownMemberType]
            if not title:
                continue
            original = " ".join(title.split("\n"))
            font_prop = ax.title.get_fontproperties()  # type: ignore[reportUnknownMemberType]
            text_w = renderer.get_text_width_height_descent(original, font_prop, ismath=False)[0]  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            axes_w = ax.get_window_extent(renderer).width  # type: ignore[reportUnknownMemberType]
            if text_w <= 0:
                continue
            # chars-per-line = current char count scaled by how the axes width compares to the text
            max_chars = max(8, int(len(original) * axes_w * 0.95 / text_w))  # type: ignore[reportUnknownArgumentType]
            ax.set_title(  # type: ignore[reportUnknownMemberType]
                "\n".join(textwrap.wrap(original, max_chars)), fontsize=ax.title.get_fontsize()
            )
    except Exception:  # noqa: BLE001
        # Title wrapping is purely cosmetic; never fail a plot over it.
        log.debug("Could not wrap titles to axes width.", exc_info=True)


def fit_figure_height(fig: Figure) -> None:
    """Shrink figure height to remove the vertical slack a suptitle leaves above letterboxed axes.

    Fixed-aspect (cartopy / image) axes are centered in tall grid cells, leaving a big gap under the
    suptitle that a tight bbox can't trim and constrained-layout won't remove; reducing the figure
    height collapses that slack while keeping colorbars attached to their maps.
    """
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:  # only grids add a suptitle; single plots are handled by the tight bbox
        return
    target_gap = 0.10
    try:
        for _ in range(5):
            fig.draw_without_rendering()  # type: ignore[reportUnknownMemberType]
            renderer = fig.canvas.get_renderer()  # type: ignore[reportAttributeAccessIssue]
            fig_h_px = fig.bbox.height
            content_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]  # type: ignore[reportUnknownMemberType]
            if not content_axes:
                return
            top = max(ax.get_window_extent(renderer).y1 for ax in content_axes) / fig_h_px  # type: ignore[reportUnknownMemberType]
            gap = suptitle.get_window_extent(renderer).y0 / fig_h_px - top
            if gap <= target_gap:
                break
            w_in, h_in = fig.get_size_inches()
            fig.set_size_inches(w_in, max(1.5, h_in * (1 - (gap - target_gap))))  # type: ignore[reportUnknownArgumentType]
    except Exception:  # noqa: BLE001
        # Height fitting is purely cosmetic; never fail a plot over it.
        log.debug("Could not fit figure height to content.", exc_info=True)


_SUPERSCRIPT_TO_ASCII = {
    ord("⁰"): "0",
    ord("¹"): "1",
    ord("²"): "2",
    ord("³"): "3",
    ord("⁴"): "4",
    ord("⁵"): "5",
    ord("⁶"): "6",
    ord("⁷"): "7",
    ord("⁸"): "8",
    ord("⁹"): "9",
    ord("⁺"): "+",
    ord("⁻"): "-",
    ord("⁼"): "=",
    ord("⁽"): "(",
    ord("⁾"): ")",
    ord("ⁿ"): "n",
    ord("ⁱ"): "i",
}
_SUBSCRIPT_TO_ASCII = {
    ord("₀"): "0",
    ord("₁"): "1",
    ord("₂"): "2",
    ord("₃"): "3",
    ord("₄"): "4",
    ord("₅"): "5",
    ord("₆"): "6",
    ord("₇"): "7",
    ord("₈"): "8",
    ord("₉"): "9",
    ord("₊"): "+",
    ord("₋"): "-",
    ord("₌"): "=",
    ord("₍"): "(",
    ord("₎"): ")",
}
_SUPERSCRIPT_RE = re.compile(r"[\u00b2\u00b3\u00b9\u2070\u2071\u2074-\u207f]+")
_SUBSCRIPT_RE = re.compile(r"[\u2080-\u208e]+")


def sanitize_display_text(text: str) -> str:
    """Rewrite unicode super/subscript runs as mathtext so missing-glyph boxes don't appear.

    The default font lacks some superscript characters (e.g. U+207B, superscript minus), which
    render as empty boxes; converting 'm⁻²' to 'm$^{-2}$' renders it correctly via mathtext.
    """
    if "$" in text:
        return text
    text = _SUPERSCRIPT_RE.sub(
        lambda m: "$^{" + m.group().translate(_SUPERSCRIPT_TO_ASCII) + "}$", text
    )
    return _SUBSCRIPT_RE.sub(
        lambda m: "$_{" + m.group().translate(_SUBSCRIPT_TO_ASCII) + "}$", text
    )


# A lat/lon map at least this much wider than tall (lon-extent / lat-extent) gets a horizontal
# colorbar, so the bar tracks the map's long edge instead of towering over a short, wide map.
_WIDE_MAP_ASPECT_THRESHOLD = 1.5


def geospatial_cbar_kwargs(
    data_array: xr.DataArray, *, label: str | None = None, **extra: object
) -> dict[str, Any]:
    """Build colorbar kwargs whose orientation adapts to a lat/lon map's aspect ratio.

    Wide, short maps get a horizontal colorbar underneath (its length then tracks the long edge);
    otherwise a vertical colorbar alongside. Extra keyword args (e.g. ``extend``, ``format``) are
    merged in. Works both as xarray ``cbar_kwargs`` and spread into ``Figure.colorbar``.
    """
    lat_extent = float(data_array["lat"].max() - data_array["lat"].min())
    lon_extent = float(data_array["lon"].max() - data_array["lon"].min())
    horizontal = lat_extent > 0 and (lon_extent / lat_extent) >= _WIDE_MAP_ASPECT_THRESHOLD
    kwargs: dict[str, Any] = {
        "orientation": "horizontal" if horizontal else "vertical",
        "shrink": 0.8,
        "pad": 0.05,
        "aspect": 40,
    }
    if label is not None:
        kwargs["label"] = sanitize_display_text(label)
    kwargs.update(extra)
    return kwargs


def timedelta_to_numeric(values: "np.ndarray[Any, Any]") -> "tuple[np.ndarray[Any, Any], str]":
    """Convert a timedelta64 array to floats in hours or days (whichever reads better).

    Forecast axes such as 'lead_time' are timedelta64; matplotlib cannot plot them against float
    data, and their raw repr is unreadable. Returns the numeric values plus a unit label.
    """
    hours = values / np.timedelta64(1, "h")  # type: ignore[reportUnknownVariableType]
    max_hours = float(np.nanmax(np.abs(hours))) if hours.size else 0.0
    if max_hours >= 48:  # noqa: PLR2004
        return values / np.timedelta64(1, "D"), "days"  # type: ignore[reportUnknownVariableType]
    return hours, "hours"


def format_coordinate_label(value: Any) -> str:  # noqa: ANN401
    """Human-readable label for a single coordinate value (e.g. an animation frame title).

    Renders a timedelta64 lead time as hours or days, and a datetime64 as an ISO timestamp,
    instead of their raw numpy reprs.
    """
    if isinstance(value, np.timedelta64):
        hours = value / np.timedelta64(1, "h")
        if abs(hours) >= 48:  # noqa: PLR2004
            return f"{value / np.timedelta64(1, 'D'):g} days"
        return f"{hours:g} h"
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="h"))  # type: ignore[reportUnknownArgumentType]
    return str(value)
