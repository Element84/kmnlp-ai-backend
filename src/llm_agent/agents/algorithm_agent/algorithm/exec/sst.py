# ruff: noqa: D102

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import cartopy  # pyright: ignore[reportMissingTypeStubs]
import cartopy.crs  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import xarray as xr
from e84_geoai_common.tracing import timed_function
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from llm_agent.agents.algorithm_agent.algorithm.exec.animation import (
    AnimationSpec,
    Animator,
    animator_registry,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.base import ExecutionContext
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DataArrayShapeError,
    materialize_data_array_var,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot import (
    annotate_plot,
    is_lat_lon_data_array,
    make_plot,
    setup_geospatial_axes,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.plot_utils import (
    geospatial_cbar_kwargs,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.basic import (
    AnnotationSpec,
    GeospatialSpec,
)
from llm_agent.agents.algorithm_agent.algorithm.function_sets.plot.sst import (
    SSTAnomalyPlotSpec,
    SSTPlotSpec,
)

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh

white_cmap = ListedColormap(["white"])


@make_plot.register(SSTPlotSpec)
async def make_sst_plot(
    spec: SSTPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., Axes],
) -> "tuple[Axes, QuadMesh]":
    """Make a sea surface temperature plot with optional sea-ice mask."""
    data_array = await materialize_data_array_var(spec.sst_data_array, context)
    data_array = data_array.squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a 2D SST DataArray.",
        )

    if not is_lat_lon_data_array(data_array):
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected SST DataArray with 'lat' and 'lon' dimensions.",
        )

    # Trim NaN values from the edges of the data array
    data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    geospatial = GeospatialSpec(map_projection=spec.map_projection)
    ax = setup_geospatial_axes(data_array, geospatial, create_ax_func)  # type: ignore[reportUnknownMemberType]
    vmin, vmax = spec.colorbar_min_and_max
    im = data_array.plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
        ax=ax,
        add_colorbar=True,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
        cbar_kwargs=geospatial_cbar_kwargs(
            data_array, label="SST (°C)", extend="both", format="%.1f"
        ),
    )

    if spec.sea_ice_data_array is not None:
        ice = await materialize_data_array_var(spec.sea_ice_data_array, context)
        ice = ice.squeeze()
        # Trim NaN values from the edges of the data array
        ice = ice.dropna(dim="lat", how="all").dropna(dim="lon", how="all")
        overlay_sea_ice(ax, ice, spec.ice_threshold)

    annotate_plot(ax, AnnotationSpec(title=spec.title or "Sea Surface Temperature"))
    return ax, im


@make_plot.register(SSTAnomalyPlotSpec)
async def make_ssta_plot(
    spec: SSTAnomalyPlotSpec,
    context: ExecutionContext,
    create_ax_func: Callable[..., Axes],
) -> "tuple[Axes, QuadMesh]":
    """Make a sea surface temperature anomaly plot."""
    data_array = await materialize_data_array_var(spec.ssta_data_array, context)
    data_array = data_array.squeeze()

    if data_array.ndim != 2:  # noqa: PLR2004
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected a 2D SSTA DataArray.",
        )

    if not is_lat_lon_data_array(data_array):
        raise DataArrayShapeError.with_shape(
            func_name=spec.spec_name,
            dims=data_array.dims,  # pyright: ignore[reportArgumentType]
            shape=data_array.shape,
            msg_prefix="Expected SSTA DataArray with 'lat' and 'lon' dimensions.",
        )

    # Trim NaN values from the edges of the data array
    data_array = data_array.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    vmin, vmax = spec.colorbar_min_and_max
    if vmin is None and vmax is None:
        # Symmetric color limits so zero is at the center of the diverging colormap
        vmax = float(np.nanmax(np.abs(data_array.values)))  # type: ignore[reportUnknownMemberType]
        vmin = -vmax

    geospatial = GeospatialSpec(map_projection=spec.map_projection)
    ax = setup_geospatial_axes(data_array, geospatial, create_ax_func)  # type: ignore[reportUnknownMemberType]
    im = data_array.plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
        ax=ax,
        add_colorbar=True,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
        cbar_kwargs=geospatial_cbar_kwargs(
            data_array, label="SST Anomaly (°C)", extend="both", format="%.1f"
        ),
    )

    annotate_plot(ax, AnnotationSpec(title=spec.title or "Sea Surface Temperature Anomaly"))
    return ax, im


def overlay_sea_ice(ax: Axes, ice_data: xr.DataArray, threshold: float) -> "QuadMesh":
    """Overlay sea-ice as a white mask where concentration >= threshold."""
    ice_mask = ice_data if ice_data.dtype == bool else ice_data.where(ice_data >= threshold)  # type: ignore[reportUnknownMemberType]
    ice_mask = ice_mask.fillna(value=False)  # Treat NaNs as non-ice for plotting
    ice_raster: xr.DataArray = xr.where(ice_mask, 1, np.nan)  # pyright: ignore[reportUnknownMemberType] # raster where ice is 1 and non-ice is NaN

    return ice_raster.plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
        ax=ax,
        add_colorbar=False,
        cmap=white_cmap,
        vmin=0,
        vmax=1,
        transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
    )


class SSTPlotAnimator(Animator[SSTPlotSpec]):
    """Animator for sea surface temperature plots with optional sea-ice mask."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[SSTPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(
        self,
    ) -> None:
        data_array = await materialize_data_array_var(
            self.spec.plot_spec.sst_data_array, self.context
        )
        data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]
        if data_array.ndim != 3:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a 3D SST DataArray for animation.",
            )
        if not is_lat_lon_data_array(data_array):
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected SST DataArray with 'lat' and 'lon' dimensions.",
            )
        self.data_array = data_array
        self.frame_dim = self.spec.frame_dim
        self.frame_vals = data_array[self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        # Precompute ice raster frames as numpy arrays to avoid per-frame pcolormesh creation
        self._ice_raster_frames: np.ndarray | None = None
        if self.spec.plot_spec.sea_ice_data_array is not None:
            ice = await materialize_data_array_var(
                self.spec.plot_spec.sea_ice_data_array, self.context
            )
            threshold = self.spec.plot_spec.ice_threshold
            ice_arr = ice.squeeze().transpose(self.frame_dim, ...).load().values  # type: ignore[reportUnknownMemberType]
            ice_masks = ice_arr if ice.dtype == bool else ice_arr >= threshold
            # Compute rasterized ice mask for each frame: 1 where ice, NaN elsewhere
            self._ice_raster_frames = np.where(ice_masks, 1.0, np.nan)
            # Keep a reference DataArray for coordinate info (used once in init)
            self._ice_coords_ref = ice.isel({self.frame_dim: 0})

        vmin, vmax = self.spec.plot_spec.colorbar_min_and_max
        self._vmin = vmin or float(data_array.min())  # type: ignore[reportUnknownMemberType]
        self._vmax = vmax or float(data_array.max())  # type: ignore[reportUnknownMemberType]

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> "Iterable[QuadMesh]":
        """Create first frame."""
        geospatial = GeospatialSpec(map_projection=self.spec.plot_spec.map_projection)
        self.ax = setup_geospatial_axes(  # type: ignore[reportUnknownMemberType]
            self.data_array.isel({self.frame_dim: 0}), geospatial, self.create_ax_func
        )
        self.im = self.data_array.isel({self.frame_dim: 0}).plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
            ax=self.ax,
            add_colorbar=True,
            cmap="plasma",
            vmin=self._vmin,
            vmax=self._vmax,
            transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
            cbar_kwargs=geospatial_cbar_kwargs(
                self.data_array, label="SST (°C)", extend="both", format="%.1f"
            ),
        )
        self._ice_overlay: QuadMesh | None = None
        if self._ice_raster_frames is not None:
            # Create the ice overlay pcolormesh once; update with set_array per frame
            ice_raster = self._ice_coords_ref.copy(data=self._ice_raster_frames[0])  # type: ignore[reportUnknownMemberType]
            self._ice_overlay = ice_raster.plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
                ax=self.ax,
                add_colorbar=False,
                cmap=white_cmap,
                vmin=0,
                vmax=1,
                transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
            )
        annotate_plot(
            self.ax,
            AnnotationSpec(title=self.spec.plot_spec.title or "Sea Surface Temperature"),
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        artists: list[QuadMesh] = [self.im]
        if self._ice_overlay is not None:
            artists.append(self._ice_overlay)
        return artists

    @timed_function
    def update(self, frame: int) -> "Iterable[QuadMesh]":
        """Create i-th frame."""
        self.im.set_array(self.data_array.isel({self.frame_dim: frame}).values.ravel())  # type: ignore[reportUnknownMemberType]

        if self._ice_raster_frames is not None and self._ice_overlay is not None:
            self._ice_overlay.set_array(self._ice_raster_frames[frame])  # type: ignore[reportUnknownMemberType]

        annotate_plot(
            self.ax,
            AnnotationSpec(title=self.spec.plot_spec.title or "Sea Surface Temperature"),
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        artists: list[QuadMesh] = [self.im]
        if self._ice_overlay is not None:
            artists.append(self._ice_overlay)
        return artists


class SSTAnomalyPlotAnimator(Animator[SSTAnomalyPlotSpec]):
    """Animator for sea surface temperature anomaly plots."""

    def __init__(  # noqa: D107
        self,
        spec: AnimationSpec[SSTAnomalyPlotSpec],
        context: ExecutionContext,
        create_ax_func: Callable[..., Axes],
    ) -> None:
        super().__init__(spec, context, create_ax_func)

    async def init_data(self) -> None:
        data_array = await materialize_data_array_var(
            self.spec.plot_spec.ssta_data_array, self.context
        )
        data_array = data_array.squeeze().load()  # type: ignore[reportUnknownMemberType]
        if data_array.ndim != 3:  # noqa: PLR2004
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected a 3D SSTA DataArray for animation.",
            )
        if not is_lat_lon_data_array(data_array):
            raise DataArrayShapeError.with_shape(
                func_name="MakeAnimation",
                dims=data_array.dims,  # pyright: ignore[reportArgumentType]
                shape=data_array.shape,
                msg_prefix="Expected SSTA DataArray with 'lat' and 'lon' dimensions.",
            )
        self.data_array = data_array
        self.frame_dim = self.spec.frame_dim
        self.frame_vals = data_array[self.frame_dim].values  # type: ignore[reportUnknownMemberType]

        vmin, vmax = self.spec.plot_spec.colorbar_min_and_max
        if vmin is None and vmax is None:
            # Symmetric color limits so zero is at the center of the diverging colormap
            vmax = float(np.nanmax(np.abs(data_array.values)))  # type: ignore[reportUnknownMemberType]
            vmin = -vmax

        self._vmax = vmax
        self._vmin = vmin

    @property
    def nframes(self) -> int:
        """Number of frames in the animation."""
        return len(self.frame_vals)

    def init(self) -> "Iterable[QuadMesh]":
        """Create first frame."""
        geospatial = GeospatialSpec(map_projection=self.spec.plot_spec.map_projection)
        self.ax = setup_geospatial_axes(  # type: ignore[reportUnknownMemberType]
            self.data_array.isel({self.frame_dim: 0}), geospatial, self.create_ax_func
        )
        self.im = self.data_array.isel({self.frame_dim: 0}).plot.pcolormesh(  # type: ignore[reportUnknownMemberType]
            ax=self.ax,
            add_colorbar=True,
            cmap="RdBu_r",
            vmin=self._vmin,
            vmax=self._vmax,
            transform=cartopy.crs.PlateCarree(),  # type: ignore[reportUnknownMemberType]
            cbar_kwargs=geospatial_cbar_kwargs(
                self.data_array, label="SST Anomaly (°C)", extend="both", format="%.1f"
            ),
        )
        annotate_plot(
            self.ax,
            AnnotationSpec(title=self.spec.plot_spec.title or "Sea Surface Temperature Anomaly"),
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[0]}",
        )
        return [self.im]

    @timed_function
    def update(self, frame: int) -> "Iterable[QuadMesh]":
        """Create i-th frame."""
        self.im.set_array(self.data_array.isel({self.frame_dim: frame}).values.ravel())  # type: ignore[reportUnknownMemberType]
        annotate_plot(
            self.ax,
            AnnotationSpec(title=self.spec.plot_spec.title or "Sea Surface Temperature Anomaly"),
            title_postfix=f"\n{self.frame_dim}: {self.frame_vals[frame]}",
        )
        return [self.im]


animator_registry.register(SSTPlotSpec, SSTPlotAnimator)
animator_registry.register(SSTAnomalyPlotSpec, SSTAnomalyPlotAnimator)
