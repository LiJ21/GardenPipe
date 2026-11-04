from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from html import escape
from os import PathLike
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

POLARS_ENGINES = {"auto", "streaming", "gpu"}
SERIES_KINDS = {"level", "line", "event"}
_X_HOVER_COL = "__viewer_x_hover"
_X_NS_COL = "__viewer_x_ns"
_X_PLOT_COL = "__viewer_x_plot"
DATETIME_X_MODES = {"datetime", "relative_ns"}


@dataclass
class Series:
    """A plotted y series within a Layer.

    If ``y`` is a Polars expression, ``name`` is required and is used as the
    temporary projected column name. DataSource inputs use ``frame(select=False)``,
    so fitted transforms are applied; pass a raw LazyFrame or transform-free
    DataSource to inspect raw data.
    """

    y: str | pl.Expr
    kind: str = "level"
    name: str | None = None
    secondary_y: bool = False
    color: str | None = None
    line_width: float = 1.4
    line_shape: str = "hv"
    point_symbol: str = "cross"
    point_size: int = 4
    marker_symbol: str = "circle"
    marker_color: str | None = None
    marker_size: int = 8
    label_cols: Sequence[str] = ()
    value_label: str | None = None
    hover: bool = True
    max_points: int | None = None
    resample: str | None = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in SERIES_KINDS:
            raise ValueError(f"kind must be one of: {sorted(SERIES_KINDS)}")
        self.label_cols = tuple(self.label_cols)
        if isinstance(self.y, pl.Expr) and not self.name:
            raise ValueError("Series.name is required when y is a Polars expression")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be non-negative or None")

    @property
    def trace_name(self) -> str:
        if self.name is not None:
            return self.name
        if isinstance(self.y, str):
            return self.y
        raise ValueError("Series.name is required when y is a Polars expression")

    @property
    def y_col(self) -> str:
        if isinstance(self.y, str):
            return self.y
        return self.trace_name


@dataclass
class Layer:
    source: Any
    series: list[Series]
    x: str = "time"
    filters: tuple[pl.Expr, ...] = ()
    label_cols: Sequence[str] = ()
    sort_x: bool = True

    def __post_init__(self) -> None:
        self.series = list(self.series)
        self.filters = tuple(self.filters)
        self.label_cols = tuple(self.label_cols)
        if not self.series:
            raise ValueError("Layer.series must contain at least one Series")


@dataclass
class Panel:
    layers: list[Layer]
    title: str | None = None
    height: float = 1.0

    def __post_init__(self) -> None:
        self.layers = list(self.layers)
        if not self.layers:
            raise ValueError("Panel.layers must contain at least one Layer")
        if self.height <= 0:
            raise ValueError("Panel.height must be positive")

    @classmethod
    def from_source(
        cls,
        source: Any,
        series: list[Series],
        *,
        x: str = "time",
        filters: tuple[pl.Expr, ...] = (),
        label_cols: Sequence[str] = (),
        sort_x: bool = True,
        title: str | None = None,
        height: float = 1.0,
    ) -> "Panel":
        return cls(
            layers=[
                Layer(
                    source=source,
                    series=series,
                    x=x,
                    filters=filters,
                    label_cols=label_cols,
                    sort_x=sort_x,
                )
            ],
            title=title,
            height=height,
        )


@dataclass
class _LayerData:
    panel_idx: int
    layer: Layer
    df: pl.DataFrame
    x_dtype: Any
    is_datetime_x: bool
    x_hover_col: str | None = None
    x_ns_col: str | None = None
    x_plot_col: str | None = None
    x_origin_ns: int | None = None


@dataclass
class _LiveSeries:
    name: str
    kind: str
    x: list[Any]
    y: list[Any]


@dataclass
class DataViewer:
    panels: list[Panel]
    shared_x: bool = True
    timezone: str | None = None
    spikemode: str = "across"
    spikesnap: str = "cursor"
    hovermode: str = "x"
    x_label: str | None = None
    datetime_x_mode: str = "datetime"
    live_readout: bool = False
    max_points: int | None = 200_000
    resample: str = "auto"
    vertical_spacing: float = 0.03
    height: int | None = None
    template: str = "plotly_white"
    polars_engine: str = "streaming"

    def __post_init__(self) -> None:
        self.panels = list(self.panels)
        if not self.panels:
            raise ValueError("DataViewer.panels must contain at least one Panel")
        self.polars_engine = self.polars_engine.lower()
        if self.polars_engine not in POLARS_ENGINES:
            raise ValueError(f"polars_engine must be one of: {sorted(POLARS_ENGINES)}")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max_points must be non-negative or None")
        if self.datetime_x_mode not in DATETIME_X_MODES:
            raise ValueError(f"datetime_x_mode must be one of: {sorted(DATETIME_X_MODES)}")
        if self.timezone is not None:
            _zoneinfo(self.timezone)

    def figure(
        self, *, start: Any = None, end: Any = None, live: bool = False
    ) -> "Any":
        """Build a Plotly Figure, or FigureWidget when ``live=True``."""

        go, make_subplots = _plotly()
        collected = self._collect_layers(start=start, end=end)
        fig, live_series = self._build_figure(go, make_subplots, collected)
        if not live:
            return fig
        widget = go.FigureWidget(fig)
        if self.live_readout:
            _attach_live_hover(widget, live_series)
        return widget

    def show(self, *, start: Any = None, end: Any = None, live: bool = False, **kw) -> None:
        fig = self.figure(start=start, end=end, live=live)
        fig.show(**kw)

    def to_html(self, path: str | PathLike[str], *, start: Any = None, end: Any = None) -> None:
        fig = self.figure(start=start, end=end, live=False)
        fig.write_html(str(path))

    def _collect_layers(self, *, start: Any, end: Any) -> list[_LayerData]:
        out = []
        for panel_idx, panel in enumerate(self.panels):
            for layer in panel.layers:
                lf = _source_to_lazy(layer.source)
                schema = lf.collect_schema()
                x_dtype = _schema_dtype(schema, layer.x)
                is_datetime_x = _is_datetime_dtype(x_dtype)

                lf = self._filter_layer(lf, layer, x_dtype, start, end)
                lf = _add_expression_series(lf, layer)
                x_hover_col = _X_HOVER_COL if is_datetime_x else None
                x_ns_col = _X_NS_COL if is_datetime_x else None
                if x_hover_col is not None:
                    lf = lf.with_columns(
                        _x_hover_expr(layer.x, x_dtype, self.timezone).alias(x_hover_col),
                        _x_ns_expr(layer.x, x_dtype).alias(x_ns_col),
                    )

                needed = _needed_columns(layer)
                if x_hover_col is not None:
                    needed.append(x_hover_col)
                if x_ns_col is not None:
                    needed.append(x_ns_col)
                lf = lf.select(needed)
                if self.timezone is not None and is_datetime_x:
                    lf = lf.with_columns(_display_time_expr(layer.x, x_dtype, self.timezone))

                df = lf.collect(engine=self.polars_engine)
                if layer.sort_x and df.height:
                    df = df.sort(layer.x)
                out.append(
                    _LayerData(
                        panel_idx=panel_idx,
                        layer=layer,
                        df=df,
                        x_dtype=x_dtype,
                        is_datetime_x=is_datetime_x,
                        x_hover_col=x_hover_col,
                        x_ns_col=x_ns_col,
                    )
                )
        if self.datetime_x_mode == "relative_ns":
            _apply_relative_ns_x(out)
        return out

    def _filter_layer(
        self,
        lf: pl.LazyFrame,
        layer: Layer,
        x_dtype: Any,
        start: Any,
        end: Any,
    ) -> pl.LazyFrame:
        predicates = list(layer.filters)
        start_value = _coerce_bound(start, x_dtype, self.timezone)
        end_value = _coerce_bound(end, x_dtype, self.timezone)
        if start_value is not None:
            predicates.append(pl.col(layer.x) >= start_value)
        if end_value is not None:
            predicates.append(pl.col(layer.x) <= end_value)
        if not predicates:
            return lf
        mask = predicates[0]
        for predicate in predicates[1:]:
            mask = mask & predicate
        return lf.filter(mask)

    def _build_figure(
        self,
        go: Any,
        make_subplots: Any,
        collected: list[_LayerData],
    ) -> tuple[Any, list[_LiveSeries]]:
        row_heights = _normalize([panel.height for panel in self.panels])
        specs = [
            [
                {
                    "secondary_y": any(
                        series.secondary_y
                        for layer in panel.layers
                        for series in layer.series
                    )
                }
            ]
            for panel in self.panels
        ]
        titles = [panel.title or "" for panel in self.panels]

        fig = make_subplots(
            rows=len(self.panels),
            cols=1,
            shared_xaxes=self.shared_x,
            vertical_spacing=self.vertical_spacing,
            row_heights=row_heights,
            subplot_titles=titles if any(titles) else None,
            specs=specs,
        )

        live_series = []
        row_has_secondary = [row[0]["secondary_y"] for row in specs]
        for item in collected:
            row = item.panel_idx + 1
            for series in item.layer.series:
                trace, live = self._trace_for_series(go, item, series)
                if row_has_secondary[item.panel_idx]:
                    fig.add_trace(
                        trace,
                        row=row,
                        col=1,
                        secondary_y=series.secondary_y,
                    )
                else:
                    fig.add_trace(trace, row=row, col=1)
                live_series.append(live)

        fig.update_xaxes(
            showspikes=True,
            spikemode=self.spikemode,
            spikesnap=self.spikesnap,
            spikethickness=1,
            spikedash="dot",
        )
        if self.datetime_x_mode == "relative_ns":
            _apply_relative_ns_axis(fig, collected, self.timezone)
        layout = {
            "hovermode": self.hovermode,
            "template": self.template,
        }
        if self.height is not None:
            layout["height"] = self.height
        fig.update_layout(**layout)
        return fig, live_series

    def _trace_for_series(
        self, go: Any, item: _LayerData, series: Series
    ) -> tuple[Any, _LiveSeries]:
        df = item.df
        label_cols = _ordered_unique([*item.layer.label_cols, *series.label_cols])
        x_col = item.x_plot_col or item.layer.x
        x_values = df.get_column(x_col).to_list() if df.height else []
        y_values = df.get_column(series.y_col).to_list() if df.height else []
        indices = _resample_indices(
            series.kind,
            y_values,
            _series_max_points(series, self.max_points),
            series.resample if series.resample is not None else self.resample,
        )
        x_values = _take(x_values, indices)
        y_values = _take(y_values, indices)
        x_hover_cols = [item.x_hover_col] if item.x_hover_col is not None else []
        custom_cols = [*x_hover_cols, *label_cols]
        customdata = _customdata(df, custom_cols, indices) if series.hover and custom_cols else None
        hovertemplate = (
            _hovertemplate(
                series.trace_name,
                label_cols,
                item.is_datetime_x,
                series.value_label,
                self.x_label or item.layer.x,
                x_customdata_index=0 if x_hover_cols else None,
                label_customdata_offset=len(x_hover_cols),
            )
            if series.hover
            else None
        )

        common = {
            "x": x_values,
            "y": y_values,
            "name": series.trace_name,
            "customdata": customdata,
            "hovertemplate": hovertemplate,
            "hoverinfo": None if series.hover else "skip",
        }
        if series.kind == "level":
            trace = go.Scatter(
                mode="lines+markers" if series.point_size > 0 else "lines",
                line=_drop_none(
                    {
                        "shape": series.line_shape,
                        "color": series.color,
                        "width": series.line_width,
                    }
                ),
                marker=_drop_none(
                    {
                        "symbol": series.point_symbol,
                        "size": series.point_size,
                        "color": series.color,
                    }
                ),
                **_drop_none(common),
            )
        elif series.kind == "line":
            trace = go.Scatter(
                mode="lines",
                line=_drop_none(
                    {
                        "shape": "linear",
                        "color": series.color,
                        "width": series.line_width,
                    }
                ),
                **_drop_none(common),
            )
        else:
            trace = go.Scatter(
                mode="markers",
                marker=_drop_none(
                    {
                        "symbol": series.marker_symbol,
                        "color": series.marker_color,
                        "size": series.marker_size,
                    }
                ),
                **_drop_none(common),
            )
        live = _LiveSeries(series.trace_name, series.kind, x_values, y_values)
        return trace, live


@dataclass
class _PanelBuild:
    panel: Panel
    yaxis_title: str | None = None
    secondary_yaxis_title: str | None = None


def plot_timeseries(
    source: Any | None = None,
    y: Any = None,
    *,
    x: str = "time",
    filters: Sequence[pl.Expr] = (),
    label_cols: Sequence[str] = (),
    events: Any = None,
    panels: Sequence[Any] | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    yaxis_titles: Sequence[str | None] | None = None,
    secondary_yaxis_titles: Sequence[str | None] | None = None,
    start: Any = None,
    end: Any = None,
    live: bool = False,
    return_viewer: bool = False,
    layout: Mapping[str, Any] | None = None,
    shared_x: bool = True,
    timezone: str | None = None,
    spikemode: str = "across",
    spikesnap: str = "cursor",
    hovermode: str = "x",
    x_label: str | None = None,
    datetime_x_mode: str = "datetime",
    live_readout: bool = False,
    max_points: int | None = 200_000,
    resample: str = "auto",
    vertical_spacing: float = 0.03,
    height: int | None = None,
    template: str = "plotly_white",
    polars_engine: str = "streaming",
) -> Any:
    """Build a Plotly figure from concise timeseries specs.

    ``source`` + ``y`` is the common path. Use ``panels`` when you need
    multiple subplot rows, different sources, or source-specific filters.
    Dict specs are translated to the lower-level ``Panel``/``Layer``/``Series``
    objects used by ``DataViewer``.
    """

    panel_builds = _coerce_panel_builds(
        panels,
        source=source,
        y=y,
        events=events,
        x=x,
        filters=filters,
        label_cols=label_cols,
        yaxis_title=yaxis_title,
    )
    viewer = DataViewer(
        [item.panel for item in panel_builds],
        shared_x=shared_x,
        timezone=timezone,
        spikemode=spikemode,
        spikesnap=spikesnap,
        hovermode=hovermode,
        x_label=x_label,
        datetime_x_mode=datetime_x_mode,
        live_readout=live_readout,
        max_points=max_points,
        resample=resample,
        vertical_spacing=vertical_spacing,
        height=height,
        template=template,
        polars_engine=polars_engine,
    )
    fig = viewer.figure(start=start, end=end, live=live)
    _apply_axis_titles(fig, panel_builds, yaxis_titles, secondary_yaxis_titles)
    if title is not None:
        fig.update_layout(title=title)
    if layout is not None:
        fig.update_layout(**dict(layout))
    return (fig, viewer) if return_viewer else fig


def x_pan_buttons(
    fig: Any,
    *,
    step: float = 0.5,
    bounds: tuple[Any, Any] | str | None = "data",
    labels: tuple[str, str] = ("Left", "Right"),
) -> Any:
    """Return notebook buttons that pan a Plotly FigureWidget along x.

    ``step`` is a fraction of the current visible x-width. ``bounds="data"``
    clamps panning to the plotted x extent; pass ``None`` to allow free panning.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("x_pan_buttons requires ipywidgets") from exc

    left = widgets.Button(description=labels[0], tooltip="Pan left")
    right = widgets.Button(description=labels[1], tooltip="Pan right")

    def pan(direction: int) -> None:
        current = _xaxis_range(fig)
        if current is None:
            current = _data_x_bounds(fig)
        if current is None:
            return
        new_range = _panned_range(
            current,
            direction=direction,
            step=step,
            bounds=_resolve_x_bounds(fig, bounds),
        )
        if new_range is None:
            return
        with fig.batch_update():
            fig.update_xaxes(range=list(new_range))

    left.on_click(lambda _: pan(-1))
    right.on_click(lambda _: pan(1))
    return widgets.HBox([left, right])


add_x_pan_buttons = x_pan_buttons


def _coerce_panel_builds(
    panels: Sequence[Any] | None,
    *,
    source: Any | None,
    y: Any,
    events: Any,
    x: str,
    filters: Sequence[pl.Expr],
    label_cols: Sequence[str],
    yaxis_title: str | None,
) -> list[_PanelBuild]:
    if panels is None:
        if source is None:
            raise ValueError("source is required when panels is not provided")
        panels = [
            {
                "source": source,
                "x": x,
                "y": y,
                "events": events,
                "filters": filters,
                "label_cols": label_cols,
                "yaxis_title": yaxis_title,
            }
        ]
        default_filters: tuple[pl.Expr, ...] = ()
        default_label_cols: tuple[str, ...] = ()
    else:
        default_filters = _expr_tuple(filters)
        default_label_cols = _str_tuple(label_cols)
        panels = _spec_items(panels)

    out = [
        _panel_build_from_spec(
            panel,
            default_source=source,
            default_x=x,
            default_filters=default_filters,
            default_label_cols=default_label_cols,
        )
        for panel in panels
    ]
    if not out:
        raise ValueError("panels must contain at least one panel")
    return out


def _panel_build_from_spec(
    spec: Any,
    *,
    default_source: Any | None,
    default_x: str,
    default_filters: Sequence[pl.Expr],
    default_label_cols: Sequence[str],
) -> _PanelBuild:
    if isinstance(spec, Panel):
        return _PanelBuild(spec)
    if not isinstance(spec, Mapping):
        raise TypeError("panel specs must be Panel instances or mappings")

    panel_source = spec.get("source", default_source)
    panel_x = spec.get("x", default_x)
    panel_filters = _merge_exprs(default_filters, _spec_filters(spec))
    panel_label_cols = _ordered_unique(
        [*default_label_cols, *_str_tuple(spec.get("label_cols", ()))]
    )
    panel_sort_x = bool(spec.get("sort_x", True))

    if "layers" in spec:
        layers = [
            _layer_from_spec(
                layer,
                default_source=panel_source,
                default_x=panel_x,
                default_filters=panel_filters,
                default_label_cols=panel_label_cols,
                default_sort_x=panel_sort_x,
            )
            for layer in _spec_items(spec["layers"])
        ]
    else:
        layers = _layers_from_panel_spec(
            spec,
            default_source=panel_source,
            default_x=panel_x,
            default_filters=default_filters,
            default_label_cols=default_label_cols,
            default_sort_x=panel_sort_x,
        )

    return _PanelBuild(
        Panel(
            layers=layers,
            title=spec.get("title"),
            height=float(spec.get("height", 1.0)),
        ),
        yaxis_title=spec.get("yaxis_title"),
        secondary_yaxis_title=spec.get("secondary_yaxis_title"),
    )


def _layers_from_panel_spec(
    spec: Mapping[str, Any],
    *,
    default_source: Any | None,
    default_x: str,
    default_filters: Sequence[pl.Expr],
    default_label_cols: Sequence[str],
    default_sort_x: bool,
) -> list[Layer]:
    base_source = spec.get("source", default_source)
    if base_source is None:
        raise ValueError("panel spec is missing source")
    base_x = spec.get("x", default_x)
    base_filters = _merge_exprs(default_filters, _spec_filters(spec))
    base_label_cols = _ordered_unique(
        [*default_label_cols, *_str_tuple(spec.get("label_cols", ()))]
    )
    base_sort_x = bool(spec.get("sort_x", default_sort_x))

    layers: list[Layer] = []
    level_series = [
        *_series_list(spec.get("series"), default_kind="level"),
        *_series_list(spec.get("y"), default_kind="level"),
    ]
    if level_series:
        layers.append(
            Layer(
                source=base_source,
                series=level_series,
                x=base_x,
                filters=base_filters,
                label_cols=base_label_cols,
                sort_x=base_sort_x,
            )
        )

    for event_spec in _series_spec_items(spec.get("events")):
        event_source = (
            event_spec.get("source", base_source)
            if isinstance(event_spec, Mapping)
            else base_source
        )
        event_x = event_spec.get("x", base_x) if isinstance(event_spec, Mapping) else base_x
        event_filters = (
            _merge_exprs(base_filters, _spec_filters(event_spec))
            if isinstance(event_spec, Mapping)
            else base_filters
        )
        event_label_cols = (
            _ordered_unique([*base_label_cols, *_str_tuple(event_spec.get("label_cols", ()))])
            if isinstance(event_spec, Mapping)
            else base_label_cols
        )
        event_sort_x = (
            bool(event_spec.get("sort_x", base_sort_x))
            if isinstance(event_spec, Mapping)
            else base_sort_x
        )
        layers.append(
            Layer(
                source=event_source,
                series=[_series_from_spec(event_spec, default_kind="event")],
                x=event_x,
                filters=event_filters,
                label_cols=event_label_cols,
                sort_x=event_sort_x,
            )
        )

    if not layers:
        raise ValueError("panel spec must include y, series, events, or layers")
    return layers


def _layer_from_spec(
    spec: Any,
    *,
    default_source: Any | None,
    default_x: str,
    default_filters: Sequence[pl.Expr],
    default_label_cols: Sequence[str],
    default_sort_x: bool,
) -> Layer:
    if isinstance(spec, Layer):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("layer specs must be Layer instances or mappings")

    source = spec.get("source", default_source)
    if source is None:
        raise ValueError("layer spec is missing source")
    series = [
        *_series_list(spec.get("series"), default_kind="level"),
        *_series_list(spec.get("y"), default_kind="level"),
        *_series_list(spec.get("events"), default_kind="event"),
    ]
    if not series:
        raise ValueError("layer spec must include y, series, or events")
    return Layer(
        source=source,
        series=series,
        x=spec.get("x", default_x),
        filters=_merge_exprs(default_filters, _spec_filters(spec)),
        label_cols=_ordered_unique(
            [*default_label_cols, *_str_tuple(spec.get("label_cols", ()))]
        ),
        sort_x=bool(spec.get("sort_x", default_sort_x)),
    )


def _series_list(specs: Any, *, default_kind: str) -> list[Series]:
    return [
        _series_from_spec(spec, default_kind=default_kind)
        for spec in _series_spec_items(specs)
    ]


def _series_spec_items(specs: Any) -> list[Any]:
    if specs is None:
        return []
    if isinstance(specs, Mapping) and "y" not in specs:
        return [_named_series_spec(name, spec) for name, spec in specs.items()]
    if _is_single_series_spec(specs):
        return [specs]
    return list(specs)


def _named_series_spec(name: Any, spec: Any) -> Any:
    if isinstance(spec, Mapping):
        out = dict(spec)
        out.setdefault("name", str(name))
        out.setdefault("y", name)
        return out
    return {"name": str(name), "y": spec}


def _is_single_series_spec(spec: Any) -> bool:
    return isinstance(spec, (Series, str, pl.Expr)) or isinstance(spec, Mapping)


def _series_from_spec(spec: Any, *, default_kind: str) -> Series:
    if isinstance(spec, Series):
        return spec
    if isinstance(spec, (str, pl.Expr)):
        return Series(y=spec, kind=default_kind)
    if not isinstance(spec, Mapping):
        raise TypeError(
            "series specs must be Series instances, column names, expressions, or mappings"
        )

    series_fields = set(Series.__dataclass_fields__)
    meta_fields = {"source", "x", "filters", "filter", "label_cols", "sort_x"}
    unknown = set(spec) - series_fields - meta_fields
    if unknown:
        raise ValueError(f"unknown Series spec keys: {sorted(unknown)}")
    values = {key: spec[key] for key in series_fields if key in spec}
    values.setdefault("kind", default_kind)
    if "y" not in values:
        raise ValueError("series spec is missing y")
    return Series(**values)


def _spec_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (Layer, Panel, Mapping)):
        return [value]
    return list(value)


def _spec_filters(spec: Any) -> tuple[pl.Expr, ...]:
    if not isinstance(spec, Mapping):
        return ()
    return _merge_exprs(
        _expr_tuple(spec.get("filters", ())),
        _expr_tuple(spec.get("filter", ())),
    )


def _expr_tuple(value: Any) -> tuple[pl.Expr, ...]:
    if value is None:
        return ()
    if isinstance(value, pl.Expr):
        return (value,)
    return tuple(value)


def _merge_exprs(*groups: Sequence[pl.Expr]) -> tuple[pl.Expr, ...]:
    return tuple(expr for group in groups for expr in group)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _apply_axis_titles(
    fig: Any,
    panel_builds: Sequence[_PanelBuild],
    yaxis_titles: Sequence[str | None] | None,
    secondary_yaxis_titles: Sequence[str | None] | None,
) -> None:
    primary_titles = list(yaxis_titles) if yaxis_titles is not None else None
    secondary_titles = list(secondary_yaxis_titles) if secondary_yaxis_titles is not None else None
    for idx, item in enumerate(panel_builds):
        row = idx + 1
        primary = (
            primary_titles[idx]
            if primary_titles is not None and idx < len(primary_titles)
            else item.yaxis_title
        )
        if primary:
            fig.update_yaxes(title_text=primary, row=row, col=1)
        secondary = (
            secondary_titles[idx]
            if secondary_titles is not None and idx < len(secondary_titles)
            else item.secondary_yaxis_title
        )
        if secondary and _panel_has_secondary_y(item.panel):
            fig.update_yaxes(title_text=secondary, row=row, col=1, secondary_y=True)


def _panel_has_secondary_y(panel: Panel) -> bool:
    return any(series.secondary_y for layer in panel.layers for series in layer.series)


def _xaxis_range(fig: Any) -> tuple[Any, Any] | None:
    xaxis = getattr(getattr(fig, "layout", None), "xaxis", None)
    value = getattr(xaxis, "range", None)
    if value is None or len(value) != 2 or value[0] is None or value[1] is None:
        return None
    return value[0], value[1]


def _resolve_x_bounds(fig: Any, bounds: tuple[Any, Any] | str | None) -> tuple[Any, Any] | None:
    if bounds is None:
        return None
    if bounds == "data":
        return _data_x_bounds(fig)
    if isinstance(bounds, str):
        raise ValueError("bounds must be 'data', None, or a (start, end) tuple")
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two values")
    return bounds


def _data_x_bounds(fig: Any) -> tuple[Any, Any] | None:
    values = []
    for trace in getattr(fig, "data", ()):
        x_values = getattr(trace, "x", None)
        if x_values is None:
            continue
        values.extend(value for value in x_values if value is not None)
    if not values:
        return None
    try:
        return min(values), max(values)
    except Exception:
        converted = [_pan_value(value) for value in values]
        try:
            return min(converted), max(converted)
        except Exception:
            return None


def _panned_range(
    current: tuple[Any, Any],
    *,
    direction: int,
    step: float,
    bounds: tuple[Any, Any] | None,
) -> tuple[Any, Any] | None:
    start, end = _pan_value(current[0]), _pan_value(current[1])
    try:
        width = end - start
        shift = width * (direction * step)
        new_start, new_end = start + shift, end + shift
    except Exception:
        return None
    return _clamp_range(new_start, new_end, bounds)


def _clamp_range(
    start: Any,
    end: Any,
    bounds: tuple[Any, Any] | None,
) -> tuple[Any, Any]:
    if bounds is None:
        return start, end
    lower, upper = _pan_value(bounds[0]), _pan_value(bounds[1])
    try:
        width = end - start
        full_width = upper - lower
        if width >= full_width:
            return lower, upper
        if start < lower:
            return lower, lower + width
        if end > upper:
            return upper - width, upper
    except Exception:
        pass
    return start, end


def _pan_value(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return _parse_datetime_like(value)
    if isinstance(value, str):
        try:
            return _parse_datetime_string(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "DataViewer requires plotly for figure(), show(), and to_html()."
        ) from exc
    return go, make_subplots


def _source_to_lazy(source: Any) -> pl.LazyFrame:
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if hasattr(source, "frame"):
        return source.frame(select=False)
    raise TypeError("Layer.source must be a DataSource, LazyFrame, or DataFrame")


def _schema_dtype(schema: Any, col: str) -> Any:
    names = schema.names() if hasattr(schema, "names") else list(schema)
    if col not in names:
        raise ValueError(f"x column {col!r} not found in layer source")
    return schema[col]


def _add_expression_series(lf: pl.LazyFrame, layer: Layer) -> pl.LazyFrame:
    exprs = []
    for series in layer.series:
        if isinstance(series.y, pl.Expr):
            exprs.append(series.y.alias(series.y_col))
    return lf.with_columns(exprs) if exprs else lf


def _needed_columns(layer: Layer) -> list[str]:
    cols = [layer.x]
    for series in layer.series:
        cols.append(series.y_col if isinstance(series.y, pl.Expr) else series.y)
        cols.extend(layer.label_cols)
        cols.extend(series.label_cols)
    return _ordered_unique(cols)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    return [float(value) / total for value in values]


def _series_max_points(series: Series, default: int | None) -> int | None:
    return default if series.max_points is None else series.max_points


def _resample_indices(
    kind: str,
    y_values: Sequence[Any],
    max_points: int | None,
    resample: str | None,
) -> list[int]:
    n = len(y_values)
    if n == 0:
        return []
    if max_points is None or resample == "none" or n <= max_points:
        return list(range(n))
    if max_points <= 0:
        return []
    if kind == "level":
        return _level_indices(y_values, max_points)
    # Continuous lines and point events are decimated uniformly. For lines this
    # depends only on row count and budget, keeping same-length signal columns
    # aligned even though each trace is constructed independently.
    return _uniform_indices(n, max_points)


def _level_indices(y_values: Sequence[Any], max_points: int) -> list[int]:
    n = len(y_values)
    indices = [0]
    for idx in range(1, n):
        if not _same_value(y_values[idx], y_values[idx - 1]):
            indices.append(idx)
    if indices[-1] != n - 1:
        indices.append(n - 1)
    if len(indices) <= max_points:
        return indices
    if max_points == 1:
        return [indices[-1]]
    return _bucket_last(indices, max_points)


def _bucket_last(indices: Sequence[int], max_points: int) -> list[int]:
    if max_points <= 1:
        return [indices[-1]]
    rest = indices[1:]
    buckets = max_points - 1
    edges = np.linspace(0, len(rest), buckets + 1, dtype=int)
    selected = [indices[0]]
    for left, right in zip(edges[:-1], edges[1:]):
        if right > left:
            selected.append(rest[right - 1])
    return _ordered_unique_int(selected)[:max_points]


def _uniform_indices(n: int, max_points: int) -> list[int]:
    if max_points <= 0:
        return []
    if max_points >= n:
        return list(range(n))
    return _ordered_unique_int(np.linspace(0, n - 1, max_points, dtype=int).tolist())


def _ordered_unique_int(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _same_value(left: Any, right: Any) -> bool:
    try:
        if left != left and right != right:
            return True
    except Exception:
        pass
    try:
        return bool(left == right)
    except Exception:
        return False


def _take(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[idx] for idx in indices]


def _customdata(
    df: pl.DataFrame,
    label_cols: Sequence[str],
    indices: Sequence[int],
) -> np.ndarray:
    cols = [_take(df.get_column(col).to_list(), indices) for col in label_cols]
    return np.array(cols, dtype=object).T


def _hovertemplate(
    name: str,
    label_cols: Sequence[str],
    is_datetime_x: bool,
    value_label: str | None = None,
    x_label: str = "x",
    x_customdata_index: int | None = None,
    label_customdata_offset: int = 0,
) -> str:
    x_value = (
        f"%{{customdata[{x_customdata_index}]}}"
        if x_customdata_index is not None
        else ("%{x|%H:%M:%S.%f}" if is_datetime_x else "%{x}")
    )
    y_name = value_label or "y"
    parts = [
        f"<b>{escape(name)}</b>",
        f"{escape(x_label)}={x_value}",
        f"{escape(y_name)}=%{{y}}",
    ]
    for idx, col in enumerate(label_cols):
        parts.append(f"{escape(col)}=%{{customdata[{idx + label_customdata_offset}]}}")
    return "<br>".join(parts) + f"<extra>{escape(name)}</extra>"


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _is_datetime_dtype(dtype: Any) -> bool:
    return _base_type(dtype) == pl.Datetime


def _is_date_dtype(dtype: Any) -> bool:
    return _base_type(dtype) == pl.Date


def _base_type(dtype: Any) -> Any:
    base_type = getattr(dtype, "base_type", None)
    return base_type() if callable(base_type) else dtype


def _dtype_time_zone(dtype: Any) -> str | None:
    return getattr(dtype, "time_zone", None)


def _zoneinfo(name: str) -> timezone | ZoneInfo:
    return timezone.utc if name.upper() == "UTC" else ZoneInfo(name)


def _coerce_bound(value: Any, x_dtype: Any, display_tz: str | None) -> Any:
    if value is None:
        return None
    if _is_datetime_dtype(x_dtype):
        return _coerce_datetime_bound(value, x_dtype, display_tz)
    if _is_date_dtype(x_dtype):
        dt = _parse_datetime_like(value)
        return dt.date() if isinstance(dt, datetime) else dt
    return value


def _coerce_datetime_bound(value: Any, x_dtype: Any, display_tz: str | None) -> datetime:
    dt = _parse_datetime_like(value)
    if not isinstance(dt, datetime):
        raise TypeError(f"cannot coerce {value!r} to a datetime bound")

    source_tz = _dtype_time_zone(x_dtype)
    source_zone = _zoneinfo(source_tz) if source_tz is not None else None

    if dt.tzinfo is None:
        if display_tz is not None:
            dt = dt.replace(tzinfo=_zoneinfo(display_tz))
            if source_zone is not None:
                return dt.astimezone(source_zone)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        if source_zone is not None:
            return dt.replace(tzinfo=source_zone)
        return dt

    if source_zone is not None:
        return dt.astimezone(source_zone)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_datetime_like(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="us")
        return _parse_datetime_string(text)
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime()
    if isinstance(value, str):
        return _parse_datetime_string(value)
    return value


def _parse_datetime_string(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _display_time_expr(x: str, x_dtype: Any, display_tz: str) -> pl.Expr:
    expr = pl.col(x)
    if _dtype_time_zone(x_dtype) is None:
        expr = expr.dt.replace_time_zone("UTC")
    return expr.dt.convert_time_zone(display_tz).alias(x)


def _x_ns_expr(x: str, x_dtype: Any) -> pl.Expr:
    expr = pl.col(x)
    if _dtype_time_zone(x_dtype) is None:
        expr = expr.dt.replace_time_zone("UTC")
    return expr.dt.epoch("ns")


def _x_hover_expr(x: str, x_dtype: Any, display_tz: str | None) -> pl.Expr:
    expr = pl.col(x)
    if display_tz is not None:
        if _dtype_time_zone(x_dtype) is None:
            expr = expr.dt.replace_time_zone("UTC")
        expr = expr.dt.convert_time_zone(display_tz)
    return expr.dt.strftime("%H:%M:%S.%f")


def _apply_relative_ns_x(items: list[_LayerData]) -> None:
    values = []
    for item in items:
        if item.x_ns_col is None or not item.df.height:
            continue
        values.extend(item.df.get_column(item.x_ns_col).drop_nulls().to_list())
    if not values:
        return
    origin = int(min(values))
    for item in items:
        if item.x_ns_col is None:
            continue
        item.df = item.df.with_columns(
            (pl.col(item.x_ns_col) - origin).alias(_X_PLOT_COL)
        )
        item.x_plot_col = _X_PLOT_COL
        item.x_origin_ns = origin


def _apply_relative_ns_axis(
    fig: Any,
    items: Sequence[_LayerData],
    display_tz: str | None,
) -> None:
    origin = next((item.x_origin_ns for item in items if item.x_origin_ns is not None), None)
    if origin is None:
        return
    values = []
    for item in items:
        if item.x_plot_col is None or not item.df.height:
            continue
        values.extend(item.df.get_column(item.x_plot_col).drop_nulls().to_list())
    if not values:
        return
    tickvals = _relative_ns_tick_values(int(min(values)), int(max(values)))
    ticktext = [_format_epoch_ns(origin + int(value), display_tz) for value in tickvals]
    fig.update_xaxes(type="linear", tickmode="array", tickvals=tickvals, ticktext=ticktext)


def _relative_ns_tick_values(low: int, high: int, count: int = 6) -> list[int]:
    if high <= low:
        return [low]
    return _ordered_unique_int(np.linspace(low, high, count, dtype=np.int64).tolist())


def _format_epoch_ns(value: int, display_tz: str | None) -> str:
    seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    if display_tz is not None:
        dt = dt.astimezone(_zoneinfo(display_tz))
    return f"{dt:%H:%M:%S}.{nanoseconds:09d}"


def _attach_live_hover(fig: Any, live_series: list[_LiveSeries]) -> None:
    if not live_series:
        return
    fig.add_annotation(
        x=1.0,
        y=1.0,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font={"size": 12},
        text="",
        visible=False,
    )
    annotation_idx = len(fig.layout.annotations) - 1

    def on_hover(trace: Any, points: Any, state: Any) -> None:
        hovered = _hover_x(points, trace)
        if hovered is None:
            return
        lines = []
        for item in live_series:
            idx = _live_index(item, hovered)
            if idx is not None:
                lines.append(f"{escape(item.name)}: {escape(str(item.y[idx]))}")
        with fig.batch_update():
            fig.layout.annotations[annotation_idx].text = "<br>".join(lines)
            fig.layout.annotations[annotation_idx].visible = bool(lines)

    def on_unhover(trace: Any, points: Any, state: Any) -> None:
        with fig.batch_update():
            fig.layout.annotations[annotation_idx].text = ""
            fig.layout.annotations[annotation_idx].visible = False

    for trace in fig.data:
        trace.on_hover(on_hover)
        trace.on_unhover(on_unhover)


def _hover_x(points: Any, trace: Any) -> Any:
    xs = getattr(points, "xs", None)
    if xs:
        return xs[0]
    point_inds = getattr(points, "point_inds", None)
    if point_inds:
        return trace.x[point_inds[0]]
    return None


def _live_index(item: _LiveSeries, hovered: Any) -> int | None:
    if not item.x:
        return None
    value = _coerce_live_value(hovered, item.x[0])
    try:
        pos = int(np.searchsorted(item.x, value, side="right"))
    except Exception:
        return None
    if item.kind == "level":
        idx = pos - 1
        return idx if 0 <= idx < len(item.x) else None
    candidates = []
    if 0 <= pos < len(item.x):
        candidates.append(pos)
    if 0 <= pos - 1 < len(item.x):
        candidates.append(pos - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: _distance(item.x[idx], value))


def _coerce_live_value(value: Any, sample: Any) -> Any:
    if isinstance(sample, datetime) and isinstance(value, str):
        parsed = _parse_datetime_string(value)
        if sample.tzinfo is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=sample.tzinfo)
        if sample.tzinfo is None and parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return value


def _distance(left: Any, right: Any) -> float:
    try:
        value = abs(left - right)
        return value.total_seconds() if hasattr(value, "total_seconds") else float(value)
    except Exception:
        return 0.0 if left == right else float("inf")


__all__ = [
    "DataViewer",
    "Layer",
    "Panel",
    "Series",
    "add_x_pan_buttons",
    "plot_timeseries",
    "x_pan_buttons",
]
