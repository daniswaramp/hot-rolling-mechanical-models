"""What-if atlas: one stored response grid per route, sliced without rescoring.

The Response surface tab scores a fresh 2D grid on every widget change, which is the right
trade for two axes held around the sidebar schedule. It cannot answer "what does the whole
thermal window look like": that is four axes at once, ~15,000 full passes through the mill's
feature engineering.

So the atlas is built once, stored as parquet on a Modal volume (``hrcm-jobs``), and sliced
locally. Building is the only expensive step and it happens only when the operator presses
"Build atlas"; loading reads the stored grid back ("Load last atlas"). Nothing here ever
builds on a rerun: with no atlas in session state the tab says so and returns.

Axes per route (the four that carry the metallurgy: entry, finish, cooling and gauge):

    Plate   entry temperature, finishing temperature, laminar header count, plate gauge
    HRC     transfer bar temperature, finishing temperature, coiling temperature, coil gauge

Every node is a full pass through ``streamlit_app.score_frame`` — the same scoring layer the
simulator uses — so a sliced atlas value is the number the simulator would give for that
schedule, not an interpolation of a proxy. The rest of the schedule (chemistry in particular)
is held at the sidebar values the atlas was built with.

Slice rendering interpolates the stored cube with ``scipy.interpolate.RegularGridInterpolator``
on a 60x60 evaluation grid, which is exact at the stored nodes and linear between them.
``scipy`` is imported inside ``interpolate_slice`` only: it is a real dependency but not worth
importing while the app boots.

``streamlit_app`` is imported lazily inside the helpers too. That is not laziness for speed:
``streamlit_app`` imports this module at boot, so a module-level import here would close the
cycle.
"""

import hashlib
import importlib
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import remote_jobs  # noqa: E402
try:
    importlib.reload(remote_jobs)
except Exception:  # noqa: BLE001
    pass

# The four axes of the atlas, per route. Order matters: it is the column order of the stored
# parquet and the order the evaluation grid is laid out in.
AXES = {
    "Plate": ["FM_Temp_Before_Rolling", "FM_Finish_Temp", "ACC_Laminar_Header", "Plate_T_mm"],
    "HRC": ["Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C", "Coil_T_mm"],
}

# Nodes per axis: 15 x 15 x 11 x 6 = 14,850 scoring passes, the resolution the plan fixed.
GRID_SIZES = {"Plate": [15, 15, 11, 6], "HRC": [15, 15, 11, 6]}

# Interpolation grid for one slice. 60x60 is smooth on screen and cheap to evaluate.
SLICE_GRID = 60

DEFAULT_ENGINE = remote_jobs.DEFAULT_ENGINE

AXIS_STYLE = dict(showline=True, linecolor="#e5e7eb", gridcolor="#f1f2f4", zeroline=False,
                  tickfont=dict(size=10, color="#6b7280"),
                  title_font=dict(size=11, color="#0b0d10"))


def _app():
    """The Streamlit app module, resolved on first use (it imports this module at boot).

    Resolved through app_ref so a `streamlit run` entry (the script executes as
    __main__) is reused instead of re-imported and re-executed.
    """
    import app_ref  # noqa: PLC0415 - avoids the import cycle
    return app_ref.get_app_module()


def _prune(key, options):
    """Drop a persisted widget value that has left the option list (else the widget raises)."""
    try:
        if key in st.session_state and st.session_state[key] not in options:
            del st.session_state[key]
    except Exception:  # noqa: BLE001 - no session state outside a script run
        pass


def _digest(payload):
    """Stable identity for one atlas payload; keys the slice cache and detects a new build."""
    return hashlib.sha1(bytes(payload)).hexdigest()[:16]


# ------------------------------------------------------------------------------- axis lookups
def spec_rows(mill):
    """The mill's SPECS rows by column key, straight from the app (the single source of ranges)."""
    return {row[1]: row for row in _app().SPECS[mill]}


def axis_labels(mill):
    rows = spec_rows(mill)
    return {key: rows[key][2] for key in AXES[mill] if key in rows}


def axis_units(mill):
    rows = spec_rows(mill)
    return {key: rows[key][3] for key in AXES[mill] if key in rows}


def axis_bounds(mill, key):
    """``(lo, hi)`` for one axis, from its SPECS row — the same window the sidebar slider uses."""
    row = spec_rows(mill)[key]
    return float(row[4]), float(row[5])


def axis_steps(mill, key):
    return float(spec_rows(mill)[key][7])


# ------------------------------------------------------------------------------- parquet read
def parse_atlas(payload, axis_order):
    """Read one stored atlas into ``(frame, grid, cube)``.

    ``frame`` is the tidy parquet table as stored (axis columns + target columns), ``grid``
    maps each axis to its sorted unique node values, and ``cube`` maps each target to a dense
    array shaped like ``grid`` in ``axis_order``. The cube is filled through the frame's own
    node positions rather than a blind reshape, so a row order change cannot silently rotate
    the response.
    """
    frame = pd.read_parquet(io.BytesIO(bytes(payload)))
    order = [str(key) for key in axis_order]
    missing = [key for key in order if key not in frame.columns]
    if missing:
        raise KeyError("atlas is missing axis column(s): " + ", ".join(missing))

    grid = {}
    for key in order:
        values = pd.to_numeric(frame[key], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"atlas carries no values for axis {key}")
        grid[key] = np.unique(values)

    shape = tuple(len(grid[key]) for key in order)
    positions = tuple(np.searchsorted(grid[key],
                                      pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float))
                      for key in order)
    cube = {}
    for column in frame.columns:
        if column in order:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if values.size != len(frame) or not np.isfinite(values).any():
            continue
        dense = np.full(shape, np.nan)
        dense[positions] = values
        cube[str(column)] = dense
    if not cube:
        raise ValueError("atlas carries no scored target column")
    return frame, grid, cube


# ------------------------------------------------------------------------------ interpolation
def interpolate_slice(values, grid, axis_order, x_axis, y_axis, fixed, n=SLICE_GRID):
    """Evaluate one 2D slice of the stored cube by multilinear interpolation.

    ``values`` is the target array shaped like ``grid`` in ``axis_order``; ``grid`` maps an
    axis to its ascending node values; ``fixed`` maps each held axis to the value it is held
    at. Returns ``(xs, ys, z)`` where ``z`` has shape ``(len(ys), len(xs))`` — the orientation
    ``go.Contour`` and ``go.Surface`` both want.

    A linear response is reproduced exactly: ``RegularGridInterpolator`` is exact at the
    nodes and linear between them. ``fixed`` values outside the stored window extrapolate
    linearly instead of raising, because a held axis is a number the operator typed and the
    honest answer there is "this is what the trend says", not a blank chart.
    """
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415 - heavy at boot

    order = [str(key) for key in axis_order]
    x_axis, y_axis = str(x_axis), str(y_axis)
    if x_axis == y_axis:
        raise ValueError("the horizontal and vertical axes must differ")
    for key in (x_axis, y_axis):
        if key not in order:
            raise KeyError(f"{key} is not one of the atlas axes {order}")

    held = [key for key in order if key not in (x_axis, y_axis)]
    nodes = [np.asarray(grid[key], dtype=float) for key in order]
    array = np.asarray(values, dtype=float)
    shape = tuple(node.size for node in nodes)
    if array.shape != shape:
        raise ValueError(f"target array {array.shape} does not match the grid {shape}")

    # Put the two swept axes first so the interpolated result reshapes straight to (y, x);
    # the held axes follow in the order the caller supplied them.
    evaluation_order = [x_axis, y_axis] + held
    position = {key: index for index, key in enumerate(order)}
    transposed = np.transpose(array, [position[key] for key in evaluation_order])
    evaluation_nodes = [np.asarray(grid[key], dtype=float) for key in evaluation_order]
    for key, node in zip(evaluation_order, evaluation_nodes):
        if node.size < 2:
            raise ValueError(f"axis {key} carries a single node, so it cannot be sliced")

    usable = transposed
    if np.isnan(usable).any():
        # A node that failed to score must not poison the whole slice: fill it from its
        # neighbours, which is what the surrounding response already implies.
        usable = _fill_nan(usable)

    interp = RegularGridInterpolator(evaluation_nodes, usable, bounds_error=False,
                                     fill_value=None)
    xs = np.linspace(evaluation_nodes[0].min(), evaluation_nodes[0].max(), int(n))
    ys = np.linspace(evaluation_nodes[1].min(), evaluation_nodes[1].max(), int(n))
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    query = [xg.ravel(), yg.ravel()]
    query += [np.full(xg.size, float(fixed[key])) for key in held]
    z = interp(np.column_stack(query)).reshape(xg.shape)
    return xs, ys, z


def _fill_nan(array):
    """Fill NaNs in a dense grid with the mean of the finite values along each axis."""
    out = np.array(array, dtype=float, copy=True)
    if not np.isfinite(out).any():
        return np.zeros_like(out)
    for axis in range(out.ndim):
        moved = np.moveaxis(out, axis, 0)
        for index in range(moved.shape[0]):
            plane = moved[index]
            bad = ~np.isfinite(plane)
            if not bad.any() or not np.isfinite(plane).any():
                continue
            plane[bad] = float(np.nanmean(plane))
            moved[index] = plane
        out = np.moveaxis(moved, 0, axis)
    return out


@st.cache_data(show_spinner=False, max_entries=32)
def _slice_cached(atlas_id, cube, grids, order, x_axis, y_axis, fixed_items, n=SLICE_GRID):
    """The slice cache: one entry per (atlas, axes, held values, target).

    Keyed on the atlas payload digest first, so pressing Build atlas again with a fresh grid
    invalidates every slice rather than replaying the previous response.
    """
    grid = {key: np.asarray(node, dtype=float) for key, node in zip(order, grids)}
    return interpolate_slice(cube, grid, order, x_axis, y_axis, dict(fixed_items), n=n)


# ------------------------------------------------------------------------------------ the tab
def view_atlas(mill, raw, engine=DEFAULT_ENGINE):
    """The What-if atlas tab: build or load one stored grid, then slice it.

    The build is the only expensive step and the only one that can spend Modal credits, so it
    sits behind the "Build atlas" button and its outcome is kept in
    ``st.session_state[f"atlas_{mill}"]``. A rerun replays that stored payload; it never
    rebuilds, and with nothing stored the tab says so and returns.
    """
    _app().section("what-if atlas")
    st.markdown('<div class="note">One stored grid per route, spanning the four axes that '
                'carry the metallurgy. Build it once on Modal, then slice any pair against any '
                'fixed values: every node is a full pass through the mill\'s own feature '
                'engineering, so the atlas agrees with the simulator.</div>',
                unsafe_allow_html=True)

    state_key = f"atlas_{mill}"
    axes = AXES[mill]
    sizes = GRID_SIZES[mill]
    specs = spec_rows(mill)
    reachable = [key for key in axes if key in specs]
    if len(reachable) != len(axes):
        st.error("This route is missing a spec row for "
                 + ", ".join(key for key in axes if key not in specs)
                 + ", so the atlas cannot cover its four axes.")
        return

    left, mid, right = st.columns([1, 1, 2], vertical_alignment="bottom")
    build = left.button("Build atlas", type="primary", key=f"atlas_build_{mill}",
                        help="Scores " + " x ".join(str(n) for n in sizes)
                             + " nodes through the mill's pipeline. This is the only "
                               "expensive step and the only one that uses Modal.")
    load = mid.button("Load last atlas", key=f"atlas_load_{mill}",
                      help="Reads the atlas stored on the Modal volume for this route.")
    current = st.session_state.get(state_key)
    right.caption(("Stored atlas in this session · " + " x ".join(str(n) for n in sizes)
                   + " nodes, built " + str(current.get("built"))) if current
                  else "Nothing stored yet. Build or load an atlas to explore this route.")

    if build:
        where = " on Modal" if remote_jobs.remote_ready() else " locally (Modal unavailable)"
        with st.spinner("Building the response atlas on Modal..."):
            payload, source = remote_jobs.build_atlas(mill, dict(raw), axes, sizes, engine=engine)
        if not payload:
            st.error(f"Could not build the atlas: {remote_jobs.last_error() or 'no bytes returned'}"
                     ". Check the model artifacts for this route, or the Modal deployment.")
        else:
            st.session_state[state_key] = {
                "bytes": payload,
                "source": source,
                "id": _digest(payload),
                "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "axes": list(axes),
                "grid_sizes": list(sizes),
                "engine": engine,
                "mill": mill,
            }
            st.caption(f"atlas built{where}")
    elif load:
        with st.spinner("Loading the last atlas..."):
            payload, source = remote_jobs.load_atlas(mill)
        if not payload:
            st.warning(f"No stored atlas for {mill} yet"
                       + (f" ({remote_jobs.last_error()})" if remote_jobs.last_error() else "")
                       + ". Press Build atlas to create one.")
        else:
            st.session_state[state_key] = {
                "bytes": payload,
                "source": source,
                "id": _digest(payload),
                "built": "a previous session",
                "axes": list(axes),
                "grid_sizes": list(sizes),
                "engine": engine,
                "mill": mill,
            }

    stored = st.session_state.get(state_key)
    if not stored:
        st.caption("Build or load an atlas to explore this route.")
        return
    if not build and not load:
        st.caption("Shown from your last atlas. Press Build atlas to rebuild it with the "
                   "current sidebar schedule.")

    _render_atlas(mill, stored, raw)


def _render_atlas(mill, stored, raw):
    """Render one stored atlas: source caption, axis pickers, the contour and the CSV export."""
    order = [str(key) for key in (stored.get("axes") or AXES[mill])]
    try:
        _frame, grid, cube = parse_atlas(stored["bytes"], order)
    except Exception as exc:  # noqa: BLE001 - a stored payload must not kill the tab
        st.error(f"Stored atlas could not be read: {type(exc).__name__}: {exc}")
        return

    labels = axis_labels(mill)
    units = axis_units(mill)
    targets = [key for key in _app().TARGET_LABEL[mill] if key in cube]
    if not targets:
        st.error("This atlas carries no property the route scores, so there is nothing to plot.")
        return

    caption = remote_jobs.source_label(stored.get("source"))
    if stored.get("built"):
        caption += f" · built {stored['built']}"
    if stored.get("engine"):
        caption += f" · {stored['engine']}"
    caption += (f" · {', '.join(str(len(grid[key])) for key in order)} nodes = "
                f"{int(np.prod([len(grid[key]) for key in order])):,} scored schedules")
    st.caption(caption)

    top = st.columns([1.1, 1.1, 1.3, 0.9], vertical_alignment="bottom")
    x_axis = top[0].selectbox("Horizontal axis", order, index=0, key=f"atlas_x_{mill}",
                              format_func=lambda k: labels.get(k, k))
    y_options = [key for key in order if key != x_axis]
    _prune(f"atlas_y_{mill}", y_options)
    y_axis = top[1].selectbox("Vertical axis", y_options, index=0, key=f"atlas_y_{mill}",
                              format_func=lambda k: labels.get(k, k))
    target = top[2].selectbox("Property", targets, index=0, key=f"atlas_t_{mill}",
                              format_func=lambda key: _app().TARGET_LABEL[mill][key][0])
    surface = top[3].toggle("3D surface", value=False, key=f"atlas_3d_{mill}")

    held = [key for key in order if key not in (x_axis, y_axis)]
    fixed = {}
    columns = st.columns(len(held) or 1, vertical_alignment="bottom")
    for column, key in zip(columns, held):
        node = np.asarray(grid[key], dtype=float)
        middle = float(node[(node.size - 1) // 2])
        lo, hi = float(node.min()), float(node.max())
        step = axis_steps(mill, key)
        if not step > 0:
            step = (hi - lo) / 100.0
        unit = units.get(key, "")
        fixed[key] = float(column.number_input(
            f"Fixed {labels.get(key, key)} [{unit}]" if unit else f"Fixed {labels.get(key, key)}",
            min_value=lo, max_value=hi, value=min(max(middle, lo), hi), step=step,
            format="%.4g" if step < 1 else "%.1f",
            key=f"atlas_fixed_{mill}_{key}",
            help="Held at this value while the two axes above are swept."))

    values = cube[target]
    try:
        xs, ys, z = _slice_cached(stored.get("id") or _digest(stored["bytes"]), values,
                                  tuple(np.asarray(grid[key], dtype=float) for key in order),
                                  tuple(order), x_axis, y_axis, tuple(sorted(fixed.items())),
                                  SLICE_GRID)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Slice failed: {type(exc).__name__}: {exc}")
        return

    label, unit = _app().TARGET_LABEL[mill][target]
    x_label, y_label = labels.get(x_axis, x_axis), labels.get(y_axis, y_axis)
    if surface:
        fig = go.Figure(go.Surface(x=xs, y=ys, z=z, colorscale="Blues", showscale=False,
                                   contours=dict(z=dict(show=True, usecolormap=True,
                                                        project_z=True))))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white",
                          scene=dict(xaxis=dict(title=x_label, **AXIS_STYLE),
                                     yaxis=dict(title=y_label, **AXIS_STYLE),
                                     zaxis=dict(title=f"{label} [{unit}]", **AXIS_STYLE),
                                     camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))))
    else:
        fig = go.Figure(go.Contour(
            x=xs, y=ys, z=z, colorscale="Blues",
            contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
            line=dict(width=0.4, color="rgba(255,255,255,.6)"),
            colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.8),
            hovertemplate=(f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<br>"
                           f"{label}: %{{z:.1f}} {unit}<extra></extra>"),
        ))
        fig.update_xaxes(title=x_label, **AXIS_STYLE)
        fig.update_yaxes(title=y_label, **AXIS_STYLE)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                          paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": True})

    held_text = ", ".join(
        f"{labels.get(key, key)} {fixed[key]:g} {units.get(key, '')}".strip() for key in held)
    st.markdown(f'<div class="note">Slice through the stored {mill} atlas: {x_label} against '
                f'{y_label}, with {held_text} held. Every grid node is a full pass through the '
                f'mill pipeline; the surface between nodes is linear interpolation, so read it '
                f'as a trend, not as a guarantee.</div>', unsafe_allow_html=True)

    tidy = pd.DataFrame({
        x_axis: np.repeat(xs, len(ys)),
        y_axis: np.tile(ys, len(xs)),
        target: z.T.ravel(),
    })
    for key in held:
        tidy[key] = fixed[key]
    st.download_button("Download this slice (CSV)",
                       data=tidy.to_csv(index=False).encode("utf-8"),
                       file_name=f"atlas_{mill}_{target}_{x_axis}_vs_{y_axis}.csv",
                       mime="text/csv", key=f"atlas_dl_{mill}")
