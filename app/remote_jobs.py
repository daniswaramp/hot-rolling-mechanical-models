"""Atlas and digital-shadow dispatch: Modal when it is configured, this process otherwise.

Two tabs in this app do work that is too heavy for the Streamlit process (and, on Streamlit
Community Cloud, too heavy for the 1 GB RAM budget):

  * the what-if response atlas  (``atlas.view_atlas``) — a four-dimensional grid, every node
    a full pass through the mill's own feature engineering, stored as parquet
  * the digital-shadow calibration (``shadow.view_shadow``) — score a sheet of measured
    results and turn the residuals into a bias / p10 / p90 summary per property

Both are reached only from an explicit button press ("Build atlas", "Load last atlas",
"Update shadow", "Load last calibration"), never from a widget change and never while the
script boots, so a user's Modal credits are never spent on automatic recomputation.

This module is deliberately a near-copy of ``remote_compute.py`` rather than an import of it:
the two dispatch paths are independent (they call different Modal apps, ``hrcm-jobs`` here
and ``hrcm-compute`` there), so a change to one must not be able to take the other down. The
four small helpers ``_modal``, ``_secret``, ``_remote`` and ``_fallback`` are duplicated on
purpose.

Design rules, all deliberate:

* ``modal`` is imported lazily inside ``_modal()``. Importing this module must never require
  the package: the public demo has to work with no Modal install at all.
* Every public function returns ``(result, source)`` with ``source`` either ``"modal"`` or
  ``"local"``, so the UI can caption the numbers honestly.
* Any failure on the remote path (package missing, no credentials, an unreachable app, a
  deserialisation error, a raised exception inside the container) falls back to the local
  computation and reports ``source == "local"``. The exception *type* is kept in
  ``last_error()`` for debugging; messages are not, and credentials are never logged,
  rendered or included in anything this module returns.
* Credentials are read from ``st.secrets`` first (try/except: a missing secrets.toml raises)
  and from the process environment second. Values found in ``st.secrets`` are mirrored into
  the environment with ``setdefault`` because the Modal client reads them from there; nothing
  is ever written to disk.
* The local fallback keeps its results in this process only. Nothing is written into the
  checkout: an atlas built with no Modal credentials lives in ``_LOCAL_ATLAS`` and a shadow
  calibration in ``_LOCAL_SHADOW``, so "Load last ..." works for the life of the process and
  a fresh process honestly reports that nothing is stored yet.

The grid build and the calibration math live here as plain functions (``atlas_frame``,
``shadow_calibration``) because the Modal functions in ``modal_jobs.py`` run exactly the same
bodies inside the container. One implementation means the remote and the local path cannot
disagree about what an atlas node or a median residual is.

Deployment of the remote side: see ``modal_jobs.py`` at the repository root
(``modal deploy modal_jobs.py``). App name ``hrcm-jobs``; functions ``build_atlas``,
``load_atlas``, ``update_shadow`` and ``load_shadow``.
"""

import io
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REMOTE_APP = "hrcm-jobs"
ATLAS_BUILD_FN = "build_atlas"
ATLAS_LOAD_FN = "load_atlas"
SHADOW_UPDATE_FN = "update_shadow"
SHADOW_LOAD_FN = "load_shadow"

DEFAULT_ENGINE = "Champion GBDT (Recommended)"

TOKEN_KEYS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# False = not attempted yet, None = the package is not installed, module = available.
# A failed import is not cached by Python itself, so remember the outcome here.
_MODAL = False

# Result of remote_ready() once computed; see remote_ready(refresh=True) to re-check.
_READY = None

# Type name of the exception that last forced a fallback (never its message: it may quote
# request headers, and there is no reason to surface credentials in the UI).
_LAST_ERROR = None

# Local-path results. In process only, on purpose: no atlas parquet and no calibration JSON
# is ever written into the checkout by a machine that has no Modal credentials.
_LOCAL_ATLAS = {}
_LOCAL_SHADOW = {}

SOURCE_LABELS = {"modal": "built on Modal", "local": "built locally"}


def _modal():
    """The ``modal`` module, or None. Lazy on purpose: never imported with this module."""
    global _MODAL
    if _MODAL is not False:
        return _MODAL
    try:
        import modal as _module
        _MODAL = _module
    except Exception:  # noqa: BLE001 - a missing/broken package is a normal fallback case
        _MODAL = None
    return _MODAL


def _secret(name):
    """One credential from st.secrets, else the environment. Never logged or returned."""
    value = None
    try:
        import streamlit as st
        try:
            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001 - no secrets.toml at all, or no such key
            value = None
    except Exception:  # noqa: BLE001
        value = None
    if value:
        # the Modal client authenticates from the environment, so mirror the secret
        os.environ.setdefault(name, str(value))
        return str(value).strip()
    value = os.environ.get(name)
    return str(value).strip() if value else None


def remote_ready(refresh: bool = False) -> bool:
    """True when modal imports and both MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are available.

    Cached in a module flag (the check imports the package and touches st.secrets, which
    parses the secrets file). Credentials normally arrive at process start, so the cache is
    what a rerun wants; tests and a long-lived process can pass ``refresh=True``.
    Never raises.
    """
    global _READY
    if _READY is not None and not refresh:
        return _READY
    try:
        ready = _modal() is not None and all(_secret(key) for key in TOKEN_KEYS)
    except Exception:  # noqa: BLE001
        ready = False
    _READY = bool(ready)
    return _READY


def last_error():
    """Name of the exception class that last forced (or defeated) a local fallback."""
    return _LAST_ERROR


def source_label(source: str) -> str:
    """One honest caption for the UI, given the source a build or load reported."""
    return SOURCE_LABELS.get(str(source or ""), "built locally")


def _remote(fn_name, *args, **kwargs):
    """Call one deployed Modal function and return its result.

    Raises on every failure (missing package, missing credentials, no such app, a timeout, an
    exception inside the container) so the caller can fall back. ``modal.Function.from_name``
    looks the function up by name on each call: nothing is deployed, imported or executed at
    module import time.
    """
    modal = _modal()
    if modal is None:
        raise RuntimeError("the modal package is not installed")
    if not remote_ready():
        raise RuntimeError("Modal is not configured")
    handle = modal.Function.from_name(REMOTE_APP, fn_name)
    return handle.remote(*args, **kwargs)


def _fallback(exc):
    """Record the failure class (never the message) and let the caller go local."""
    global _LAST_ERROR
    _LAST_ERROR = type(exc).__name__


# --------------------------------------------------------------------------- atlas compute
def grid_spec(mill, axes, grid_sizes):
    """``{axis: (lo, hi, n)}`` for the requested grid, read from the mill's SPECS rows.

    The bounds are the mill envelope in ``streamlit_app.SPECS`` — the same table the sidebar
    sliders come from — so an atlas node is always a schedule the operator could have typed.
    Read on both paths (here and inside ``modal_jobs.build_atlas``) so the remote grid and a
    local fallback cover identical geometry.
    """
    import streamlit_app  # noqa: PLC0415 - the heavy module, imported only when a build runs

    rows = {row[1]: row for row in streamlit_app.SPECS[mill]}
    axes = [str(key) for key in axes]
    sizes = [int(size) for size in grid_sizes]
    if len(sizes) != len(axes):
        raise ValueError(f"{len(axes)} axes but {len(sizes)} grid sizes")
    spec = {}
    for key, size in zip(axes, sizes):
        row = rows.get(key)
        if row is None:
            raise KeyError(f"{mill} carries no spec row for {key}")
        lo, hi = float(row[4]), float(row[5])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            raise ValueError(f"{mill}/{key}: spec range {lo}-{hi} is not usable")
        spec[key] = (lo, hi, max(size, 2))
    return spec


def atlas_frame(mill, raw, axes, grid_sizes, engine=DEFAULT_ENGINE):
    """The full product grid as a tidy frame: one column per axis plus one per scored target.

    Every row is a complete pass through the mill's own feature engineering
    (``streamlit_app.score_frame``), never a proxy. The rest of the schedule is held at the
    values in ``raw``; only the named axes move. Column order is axis order, so a reader can
    rebuild the N-dimensional cube from the frame without guessing.
    """
    import itertools  # noqa: PLC0415

    import streamlit_app  # noqa: PLC0415

    spec = grid_spec(mill, axes, grid_sizes)
    keys = list(spec)
    values = [np.linspace(lo, hi, n) for lo, hi, n in spec.values()]
    combinations = list(itertools.product(*values))
    frame = pd.DataFrame([dict(raw) for _ in combinations])
    for position, key in enumerate(keys):
        frame[key] = [combination[position] for combination in combinations]
    scored = streamlit_app.score_frame(mill, frame, engine=engine)
    out = frame[keys].copy()
    for target in streamlit_app.TARGET_LABEL[mill]:
        if target in scored.columns:
            out[target] = np.asarray(pd.to_numeric(scored[target], errors="coerce"), dtype=float)
    return out


def atlas_bytes_at(mill, raw, axes, grid_sizes, engine=DEFAULT_ENGINE):
    """The atlas as parquet bytes, built in this process (the local fallback body)."""
    return atlas_frame(mill, raw, axes, grid_sizes, engine=engine).to_parquet(index=False)


# -------------------------------------------------------------------------- shadow compute
def residual_summary(residuals):
    """Median bias plus the p10/p90 spread of one residual vector, or None when it is empty.

    The residual is measured minus predicted, so a positive bias means the model under-reads
    the property and ``predicted + bias`` is the centre of the measured population.
    """
    values = np.asarray(residuals, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    p10, p90 = np.percentile(values, [10, 90])
    return {"bias": float(np.median(values)), "p10": float(p10), "p90": float(p90),
            "n": int(values.size)}


def calibrate_residuals(residuals, grades=None):
    """Median bias + p10/p90 per target, globally and (when ``grades`` is given) per grade.

    ``residuals`` maps a target key to its residual vector (measured minus predicted, one
    entry per uploaded row, NaN for rows with no measurement). ``grades`` is the per-row grade
    label or None; an empty grade cell is dropped from the per-grade block rather than
    reported as a grade called "nan".
    """
    calibration = {}
    for target, values in residuals.items():
        summary = residual_summary(values)
        if summary is not None:
            calibration[str(target)] = summary

    by_grade = {}
    if grades is not None:
        labels = pd.Series(list(grades), dtype="string")
        vectors = {str(target): np.asarray(values, dtype=float).ravel()
                   for target, values in residuals.items()}
        for grade in sorted(set(labels.dropna().tolist())):
            text = str(grade).strip()
            if not text or text.lower() in ("nan", "none"):
                continue
            mask = (labels == grade).fillna(False).to_numpy(dtype=bool)
            block = {}
            for target, values in vectors.items():
                if values.size != mask.size:
                    continue
                summary = residual_summary(values[mask])
                if summary is not None:
                    block[target] = summary
            if block:
                by_grade[text] = block
    return {"calibration": calibration, "by_grade": by_grade}


def measured_columns(frame):
    """``{target: measured column}`` for every ``Measured_*`` column in the uploaded sheet.

    The prefix is matched case-insensitively; the suffix is the target key the app scores
    against (``Measured_TS_BN`` -> ``TS_BN``), because a calibration whose key does not match
    a scored target could never be applied.
    """
    out = {}
    for column in frame.columns:
        name = str(column)
        if name.lower().startswith("measured_"):
            out.setdefault(name[len("measured_"):], name)
    return out


def shadow_calibration(mill, csv_bytes, engine=DEFAULT_ENGINE):
    """Score an uploaded sheet of measured results and summarise the residuals.

    Returns the dict both the Shadow tab and ``modal_jobs.update_shadow`` store: the
    timestamp, the row count, the per-target calibration (median bias, p10/p90, n), the
    per-grade block when the sheet carries a ``Grade`` column, and the engine that produced
    the predictions. Models are read, never written: this is a measurement of bias, not a
    refit.
    """
    import streamlit_app  # noqa: PLC0415

    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if frame.empty:
        raise ValueError("the uploaded sheet carries no rows")
    measured = measured_columns(frame)
    if not measured:
        raise ValueError("the uploaded sheet carries no Measured_* column")

    scored = streamlit_app.score_frame(mill, frame, engine=engine)
    residuals, used = {}, {}
    for target, column in measured.items():
        if target not in scored.columns:
            continue
        predicted = np.asarray(pd.to_numeric(scored[target], errors="coerce"), dtype=float)
        actual = np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=float)
        residuals[target] = actual - predicted
        used[target] = column
    if not residuals:
        raise ValueError("no Measured_* column matches a property this route scores")

    grades = frame["Grade"] if "Grade" in frame.columns else None
    result = calibrate_residuals(residuals, grades)
    return {
        "mill": mill,
        "engine": engine,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(frame)),
        "measured_columns": used,
        "calibration": result["calibration"],
        "by_grade": result["by_grade"],
    }


# ------------------------------------------------------------------------------ public API
def build_atlas(mill, raw, axes, grid_sizes, engine=DEFAULT_ENGINE):
    """Build the response atlas. Returns ``(parquet bytes or None, source)``.

    The parquet payload is the ``atlas_frame`` grid (axis columns first, then every target
    the route scores), so the same reader parses a Modal build and a local one.
    """
    if remote_ready():
        try:
            return _remote(ATLAS_BUILD_FN, mill, dict(raw), [str(k) for k in axes],
                           [int(n) for n in grid_sizes], engine), "modal"
        except Exception as exc:  # noqa: BLE001 - any remote failure falls back
            _fallback(exc)
    try:
        payload = atlas_bytes_at(mill, raw, axes, grid_sizes, engine=engine)
    except Exception as exc:  # noqa: BLE001 - a failed build must not kill the tab
        _fallback(exc)
        return None, "local"
    _LOCAL_ATLAS[mill] = payload
    return payload, "local"


def load_atlas(mill):
    """The last stored atlas for this route. Returns ``(parquet bytes or None, source)``."""
    if remote_ready():
        try:
            return _remote(ATLAS_LOAD_FN, mill), "modal"
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    return _LOCAL_ATLAS.get(mill), "local"


def update_shadow(mill, csv_bytes, engine=DEFAULT_ENGINE):
    """Calibrate the route's model bias against measured results. Returns ``(dict or None, source)``.

    ``csv_bytes`` is the uploaded sheet verbatim; it travels to the container as bytes so the
    remote read is byte-identical to the local one.
    """
    if remote_ready():
        try:
            return _remote(SHADOW_UPDATE_FN, mill, csv_bytes, engine), "modal"
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    try:
        result = shadow_calibration(mill, csv_bytes, engine=engine)
    except Exception as exc:  # noqa: BLE001
        _fallback(exc)
        return None, "local"
    _LOCAL_SHADOW[mill] = result
    return result, "local"


def load_shadow(mill):
    """The last stored calibration for this route. Returns ``(dict or None, source)``."""
    if remote_ready():
        try:
            return _remote(SHADOW_LOAD_FN, mill), "modal"
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    return _LOCAL_SHADOW.get(mill), "local"
