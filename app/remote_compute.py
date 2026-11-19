"""Heavy calculation dispatch: Modal when it is configured, the local process otherwise.

Three calculations in this app are expensive enough to be worth moving off the Streamlit
process (and, on Streamlit Community Cloud, off the 1 GB RAM budget):

  * the inverse-optimizer recipe search  (``optimizer.run_recipe_search``)
  * batch scoring of an uploaded sheet   (``streamlit_app.score_frame``)
  * the Monte-Carlo capability study     (``streamlit_app.capability_resample``)

Each one is reached only from an explicit button press ("Calculate ..."), never from a
widget change and never while the script boots, so a user's Modal credits are never spent
on automatic recomputation. Per-keystroke simulator predictions stay local: they take
milliseconds and must remain interactive.

Design rules, all deliberate:

* ``modal`` is imported lazily inside ``_modal()``. Importing this module must never
  require the package: the public demo has to work with no Modal install at all.
* Every public function returns ``(result, source)`` with ``source`` either ``"modal"``
  or ``"local"``, so the UI can caption the numbers honestly.
* Any failure on the remote path (package missing, no credentials, an unreachable app,
  a deserialisation error, a raised exception inside the container) falls back to the
  local computation and reports ``source == "local"``. The exception type is kept in
  ``last_error()`` for debugging; messages are not, and credentials are never logged,
  rendered or included in anything this module returns.
* Credentials are read from ``st.secrets`` first (try/except: a missing secrets.toml
  raises) and from the process environment second. Values found in ``st.secrets`` are
  mirrored into the environment with ``setdefault`` because the Modal client reads them
  from there; nothing is ever written to disk.

Deployment of the remote side: see ``modal_app.py`` at the repository root
(``modal deploy modal_app.py``). App name ``hrcm-compute``; functions ``recipe_search``,
``score_batch`` and ``capability``.
"""

import io
import os
import sys

import pandas as pd

REMOTE_APP = "hrcm-compute"
RECIPE_FN = "recipe_search"
SCORE_FN = "score_batch"
CAPABILITY_FN = "capability"

TOKEN_KEYS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# False = not attempted yet, None = the package is not installed, module = available.
# A failed import is not cached by Python itself, so remember the outcome here.
_MODAL = False

# Result of remote_ready() once computed; see remote_ready(refresh=True) to re-check.
_READY = None

# Type name of the exception that last forced a fallback (never its message: it may
# quote request headers, and there is no reason to surface credentials in the UI).
_LAST_ERROR = None


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
    parses the secrets file). Credentials normally arrive at process start, so the cache
    is what a rerun wants; tests and a long-lived process can pass ``refresh=True``.
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
    """Name of the exception class that last forced a local fallback, or None."""
    return _LAST_ERROR


def source_label(source: str) -> str:
    """One honest caption for the UI, given the source a calculation reported."""
    if source == "modal":
        return "computed on Modal"
    return "computed locally (Modal unavailable or failed)"


def _remote(fn_name, *args, **kwargs):
    """Call one deployed Modal function and return its result.

    Raises on every failure (missing package, missing credentials, no such app, a timeout,
    an exception inside the container) so the caller can fall back. ``modal.Function
    .from_name`` looks the function up by name on each call: nothing is deployed,
    imported or executed at module import time.
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


def recipe_search(mill, raw, thickness, requirements, chemistries, entry_range,
                  finish_range, cooling="air", headers_range=(4, 12), ct_range=None,
                  k=1.0):
    """Hottest-feasible-first recipe search, remote when configured.

    Returns ``(result, source)``; the result is exactly what
    ``optimizer.run_recipe_search`` returns on both paths.
    """
    if remote_ready():
        try:
            return _remote(RECIPE_FN, mill, raw, thickness, requirements, chemistries,
                           entry_range, finish_range, cooling=cooling,
                           headers_range=headers_range, ct_range=ct_range, k=k), "modal"
        except Exception as exc:  # noqa: BLE001 - any remote failure falls back
            _fallback(exc)
    import optimizer
    return optimizer.run_recipe_search(
        mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
        cooling=cooling, headers_range=headers_range, ct_range=ct_range, k=k), "local"


def score_batch(mill, frame, engine="Champion GBDT (Recommended)"):
    """Score an uploaded sheet, remote when configured. Returns ``(frame, source)``.

    The frame travels as parquet bytes both ways, so column dtypes and NaN patterns
    survive the hop; the remote side runs the same ``streamlit_app.score_frame``.
    """
    if remote_ready():
        try:
            payload = frame.to_parquet(index=False)
            scored = _remote(SCORE_FN, mill, payload, engine)
            return pd.read_parquet(io.BytesIO(scored)), "modal"
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    import streamlit_app
    return streamlit_app.score_frame(mill, frame, engine=engine), "local"


def capability(mill, raw, target, lsl, usl, samples, bands,
               engine="Champion GBDT (Recommended)"):
    """Monte-Carlo capability study, remote when configured. Returns ``(dict, source)``.

    ``engine`` is the one addition to the documented signature: the UI lets the operator
    pick a scoring engine, and resampling through a different engine than the reference
    schedule would report a capability the app never promised.

    The dict is what ``streamlit_app.capability_resample`` returns: per-point sampled
    inputs plus the scored values, and the Cp/Cpk/out-of-spec summary.
    """
    payload = {
        "mill": mill,
        "raw": dict(raw),
        "target": target,
        "lsl": float(lsl),
        "usl": float(usl),
        "samples": int(samples),
        "bands": {str(key): float(band) for key, band in (bands or {}).items()},
        "engine": engine,
    }
    if remote_ready():
        try:
            return _remote(CAPABILITY_FN, payload), "modal"
        except Exception as exc:  # noqa: BLE001
            _fallback(exc)
    import streamlit_app
    return streamlit_app.capability_resample(**payload), "local"
