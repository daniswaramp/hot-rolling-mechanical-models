"""Modal deployment for the what-if atlas and the digital shadow (called by ``app/remote_jobs.py``).

Deploy once from anywhere, with Modal credentials in the environment::

    modal deploy modal_jobs.py

App name: ``hrcm-jobs``. Four functions, each invoked only by an explicit user action in the
Streamlit UI ("Build atlas", "Load last atlas", "Update shadow", "Load last calibration"):

    build_atlas(mill, raw, axes, grid_sizes, engine)
        -> the atlas as parquet bytes (axis columns first, then every target the route
           scores); the same bytes are written to /data/atlas/{mill}.parquet and committed

    load_atlas(mill)
        -> the parquet bytes stored at /data/atlas/{mill}.parquet, or None

    update_shadow(mill, csv_bytes, engine)
        -> the calibration dict ``remote_jobs.shadow_calibration`` returns (timestamp,
           n_rows, per-target median bias / p10 / p90 / n, per-grade block when the sheet
           carries a Grade column); the same dict is written to /data/shadow/{mill}.json
           and committed

    load_shadow(mill)
        -> the calibration dict stored at /data/shadow/{mill}.json, or None

The image is the repository's own requirements.txt plus the repository itself, mounted at
/root/repo, so the remote code is byte-identical to the checkout that was deployed: the same
feature modules, the same model artifacts, the same grid geometry and the same calibration
math. The grid build and the residual summary are not reimplemented here — they are called
from ``app/remote_jobs.py`` inside the container, so the remote and the local fallback path
cannot drift apart.

Storage is two Modal volumes, both mounted under /data:

    hrcm-atlas    -> /data        (holds /data/atlas/{mill}.parquet)
    hrcm-shadow   -> /data/shadow (holds /data/shadow/{mill}.json)

Two volumes cannot share one mount point in Modal, so the shadow volume is mounted one level
below the atlas volume. Every path published to the client is still the one this feature
documents: /data/atlas/{mill}.parquet and /data/shadow/{mill}.json.

Nothing in this file runs at import time beyond declaring the app, the image and the volume
handles; the containers only start when one of the four functions is called.

Do not put credentials in this file or anywhere in the repository. The Modal client reads
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment of the deploying machine, and the
Streamlit app reads them from st.secrets or its own environment.
"""

import json
import os
import sys

import modal

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REMOTE_REPO = "/root/repo"
REMOTE_APP_DIR = "/root/repo/app"

# The mount carries the repo, not the workshop: no history, no caches, no scratch output.
# Identical to modal_app.py so both deployments ship the same tree.
IMAGE_IGNORE = [
    ".git", ".git/**",
    "**/.git/**",
    "**/__pycache__/**", "**/*.pyc", "**/*.pyo",
    ".pytest_cache/**", "**/.pytest_cache/**",
    "tests", "tests/**",
    "notebooks", "notebooks/**",
    "docs", "docs/**",
    "graphify-out", "graphify-out/**",
    "push-to-GH", "push-to-GH/**",
    ".devcontainer", ".devcontainer/**",
    ".github", ".github/**",
    ".commandcode", ".commandcode/**",
]

app = modal.App("hrcm-jobs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(os.path.join(REPO_ROOT, "requirements.txt"))
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, ignore=IMAGE_IGNORE)
)

ATLAS_VOLUME = modal.Volume.from_name("hrcm-atlas", create_if_missing=True)
SHADOW_VOLUME = modal.Volume.from_name("hrcm-shadow", create_if_missing=True)

DATA_ROOT = "/data"
# POSIX literals, not os.path.join: these are evaluated on the deploying machine (this repo is
# developed on Windows) but consumed inside a Linux container, so a backslash here would be a
# mount path Modal cannot resolve.
ATLAS_DIR = "/data/atlas"
SHADOW_DIR = "/data/shadow"

# See the module docstring: the shadow volume sits one level below the atlas volume because
# two Modal volumes cannot be mounted on the same point.
VOLUMES = {DATA_ROOT: ATLAS_VOLUME, SHADOW_DIR: SHADOW_VOLUME}

# cpu / memory / timeout as documented in the repo plan; the idle window before a container
# scales down is `scaledown_window` in modal 1.x. A four-dimensional grid is ~15k full passes
# through the feature module, so this budget is deliberately larger than hrcm-compute's.
RESOURCES = dict(image=image, cpu=2.0, memory=2048, timeout=900, scaledown_window=600,
                 volumes=VOLUMES)


def _enter_repo():
    """Make the shipped app importable and the repository the working directory.

    The app's modules import each other flat (``import model_loader``), so app/ must be on
    sys.path; the model artifacts and the specification workbook are located relative to the
    repository root, so the working directory has to be /root/repo. The chdir is guarded so a
    local in-process run (``Function.local``) also works off a checkout.
    """
    for path in (REMOTE_REPO, REMOTE_APP_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    if os.path.isdir(REMOTE_REPO):
        os.chdir(REMOTE_REPO)


def _atlas_path(mill):
    return os.path.join(ATLAS_DIR, f"{mill}.parquet")


def _shadow_path(mill):
    return os.path.join(SHADOW_DIR, f"{mill}.json")


@app.function(**RESOURCES)
def build_atlas(mill, raw, axes, grid_sizes, engine="Champion GBDT (Recommended)"):
    """Score the four-dimensional grid and store it as parquet.

    The grid itself is built by ``remote_jobs.atlas_frame`` — the same function the app's
    local fallback calls — so the container and a credential-less laptop produce the same
    atlas. Bytes go back to the caller *and* to /data/atlas/{mill}.parquet.
    """
    _enter_repo()
    import remote_jobs  # noqa: PLC0415 - imported after sys.path is set up

    payload = remote_jobs.atlas_bytes_at(mill, raw, axes, grid_sizes, engine=engine)
    os.makedirs(ATLAS_DIR, exist_ok=True)
    with open(_atlas_path(mill), "wb") as handle:
        handle.write(payload)
    ATLAS_VOLUME.commit()
    return payload


@app.function(**RESOURCES)
def load_atlas(mill):
    """The last atlas committed for this route, as parquet bytes, or None."""
    _enter_repo()
    ATLAS_VOLUME.reload()  # see writes committed by another container since this one started
    path = _atlas_path(mill)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return handle.read()


@app.function(**RESOURCES)
def update_shadow(mill, csv_bytes, engine="Champion GBDT (Recommended)"):
    """Calibrate the route's model bias against a sheet of measured results.

    The residual summary is computed by ``remote_jobs.shadow_calibration`` — the same
    function the app's local fallback calls. The dict goes back to the caller *and* to
    /data/shadow/{mill}.json, so "Load last calibration" works from a fresh session.
    """
    _enter_repo()
    import remote_jobs  # noqa: PLC0415 - imported after sys.path is set up

    result = remote_jobs.shadow_calibration(mill, csv_bytes, engine=engine)
    os.makedirs(SHADOW_DIR, exist_ok=True)
    with open(_shadow_path(mill), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    SHADOW_VOLUME.commit()
    return result


@app.function(**RESOURCES)
def load_shadow(mill):
    """The last calibration committed for this route, as the dict the tab renders, or None."""
    _enter_repo()
    SHADOW_VOLUME.reload()  # see writes committed by another container since this one started
    path = _shadow_path(mill)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
