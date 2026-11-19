"""Modal deployment for the app's heavy calculations (called by ``app/remote_compute.py``).

Deploy once from anywhere, with Modal credentials in the environment::

    modal deploy modal_app.py

App name: ``hrcm-compute``. Three functions, each invoked only by an explicit user action
in the Streamlit UI ("Calculate recipe", "Calculate scores", "Calculate capability"):

    recipe_search(mill, raw, thickness, requirements, chemistries, entry_range,
                  finish_range, cooling, headers_range, ct_range, k)
        -> the dict ``optimizer.run_recipe_search`` returns

    score_batch(mill, payload, engine)
        -> scored frame as parquet bytes (payload is the uploaded frame as parquet bytes)

    capability(payload)
        -> the dict ``streamlit_app.capability_resample`` returns (payload is the small
           argument dict: mill, raw, target, lsl, usl, samples, bands, engine)

The image is the repository's own requirements.txt plus the repository itself, mounted at
/root/repo, so the remote code is byte-identical to the checkout that was deployed: the
same feature modules, the same model artifacts and the same selection math. Nothing in
this file runs at import time beyond declaring the app and the image; the containers only
start when one of the three functions is called.

Do not put credentials in this file or anywhere in the repository. The Modal client reads
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment of the deploying machine, and the
Streamlit app reads them from st.secrets or its own environment.
"""

import os
import sys

import modal

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REMOTE_REPO = "/root/repo"
REMOTE_APP_DIR = "/root/repo/app"

# The mount carries the repo, not the workshop: no history, no caches, no scratch output.
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

app = modal.App("hrcm-compute")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(os.path.join(REPO_ROOT, "requirements.txt"))
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, ignore=IMAGE_IGNORE)
)


def _enter_repo():
    """Make the shipped app importable and the repository the working directory.

    The app's modules import each other flat (``import model_loader``), so app/ must be on
    sys.path; the model artifacts and the specification workbook are located relative to
    the repository root, so the working directory has to be /root/repo. The chdir is
    guarded so a local in-process run (``Function.local``) also works off a checkout.
    """
    for path in (REMOTE_REPO, REMOTE_APP_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    if os.path.isdir(REMOTE_REPO):
        os.chdir(REMOTE_REPO)


# cpu / memory / timeout as documented in the repo plan; the idle window before a container
# scales down is `scaledown_window` in modal 1.x (the 0.x name `container_idle_timeout` is
# rejected by the installed client, so the deploy would fail with a TypeError).
@app.function(image=image, cpu=2.0, memory=2048, timeout=600, scaledown_window=600)
def recipe_search(mill, raw, thickness, requirements, chemistries, entry_range,
                  finish_range, cooling="air", headers_range=(4, 12), ct_range=None,
                  k=1.0):
    """The inverse-optimizer recipe search, running the local selection math verbatim."""
    _enter_repo()
    import optimizer  # noqa: PLC0415 - imported after sys.path is set up

    return optimizer.run_recipe_search(
        mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
        cooling=cooling, headers_range=headers_range, ct_range=ct_range, k=k)


@app.function(image=image, cpu=2.0, memory=2048, timeout=600, scaledown_window=600)
def score_batch(mill, payload, engine="Champion GBDT (Recommended)"):
    """Score an uploaded sheet: parquet bytes in, parquet bytes out.

    streamlit_app is imported inside the function because it is the heavy module in the
    app package (page config, styling, every view). Importing it here rather than at
    container import time keeps the cold start of the cheap functions short.
    """
    import io

    import pandas as pd

    _enter_repo()
    from streamlit_app import score_frame  # noqa: PLC0415

    frame = pd.read_parquet(io.BytesIO(payload))
    return score_frame(mill, frame, engine=engine).to_parquet(index=False)


@app.function(image=image, cpu=2.0, memory=2048, timeout=600, scaledown_window=600)
def capability(payload):
    """The Monte-Carlo capability study, through the same pure helper the local path uses."""
    _enter_repo()
    from streamlit_app import capability_resample  # noqa: PLC0415

    return capability_resample(
        payload["mill"], payload["raw"], payload["target"], payload["lsl"],
        payload["usl"], payload["samples"], payload["bands"], engine=payload["engine"])
