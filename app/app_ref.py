"""Resolve the running app module without re-executing it.

`streamlit run app/streamlit_app.py` executes the entry script as __main__, so a
plain `import streamlit_app` from a tab module builds a second copy of the module
and re-runs st.set_page_config, which Streamlit rejects. Resolution order:
an already-imported streamlit_app module, the __main__ script when it is the
app (it carries SPECS once the module body has run), then a real import for
test and notebook contexts where importing is safe.
"""

import sys


def get_app_module():
    mod = sys.modules.get("streamlit_app")
    if mod is not None and hasattr(mod, "SPECS"):
        return mod
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "SPECS"):
        return main_mod
    import streamlit_app  # noqa: PLC0415 - only reached outside a live run
    return streamlit_app
