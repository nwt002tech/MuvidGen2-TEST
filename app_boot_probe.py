import streamlit as st, traceback

st.title("MuVidGen Boot Probe")
st.caption("Importing your real app and surfacing any startup exceptions…")

try:
    import app_real
    st.success("Imported app_real.py successfully.")
    st.write({
        "has_render": hasattr(app_real, "render"),
        "has_main": hasattr(app_real, "main")
    })
    if not (hasattr(app_real, "render") or hasattr(app_real, "main")):
        st.info("Tip: expose a render() function so app.py can call it.")
except Exception as e:
    st.error(f"Import error: {e.__class__.__name__}: {e}")
    st.code(traceback.format_exc())
