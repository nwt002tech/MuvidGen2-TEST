import streamlit as st, traceback
st.title("MuVidGen Boot Probe")
try:
    import app_real
    st.success("Imported app_real.py successfully.")
    st.write({"has_render": hasattr(app_real, "render"), "has_main": hasattr(app_real, "main")})
except Exception as e:
    st.error(f"Import error: {e.__class__.__name__}: {e}")
    st.code(traceback.format_exc())
