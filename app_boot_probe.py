import streamlit as st, traceback
st.set_page_config(page_title="MuVidGen – Boot Probe", layout="wide")
st.title("MuVidGen Boot Probe")
st.caption("If anything explodes below, we’ll see it instead of hanging.")
try:
    import app_real
    st.success("Imported app_real.py successfully.")
except Exception as e:
    st.error(f"Import error: {e.__class__.__name__}: {e}")
    import traceback as tb
    st.code(tb.format_exc())
