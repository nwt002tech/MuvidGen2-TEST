import streamlit as st
st.set_page_config(page_title="MuVidGen — Free Animated Edition (v2f)", layout="wide")
try:
    import app_real as app
    if hasattr(app, "render") and callable(app.render):
        app.render()
    elif hasattr(app, "main") and callable(app.main):
        app.main()
    else:
        st.warning("app_real.py has no render()/main().")
except Exception as e:
    st.error(f"Failed to import app_real: {e}")
