import streamlit as st
st.set_page_config(page_title="MuVidGen — Pixar 3D", layout="wide")

st.sidebar.markdown("**Booting MuVidGen…**")
try:
    import app_real as app
except Exception as e:
    st.error(f"Failed to import app_real: {e}")
else:
    if hasattr(app, "render") and callable(app.render):
        app.render()
    elif hasattr(app, "main") and callable(app.main):
        app.main()
    else:
        st.warning(
            "app_real.py imported but did not expose a render() or main() function.\n\n"
            "Fix: expose a render() function so app.py can call it."
        )
        st.stop()
