import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="MuVidGen — Pixar 3D", layout="wide")

st.sidebar.markdown("**Booting MuVidGen…**")

try:
    import app_real as app
except Exception as e:
    st.error(f"Failed to import app_real: {e}")
else:
    if hasattr(app, "render") and callable(app.render):
        st.caption("Calling app_real.render()")
        app.render()
    elif hasattr(app, "main") and callable(app.main):
        st.caption("Calling app_real.main()")
        app.main()
    else:
        st.warning(
            "app_real.py imported but did not expose a render() or main() function.\n\n"
            "Fix: add at the bottom of app_real.py:\n\n"
            "def render():\n"
            "    # build your Streamlit UI here\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    render()\n"
        )
        st.stop()
