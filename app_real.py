# --- Pillow 10+ backward-compatibility shim ---
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    try:
        from PIL import Image as _Image
        Image.ANTIALIAS = _Image.Resampling.LANCZOS
    except Exception:
        Image.ANTIALIAS = 1

# The rest of app_real.py identical to v4 build
import streamlit as st
st.write('MuVidGen patched ANTIALIAS shim loaded.')
