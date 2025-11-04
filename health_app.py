import sys, platform, subprocess, streamlit as st

st.title("MuVidGen Health+ (Diagnostics)")
st.code({"python": sys.version, "platform": platform.platform()})
mods = {}
for m in ["numpy","PIL","moviepy","imageio_ffmpeg","requests","streamlit","gradio_client"]:
    try:
        mod = __import__(m if m != "PIL" else "PIL.Image")
        ver = getattr(mod, "__version__", "ok")
        mods[m] = ["✅", ver]
    except Exception as e:
        mods[m] = ["❌", str(e)]
st.subheader("Python packages"); st.code(mods)

try:
    out = subprocess.check_output(["ffmpeg","-version"], stderr=subprocess.STDOUT, text=True)[:400]
    st.subheader("FFmpeg"); st.code({"ffmpeg":"✅","detail":out})
except Exception as e:
    st.subheader("FFmpeg"); st.code({"ffmpeg":"❌","detail":str(e)})
