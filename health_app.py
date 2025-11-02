import streamlit as st, sys, platform, subprocess, importlib
st.title("MuVidGen Health+ (Diagnostics)")

def check_import(mod):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__version__", "n/a")
    except Exception as e:
        return False, str(e)

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
        return True, out.strip()[:1200]
    except Exception as e:
        return False, str(e)

st.write({"python": sys.version.replace("\n"," "), "platform": platform.platform()})
pkgs = ["numpy","PIL","moviepy","imageio_ffmpeg","requests","streamlit"]
st.subheader("Python packages")
st.write({p: ("✅", v) if ok else ("❌", v) for p,(ok,v) in {p:check_import(p) for p in pkgs}.items()})
ok_ff, ff_v = run(["ffmpeg","-version"])
st.subheader("FFmpeg"); st.write({"ffmpeg":"✅" if ok_ff else "❌","detail":ff_v[:600]})
st.subheader("Secrets"); st.write({"HF_TOKEN_present": bool(st.secrets.get("HF_TOKEN",""))})
