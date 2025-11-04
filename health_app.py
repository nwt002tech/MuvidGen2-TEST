import os, streamlit as st, sys, platform
st.title("MuVidGen Health+ (Tokens & Env)")
st.code({
    "python": sys.version,
    "platform": platform.platform(),
})
danger_keys = [k for k in os.environ.keys() if "HF" in k.upper() or "HUGGINGFACE" in k.upper()]
st.subheader("Env keys that look like HF tokens/dirs")
st.write(sorted(danger_keys))
st.subheader("Secrets snapshot (names only)")
try:
    st.write(sorted(list(st.secrets.keys())))
except Exception as e:
    st.write("No secrets or cannot read secrets names.", e)
st.caption("If any token-like keys exist (HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, etc.), clear them in App → Settings → Secrets and App → Settings → Advanced → Environment variables.")
