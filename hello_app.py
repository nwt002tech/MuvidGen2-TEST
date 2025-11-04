import os, sys, pathlib, streamlit as st
st.title("Hello from MuVidGen FreeAnim v2b")
st.write({"cwd": os.getcwd(), "files_here": sorted(p.name for p in pathlib.Path('.').iterdir()), "sys.argv": sys.argv})
st.success("If you see this, your Main file is correct and the app is executing.")
