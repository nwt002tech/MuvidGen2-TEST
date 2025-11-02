import os, re, io, uuid, time, tempfile, logging, sys
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from collections import Counter

print("BOOT: enter app_real.py", flush=True)
st.set_page_config(page_title="MuVidGen – Pixar 3D", layout="wide")
st.markdown("""
<style>
button[kind=\"primary\"] { font-size: 1.05rem; padding: 0.7rem 1rem; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
textarea, input, select { font-size: 1rem !important; }
@media (max-width: 420px) { .stActionButton { transform: scale(1.05); }
}
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger("muvidgen"); logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout); handler.setLevel(logging.INFO)
logger.handlers = [handler]
def log(msg: str):
    logger.info(msg)
    try: st.sidebar.write(f"🧭 {msg}")
    except Exception: pass

st.sidebar.write("🔧 Boot A: page configured")

_STOP = set(x.strip() for x in """
a an and are as at be by for from has he her his i in is it its of on that the they this to was were will with
me my our ours us we youre you're im ive ill ya yo oh uh um your ur nah yeah yes no not do does did done have
has had am been being or if so than then there here when where why how what which who whom about into through
over under again further once
""".split())

def simple_keywords(text: str, top_k: int = 12) -> List[str]:
    words = re.findall(r"[A-Za-z]{3,}", (text or "").lower())
    words = [w for w in words if w not in _STOP]
    if not words: return []
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]

def chunk_lines(lyrics: str) -> List[str]:
    lines = [ln.strip() for ln in (lyrics or "").split("\n") if ln.strip()]
    return lines or ["(instrumental)"]

@dataclass
class AudioAnalysis:
    duration: float
    tempo_bpm: float
    beat_times: List[float]

def _st_energy_onsets(y: np.ndarray, sr: int, frame_len_s=0.046, hop_s=0.023):
    if y.ndim > 1: y = y.mean(axis=1)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9); y = y / peak
    frame_len = max(1, int(frame_len_s * sr)); hop = max(1, int(hop_s * sr))
    n_frames = 1 + max(0, (len(y) - frame_len) // hop)
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop; e = s + frame_len
        seg = y[s:e] if e <= len(y) else y[s:]
        if seg.size == 0: seg = np.zeros(1, np.float32)
        energies[i] = float(np.sqrt(np.mean(seg**2)))
    if n_frames >= 5:
        energies = np.convolve(energies, np.ones(5)/5, mode="same")
    onset = np.maximum(0, np.diff(energies, prepend=energies[:1]))
    return onset, hop / sr

def _peak_pick(onset: np.ndarray, hop_sec: float) -> List[float]:
    if onset.size == 0: return []
    thr = float(np.percentile(onset, 80) * 0.7)
    peaks = []; min_gap = int(max(1, 0.2 / max(hop_sec, 1e-6))); last = -min_gap
    for i, val in enumerate(onset):
        if val > thr and (i - last) >= min_gap:
            if val >= onset[max(0,i-2):i+3].max():
                peaks.append(i); last = i
    return [float(p * hop_sec) for p in peaks]

def _estimate_bpm(beat_times: List[float]) -> float:
    if len(beat_times) < 2: return 0.0
    intervals = np.diff(np.array(beat_times, dtype=np.float32))
    if intervals.size == 0: return 0.0
    intervals = np.clip(intervals, 0.2, 2.0)
    med = float(np.median(intervals))
    if med <= 0: return 0.0
    bpm = 60.0 / med
    if bpm < 60: bpm *= 2
    if bpm > 200: bpm *= 0.5
    return float(bpm)

def analyze_audio(file_bytes: bytes, sr_target: int = 44100) -> AudioAnalysis:
    log("Audio: using MoviePy (FFmpeg)")
    from moviepy.editor import AudioFileClip
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush(); path = tmp.name
    with AudioFileClip(path) as ac:
        duration = float(ac.duration or 0.0)
        y = ac.to_soundarray(fps=sr_target)
    if y.ndim > 1: y = y.mean(axis=1)
    onset, hop_sec = _st_energy_onsets(y, sr_target)
    peaks = _peak_pick(onset, hop_sec)
    if len(peaks) < 4 and duration > 4 and onset.size > 32:
        f = np.fft.rfft(onset - onset.mean()); acf = np.fft.irfft(f * np.conj(f))
        acf[:int(0.2/hop_sec)] = 0
        lag = int(np.argmax(acf[:int(2.0/hop_sec)]))
        est_int = max(0.5, min(1.0, lag * hop_sec)); beat_times = list(np.arange(0, duration, est_int))
    else:
        beat_times = peaks
    bpm = _estimate_bpm(beat_times)
    log(f"Audio analyzed: duration={duration:.2f}s bpm≈{bpm:.1f} beats={len(beat_times)}")
    return AudioAnalysis(duration=duration, tempo_bpm=bpm, beat_times=[float(t) for t in beat_times])

@dataclass
class StyleSettings:
    visual_style: str
    characters: str
    palette: str
    camera: str
    model_hint: str
    character_bible: str
    negative_hint: str

DEFAULT_VISUAL_STYLE = "Pixar-inspired 3D animation, stylized kid-friendly characters, soft studio lighting, GI, SSS, PBR materials, cinematic DOF, soft rim light, vibrant but clean color, wholesome mood"
DEFAULT_CHARACTERS = "Cute anthropomorphic letters and happy kids dancing; friendly faces, big expressive eyes, rounded shapes"
DEFAULT_PALETTE = "pastels with pops of yellow, teal, magenta; warm key, cool fill"
DEFAULT_CAMERA = "gentle dolly and slow zoom-in; 35mm lens equivalent"
DEFAULT_MODEL_HINT = "family-safe, cheerful, vibrant, consistent stylization"
DEFAULT_NEGATIVE = "no text, no watermark, no logo, no signature, no disfigured hands, no scary, no photoreal humans, no copyrighted characters"

def build_prompt(line: str, tags: List[str], global_kw: List[str], style: StyleSettings) -> str:
    kw_part = ", ".join(sorted(list(set((tags or []) + global_kw[:5]))))
    bible = (style.character_bible or "").strip()
    main = (f"{style.visual_style}, {style.characters}. "
            f"Scene inspired by lyric: “{line}”. Motifs: {kw_part}. "
            f"{style.camera}. Color palette: {style.palette}. "
            f"{style.model_hint}. ")
    if bible: main += f" Character bible: {bible}. "
    if style.negative_hint: main += f" Avoid: {style.negative_hint}."
    return main

def hf_txt2img(prompt: str, hf_token: str, model_id: str = "stabilityai/sdxl-turbo", width=768, height=512, steps=6, guidance=3.0):
    from PIL import Image
    import requests
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": prompt, "parameters": {"width": width, "height": height, "num_inference_steps": steps, "guidance_scale": guidance}}
        print(f"HF call → model={model_id}", flush=True)
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image/"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if r.status_code == 503:
            time.sleep(3)
            r2 = requests.post(url, headers=headers, json=payload, timeout=120)
            if r2.status_code == 200 and r2.headers.get("content-type","").startswith("image/"):
                return Image.open(io.BytesIO(r2.content)).convert("RGB")
        print(f"HF fail: status={r.status_code} ct={r.headers.get('content-type','')}", flush=True); return None
    except Exception as e:
        print(f"HF exception: {e}", flush=True); return None

def fallback_card(text: str, size=(768,512)):
    w, h = size
    bg = Image.new("RGB", size, (16, 22, 54))
    grad = Image.new("RGBA", size); gdraw = ImageDraw.Draw(grad)
    for i in range(h):
        alpha = int(180 * (1 - i / h))
        gdraw.line([(0,i),(w,i)], fill=(79,70,229,alpha))
    bg.paste(grad, (0,0), grad)
    rim = Image.new("RGBA", size, (0,0,0,0)); rdraw = ImageDraw.Draw(rim)
    rdraw.ellipse([int(-0.1*w), int(0.5*h), int(1.1*w), int(1.3*h)], fill=(255,255,255,30))
    bg.paste(rim, (0,0), rim)
    draw = ImageDraw.Draw(bg)
    def wrap(t, width=38):
        words = t.split(); lines, cur = [], []
        for w_ in words:
            cur.append(w_)
            if len(" ".join(cur)) > width:
                lines.append(" ".join(cur)); cur = []
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)
    txt = wrap(text, width=40)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28); font2 = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default(); font2 = ImageFont.load_default()
    title = "Pixar-inspired Scene"
    bbox = draw.textbbox((0,0), title, font=font); tw = bbox[2]-bbox[0]
    draw.text(((w-tw)//2, 22), title, fill=(255,255,255), font=font)
    draw.multiline_text((40, 90), txt, fill=(230,235,255), font=font2, spacing=6)
    return bg

def subtitle_clip(text: str, width: int, height: int, duration: float):
    from PIL import Image, ImageDraw, ImageFont
    from moviepy.editor import ImageClip
    pad_h = 20; box_h = 110
    img = Image.new("RGBA", (width, height), (0,0,0,0)); draw = ImageDraw.Draw(img)
    draw.rectangle([(0, height - box_h - pad_h), (width, height)], fill=(0,0,0,130))
    words = text.split(); lines, cur = [], []; max_chars = max(10, int(width/20))
    for w_ in words:
        cur.append(w_)
        if len(" ".join(cur)) > max_chars:
            lines.append(" ".join(cur)); cur = []
    if cur: lines.append(" ".join(cur))
    try: font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except: font = ImageFont.load_default()
    y = height - box_h - pad_h + 20
    for ln in lines[:3]:
        bbox = draw.textbbox((0,0), ln, font=font); tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((width - tw)//2, y), ln, font=font, fill=(255,255,255,255)); y += th + 6
    from moviepy.editor import ImageClip
    return ImageClip(np.array(img)).set_duration(duration).set_position(("center","center"))

def ken_burns(image, duration: float, zoom: float = 1.1):
    from moviepy.editor import ImageClip
    from moviepy.video.fx.all import resize
    clip = ImageClip(np.array(image)).set_duration(duration)
    return clip.fx(resize, lambda t: 1 + (zoom - 1) * (t / duration))

def assemble_video(images: List, lines: List[str], audio_path: str, beat_times: List[float], total_duration: float, scene_pad: float = 0.15):
    from moviepy.editor import AudioFileClip, CompositeVideoClip, concatenate_videoclips
    log("MoviePy render starting")
    width, height = images[0].size
    n = max(1, len(lines)); target_seg = total_duration / n
    t_cursor = 0.0; clips = []; beat_idx = 0
    for i in range(n):
        seg_end_target = t_cursor + target_seg
        while beat_idx < len(beat_times) and beat_times[beat_idx] < seg_end_target:
            beat_idx += 1
        seg_end = beat_times[beat_idx] if beat_idx < len(beat_times) else seg_end_target
        seg_duration = max(0.6, seg_end - t_cursor)
        img_clip = ken_burns(images[i % len(images)], duration=seg_duration)
        sub_clip = subtitle_clip(lines[i], width, height, max(0.6, seg_duration - scene_pad)).set_start(scene_pad/2.0)
        from moviepy.editor import CompositeVideoClip
        clips.append(CompositeVideoClip([img_clip, sub_clip], size=(width, height)).set_duration(seg_duration))
        t_cursor = seg_end
    current_sum = sum([c.duration for c in clips])
    if current_sum < total_duration and clips:
        clips[-1] = clips[-1].set_duration(clips[-1].duration + (total_duration - current_sum))
    from moviepy.editor import concatenate_videoclips, AudioFileClip
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, total_duration)
    final = video.set_audio(audio)
    out_path = os.path.join(tempfile.gettempdir(), f"muvidgen_{uuid.uuid4().hex}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac",
                          temp_audiofile=os.path.join(tempfile.gettempdir(), "temp-audio.m4a"),
                          remove_temp=True, threads=1, verbose=False, logger=None)
    final.close(); video.close(); audio.close()
    log("MoviePy render done"); return out_path, total_duration

def human_seconds(sec: float) -> str:
    m = int(sec // 60); s = int(round(sec % 60)); return f"{m}:{s:02d}"

st.sidebar.write("🔧 Boot B: imports done")
st.title("🎬 MuVidGen — Pixar-Inspired 3D (No‑Pandas)")

with st.sidebar:
    st.header("Settings")
    secrets_token = st.secrets.get("HF_TOKEN", "")
    if secrets_token: st.caption("✅ Using Hugging Face token from Streamlit secrets."); manual_token = ""
    else: manual_token = st.text_input("Hugging Face Token (optional)", type="password")
    hf_token = secrets_token or (manual_token.strip() if manual_token else "")
    model_id = st.text_input("HF Image Model ID", value="stabilityai/sdxl-turbo")
    force_fallback = st.toggle("Force fallback (skip HF calls)", value=False)

    st.divider(); st.subheader("Visual Style (Pixar-Inspired)")
    visual_style = st.text_area("Base style", value="Pixar-inspired 3D animation, stylized kid-friendly characters, soft studio lighting, GI, SSS, PBR materials, cinematic DOF, soft rim light, vibrant but clean color, wholesome mood", height=90)
    characters = st.text_input("Characters", value="Cute anthropomorphic letters and happy kids dancing; friendly faces, big expressive eyes, rounded shapes")
    palette = st.text_input("Color Palette", value="pastels with pops of yellow, teal, magenta; warm key, cool fill")
    camera = st.text_input("Camera Direction", value="gentle dolly and slow zoom-in; 35mm lens equivalent")
    model_hint = st.text_input("Model Hint", value="family-safe, cheerful, vibrant, consistent stylization")
    character_bible = st.text_area("Character Bible", value="Green letter F (forward-hook top curve), kid-safe smile, consistent eyes/mouth; rounded limbs; classroom setting.", height=90)
    negative_hint = st.text_input("Avoid / negatives", value="no text, no watermark, no logo, no signature, no disfigured hands, no scary, no photoreal humans, no copyrighted characters")

st.markdown("#### 1) Upload audio and paste lyrics")
audio_file = st.file_uploader("Upload song (MP3/WAV/M4A)", type=["mp3","wav","m4a","aac"])
lyrics = st.text_area("Paste full song lyrics", height=220, placeholder="One lyric line per scene.")

colA, colB = st.columns(2)
with colA:
    if st.button("Analyze Audio + Lyrics", type="secondary", use_container_width=True):
        log("Analyze clicked")
        if not audio_file or not lyrics.strip():
            st.error("Please upload an audio file and paste lyrics first.")
        else:
            with st.spinner("Analyzing audio…"):
                audio_bytes = audio_file.read(); ana = analyze_audio(audio_bytes)
            with st.spinner("Analyzing lyrics…"):
                lines = chunk_lines(lyrics); kw = simple_keywords(lyrics, top_k=14)
                tags = []
                for ln in lines:
                    low = re.findall(r"[A-Za-z]+", ln.lower())
                    t = [w for w in low if w in kw]
                    if any(w in low for w in ["dance","dancin","dancing","groove","move","twist","spin","clap","jump"]): t.append("dancing")
                    if any(w in low for w in ["sun","sunshine","day","bright","yellow"]): t.append("daytime")
                    if any(w in low for w in ["moon","night","stars","glow"]): t.append("night")
                    tags.append(sorted(list(set(t))))
            st.session_state.analysis = {"duration": ana.duration, "tempo_bpm": ana.tempo_bpm, "beat_times": ana.beat_times}
            st.session_state.story = {"lines": lines, "keywords": list(kw), "line_tags": tags}
            st.session_state.audio_bytes = audio_bytes; st.session_state.audio_name = audio_file.name
            st.session_state.style = {"visual_style": visual_style, "characters": characters, "palette": palette, "camera": camera, "model_hint": model_hint, "character_bible": character_bible, "negative_hint": negative_hint}
            st.success("Analysis complete ✅")
            st.write(f"**Duration:** {human_seconds(ana.duration)}  |  **Tempo:** {round(ana.tempo_bpm)} BPM  |  **Detected beats:** {len(ana.beat_times)}")
            st.write("**Top lyric keywords:**", ", ".join(kw))
with colB:
    render_ready = st.session_state.get("analysis") is not None
    render_btn = st.button("Generate Full Music Video", type="primary", use_container_width=True, disabled=not render_ready)

if render_btn:
    log("Render clicked")
    if not st.session_state.get("analysis") or not st.session_state.get("story"):
        st.error("Please run analysis first.")
    else:
        style = StyleSettings(**st.session_state.style)
        lines = st.session_state.story["lines"]; kws = st.session_state.story["keywords"]; tags = st.session_state.story["line_tags"]
        beat_times = st.session_state.analysis["beat_times"]; duration = st.session_state.analysis["duration"]
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(st.session_state.audio_name)[-1], delete=False) as atmp:
            atmp.write(st.session_state.audio_bytes); atmp.flush(); audio_path = atmp.name

        st.markdown("#### 3) Scene Generation (Pixar-Inspired 3D)")
        st.caption("Creating per-line scenes via HF Inference (or stylized fallback).")
        gen_progress = st.progress(0); images: List[Image.Image] = []; total = max(1, len(lines))
        for i, line in enumerate(lines):
            prompt = build_prompt(line, tags[i] if i < len(tags) else [], kws, style)
            img = None
            if (st.secrets.get("HF_TOKEN","") or "").strip():
                img = hf_txt2img(prompt, st.secrets["HF_TOKEN"], model_id=model_id.strip() or "stabilityai/sdxl-turbo")
            if img is None: img = fallback_card(line)
            images.append(img); gen_progress.progress(int(((i+1)/total)*100))
            if (i+1) % max(1, total//3) == 0 or (i+1) == total:
                st.image(img, caption=f"Scene {i+1}: {line[:80]}", use_column_width=True)

        st.success("Scenes ready ✅")
        st.markdown("#### 4) Rendering Video")
        out_path, vid_dur = assemble_video(images, lines, audio_path, beat_times, total_duration=duration)
        st.success(f"Video ready ✅  (length: {human_seconds(vid_dur)})")
        with open(out_path, "rb") as f: video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("⬇️ Download MP4", data=video_bytes, file_name="muvidgen_video.mp4", mime="video/mp4", use_container_width=True)
