
# --- Pillow 10+ backward-compatibility shim ---
from PIL import Image as _Image
if not hasattr(_Image, "ANTIALIAS"):
    try:
        _Image.ANTIALIAS = _Image.Resampling.LANCZOS
    except Exception:
        _Image.ANTIALIAS = 1

import os, re, io, uuid, time, tempfile, logging, sys, subprocess
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Logging
logger = logging.getLogger("muvidgen"); logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout); _handler.setLevel(logging.INFO)
logger.handlers = [_handler]
def log(msg: str):
    logger.info(msg)
    try: st.sidebar.write(f"🧭 {msg}")
    except Exception: pass

# --------------------------- Lyrics helpers ----------------------------
_STOP = set(x.strip() for x in (
    "a an and are as at be by for from has he her his i in is it its of on that the they this to was were will with "
    "me my our ours us we youre you're im ive ill ya yo oh uh um your ur nah yeah yes no not do does did done have "
    "has had am been being or if so than then there here when where why how what which who whom about into through "
    "over under again further once"
).split())

def simple_keywords(text: str, top_k: int = 12) -> List[str]:
    words = re.findall(r"[A-Za-z]{3,}", (text or "").lower())
    words = [w for w in words if w not in _STOP]
    if not words: return []
    from collections import Counter
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]

def chunk_lines(lyrics: str) -> List[str]:
    lines = [ln.strip() for ln in (lyrics or "").split("\n") if ln.strip()]
    return lines or ["(instrumental)"]

# --------------------------- Audio analysis ----------------------------
@dataclass
class AudioAnalysis:
    duration: float
    tempo_bpm: float
    beat_times: List[float]

def _st_energy_onsets(y: np.ndarray, sr: int, frame_len_s=0.046, hop_s=0.023):
    if y.ndim > 1: y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
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
        energies = np.convolve(energies, np.ones(5, dtype=np.float32)/5, mode="same")
    onset = np.maximum(0, np.diff(energies, prepend=energies[:1]))
    return onset, hop / sr

def _peak_pick(onset: np.ndarray, hop_sec: float) -> List[float]:
    if onset.size == 0: return []
    thr = float(np.percentile(onset, 80) * 0.7)
    peaks = []; min_gap = int(max(1, 0.2 / max(hop_sec, 1e-6))); last = -min_gap
    for i, val in enumerate(onset):
        if val > thr and (i - last) >= min_gap:
            window = onset[max(0,i-2):i+3]
            if val >= window.max():
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

def _ffmpeg_pcm(path: str, sr: int = 44100) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0 or out is None:
        raise RuntimeError(f"FFmpeg decode failed: {err.decode('utf-8', 'ignore')}")
    return np.frombuffer(out, dtype=np.float32)

def analyze_audio(file_bytes: bytes, sr_target: int = 44100) -> AudioAnalysis:
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush(); path = tmp.name

    duration = 0.0; y = None
    try:
        from moviepy.editor import AudioFileClip
        log("Audio: trying MoviePy loader")
        with AudioFileClip(path) as ac:
            duration = float(ac.duration or 0.0)
            try:
                y_arr = ac.to_soundarray(fps=sr_target)
                if y_arr is not None and len(y_arr) > 0:
                    y = y_arr if y_arr.ndim == 1 else y_arr.mean(axis=1)
            except Exception as e:
                log(f"MoviePy to_soundarray failed: {e}")
    except Exception as e:
        log(f"MoviePy loader failed: {e}")

    if y is None or len(y) == 0:
        log("Audio: using FFmpeg pipe fallback")
        y = _ffmpeg_pcm(path, sr=sr_target)
        duration = float(len(y) / float(sr_target))

    onset, hop_sec = _st_energy_onsets(y, sr_target)
    peaks = _peak_pick(onset, hop_sec)
    if len(peaks) < 4 and duration > 4 and onset.size > 32:
        f = np.fft.rfft(onset - onset.mean())
        acf = np.fft.irfft(f * np.conj(f))
        acf[:int(0.2/hop_sec)] = 0
        lag = int(np.argmax(acf[:int(2.0/hop_sec)]))
        est_int = max(0.5, min(1.0, lag * hop_sec)); beat_times = list(np.arange(0, duration, est_int))
    else:
        beat_times = peaks
    bpm = _estimate_bpm(beat_times)
    log(f"Audio analyzed: duration={duration:.2f}s bpm≈{bpm:.1f} beats={len(beat_times)}")
    return AudioAnalysis(duration=duration, tempo_bpm=bpm, beat_times=[float(t) for t in beat_times])

# --------------------------- FreeAnim via Hugging Face Spaces ----------------------------
from gradio_client import Client
try:
    from gradio_client import file as gc_file
except Exception:
    def gc_file(path): return path  # fallback

def _space_client(space_id: str) -> Optional[Client]:
    # Force anonymous access: do not send any token (fixes 401 when a bad token exists in env)
    try:
        return Client(space_id, hf_token=None)
    except Exception as e:
        st.warning(f"Could not connect to Space '{space_id}': {e}")
        return None

def _first_predict_endpoint(meta: dict, want_image: bool=False, want_text: bool=False) -> Optional[str]:
    apis = meta.get("endpoints", []) or meta.get("named_endpoints", [])
    if isinstance(apis, dict):
        for name, schema in apis.items():
            inputs = (schema.get("inputs") if isinstance(schema.get("inputs"), list) else []) if isinstance(schema, dict) else []
            types = [i.get("type","") for i in inputs if isinstance(i, dict)]
            if want_text and any(t in ("textbox","text","string") for t in types): return name
            if want_image and any(t in ("image","filepath","numpy","pil") for t in types): return name
        return None
    for ep in apis:
        if not isinstance(ep, dict): continue
        schema = ep.get("schema") or ep.get("config") or ep
        name   = ep.get("endpoint") or ep.get("name") or ep.get("path") or "/predict"
        inputs = (schema.get("inputs") if isinstance(schema.get("inputs"), list) else []) if isinstance(schema, dict) else []
        types = [i.get("type","") for i in inputs if isinstance(i, dict)]
        if want_text and any(t in ("textbox","text","string") for t in types): return name
        if want_image and any(t in ("image","filepath","numpy","pil") for t in types): return name
    return None

ALT_T2V_SPACES = [
    "THUDM/CogVideoX-5b",
    "ali-vilab/InstructVideo",
    "kandinsky-community/video-kandinsky",
]
ALT_I2V_SPACES = [
    "stabilityai/stable-video-diffusion-img2vid",
    "ByteDance/AnimateDiff-Lightning",
]

def _try_spaces_t2v(prompt: str, seconds: int, space_ids: List[str]) -> Optional[bytes]:
    for sid in space_ids:
        c = _space_client(sid)
        if not c: continue
        try:
            meta = c.view_api(all_endpoints=True)
        except Exception:
            meta = {}
        endpoint = _first_predict_endpoint(meta if isinstance(meta, dict) else {}, want_text=True) or "/predict"
        candidates = [
            {"prompt": prompt, "num_frames": min(125, max(50, seconds*25)), "fps": 25, "seed": 0},
            {"text": prompt, "num_frames": min(125, max(50, seconds*25)), "fps": 20, "seed": 0},
            {"prompt": prompt},
        ]
        for payload in candidates:
            try:
                res = c.predict(api_name=endpoint, **payload)
                vid_path = None
                if isinstance(res, (list, tuple)):
                    for x in res:
                        if isinstance(x, str) and x.lower().endswith((".mp4",".webm",".mov")):
                            vid_path = x; break
                elif isinstance(res, dict):
                    for v in res.values():
                        if isinstance(v, str) and v.lower().endswith((".mp4",".webm",".mov")):
                            vid_path = v; break
                elif isinstance(res, str):
                    if res.lower().endswith((".mp4",".webm",".mov")):
                        vid_path = res
                if vid_path:
                    local = c.download(vid_path)
                    with open(local, "rb") as f: return f.read()
            except Exception:
                continue
    return None

def _try_spaces_i2v(image: Image.Image, seconds: int, space_ids: List[str]) -> Optional[bytes]:
    tmp_img = os.path.join(tempfile.gettempdir(), f"svd_{uuid.uuid4().hex}.png")
    image.save(tmp_img)
    for sid in space_ids:
        c = _space_client(sid)
        if not c: continue
        try:
            meta = c.view_api(all_endpoints=True)
        except Exception:
            meta = {}
        endpoint = _first_predict_endpoint(meta if isinstance(meta, dict) else {}, want_image=True) or "/predict"
        candidates = [
            {"image": gc_file(tmp_img), "fps": 12, "motion_bucket_id": 64, "seed": 0},
            {"image": gc_file(tmp_img)},
        ]
        for payload in candidates:
            try:
                res = c.predict(api_name=endpoint, **payload)
                vid_path = None
                if isinstance(res, (list, tuple)):
                    for x in res:
                        if isinstance(x, str) and x.lower().endswith((".mp4",".webm",".mov")):
                            vid_path = x; break
                elif isinstance(res, dict):
                    for v in res.values():
                        if isinstance(v, str) and v.lower().endswith((".mp4",".webm",".mov")):
                            vid_path = v; break
                elif isinstance(res, str):
                    if res.lower().endswith((".mp4",".webm",".mov")):
                        vid_path = res
                if vid_path:
                    local = c.download(vid_path)
                    with open(local, "rb") as f: return f.read()
            except Exception:
                continue
    return None

def freeanim_t2v(prompt: str, seconds: int = 5, space_id: str = "THUDM/CogVideoX-5b") -> Optional[bytes]:
    return _try_spaces_t2v(prompt, seconds, [space_id] + [s for s in ALT_T2V_SPACES if s != space_id])

def freeanim_i2v(image: Image.Image, seconds: int = 4, space_id: str = "stabilityai/stable-video-diffusion-img2vid") -> Optional[bytes]:
    return _try_spaces_i2v(image, seconds, [space_id] + [s for s in ALT_I2V_SPACES if s != space_id])

# --------------------------- Image fallback + Video assembly ----------------------------
def fallback_card(text: str, size=(768,512)):
    w, h = size
    bg = Image.new("RGB", size, (16, 22, 54))
    overlay = Image.new("RGBA", size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    for i in range(h):
        alpha = int(160 * (1 - i / h))
        od.line([(0,i),(w,i)], fill=(79,70,229,alpha))
    bg.paste(overlay, (0,0), overlay)
    draw = ImageDraw.Draw(bg)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
        body_font  = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        title_font = body_font = ImageFont.load_default()
    title = "Pixar-inspired Scene"
    tw = draw.textlength(title, font=title_font)
    draw.text(((w - tw)//2, 20), title, fill=(255,255,255), font=title_font)
    # wrap
    words = text.split(); lines = []; cur = ""
    for wd in words:
        t2 = wd if not cur else f"{cur} {wd}"
        if draw.textlength(t2, font=body_font) < (w - 80): cur = t2
        else: lines.append(cur); cur = wd
    if cur: lines.append(cur)
    y = 80
    for ln in lines[:5]:
        draw.text((40, y), ln, fill=(230,235,255), font=body_font); y += 28
    return bg

def subtitle_clip(text: str, width: int, height: int, duration: float):
    from PIL import ImageDraw, ImageFont
    from moviepy.editor import ImageClip
    pad_h = 20; box_h = 110
    img = Image.new("RGBA", (width, height), (0,0,0,0)); draw = ImageDraw.Draw(img)
    draw.rectangle([(0, height - box_h - pad_h), (width, height)], fill=(0,0,0,130))
    try: font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except: font = ImageFont.load_default()
    words = text.split(); lines = []; cur = ""
    for wd in words:
        t2 = wd if not cur else f"{cur} {wd}"
        if draw.textlength(t2, font=font) < (width - 120): cur = t2
        else: lines.append(cur); cur = wd
    if cur: lines.append(cur)
    y = height - box_h - pad_h + 20
    from moviepy.editor import ImageClip
    for ln in lines[:3]:
        tw = draw.textlength(ln, font=font)
        draw.text(((width - tw)//2, y), ln, font=font, fill=(255,255,255,255)); y += 42
    return ImageClip(np.array(img)).set_duration(duration).set_position(("center","center"))

def ken_burns(image, duration: float, zoom: float = 1.1):
    from moviepy.editor import ImageClip
    from moviepy.video.fx.all import resize
    clip = ImageClip(np.array(image)).set_duration(duration)
    return clip.fx(resize, lambda t: 1 + (zoom - 1) * (t / max(duration, 1e-6)))

def assemble_video_from_clips(clip_paths: List[str], audio_path: str, total_duration: float) -> str:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
    subclips = [VideoFileClip(p) for p in clip_paths]
    stitched = concatenate_videoclips(subclips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, total_duration)
    final = stitched.set_audio(audio)
    out_path = os.path.join(tempfile.gettempdir(), f"muvidgen_freeanim_{uuid.uuid4().hex}.mp4")
    final.write_videofile(out_path, fps=25, codec="libx264", audio_codec="aac",
                          temp_audiofile=os.path.join(tempfile.gettempdir(), "temp-audio.m4a"),
                          remove_temp=True, threads=1, verbose=False, logger=None)
    for c in subclips: c.close()
    final.close(); audio.close()
    return out_path

def assemble_video_from_images(images: List[Image.Image], lines: List[str], audio_path: str, beat_times: List[float], total_duration: float, scene_pad: float = 0.15):
    from moviepy.editor import AudioFileClip, CompositeVideoClip, concatenate_videoclips
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
        clips.append(CompositeVideoClip([img_clip, sub_clip], size=(width, height)).set_duration(seg_duration))
        t_cursor = seg_end
    current_sum = sum([c.duration for c in clips])
    if current_sum < total_duration and clips:
        clips[-1] = clips[-1].set_duration(clips[-1].duration + (total_duration - current_sum))
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, total_duration)
    final = video.set_audio(audio)
    out_path = os.path.join(tempfile.gettempdir(), f"muvidgen_freeanim_{uuid.uuid4().hex}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac",
                          temp_audiofile=os.path.join(tempfile.gettempdir(), "temp-audio.m4a"),
                          remove_temp=True, threads=1, verbose=False, logger=None)
    final.close(); video.close(); audio.close()
    return out_path

# --------------------------- UI ----------------------------
def human_seconds(sec: float) -> str:
    m = int(sec // 60); s = int(round(sec % 60)); return f"{m}:{s:02d}"

def render():
    st.title("🎬 MuVidGen — Free Animated Edition (HF Spaces, No Keys)")
    with st.sidebar:
        st.header("Generation Backends")
        use_free_anim = st.toggle("Use FREE animated Spaces (real motion)", value=True)
        backend = st.selectbox("Animation mode", ["Text→Video (CogVideoX Space)", "Image→Video (SVD Space)"])
        t2v_space = st.text_input("T2V Space", value="THUDM/CogVideoX-5b")
        i2v_space = st.text_input("I2V Space", value="stabilityai/stable-video-diffusion-img2vid")
        st.caption("If you see 401 Unauthorized, remove any Hugging Face token from your Streamlit secrets/env. This app forces anonymous access.")

        st.divider(); st.subheader("Visual Style (for still images fallback)")
        visual_style = st.text_area("Base style", value="Pixar-inspired 3D animation, kid-friendly characters, soft lighting, GI/SSS, PBR materials, cinematic DOF", height=80)
        characters = st.text_input("Characters", value="Cute anthropomorphic letters and happy kids dancing; big expressive eyes, rounded shapes")
        palette = st.text_input("Palette", value="pastels with pops of yellow, teal, magenta; warm key, cool fill")
        camera = st.text_input("Camera", value="35mm lens, gentle dolly, slow zoom-in, soft rim light")
        model_hint = st.text_input("Hint", value="family-safe, cheerful, consistent stylization")
        character_bible = st.text_area("Character Bible", value="Green letter F (forward-hook top curve), friendly face; classroom setting.", height=80)
        negative_hint = st.text_input("Avoid", value="no text, no watermark, no logo, no scary, no copyrighted characters")

    st.markdown("#### 1) Upload audio and paste lyrics")
    audio_file = st.file_uploader("Upload song (MP3/WAV/M4A/AAC)", type=["mp3","wav","m4a","aac"])
    lyrics = st.text_area("Paste full song lyrics", height=200, placeholder="One lyric line per scene.")

    colA, colB = st.columns(2)
    if "analysis" not in st.session_state: st.session_state.analysis = None
    if "story" not in st.session_state: st.session_state.story = None

    with colA:
        if st.button("Analyze Audio + Lyrics", type="secondary", use_container_width=True):
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
        ready = st.session_state.get("analysis") is not None
        render_btn = st.button("Generate Full Music Video", type="primary", use_container_width=True, disabled=not ready)

    def fallback_image_for_line(line: str) -> Image.Image:
        kw = st.session_state.story["keywords"]
        style = st.session_state.style
        text = f"{style['visual_style']}\n{style['characters']}\nLyric: {line}\nMotifs: {', '.join(kw[:5])}\n{style['camera']}"
        return fallback_card(text)

    if render_btn:
        lines = st.session_state.story["lines"]
        beats = st.session_state.analysis["beat_times"]; duration = st.session_state.analysis["duration"]
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(st.session_state.audio_name)[-1], delete=False) as atmp:
            atmp.write(st.session_state.audio_bytes); atmp.flush(); audio_path = atmp.name

        clip_paths: List[str] = []
        images: List[Image.Image] = []
        per_scene = max(4, min(10, int(duration/max(1,len(lines)))))  # 4-10s

        st.markdown("#### 3) Scene Generation (Free Spaces or fallback)")
        prog = st.progress(0)
        for i, line in enumerate(lines):
            mp4_bytes: Optional[bytes] = None
            if use_free_anim:
                if backend.startswith("Text→Video"):
                    mp4_bytes = freeanim_t2v(line, seconds=per_scene, space_id=t2v_space)
                else:
                    still = fallback_image_for_line(line)
                    mp4_bytes = freeanim_i2v(still, seconds=per_scene, space_id=i2v_space)
            if mp4_bytes:
                p = os.path.join(tempfile.gettempdir(), f"freeanim_{uuid.uuid4().hex}.mp4")
                with open(p, "wb") as f: f.write(mp4_bytes)
                clip_paths.append(p)
            else:
                images.append(fallback_image_for_line(line))
            prog.progress(int(((i+1)/max(1,len(lines)))*100))
            if (i+1) % max(1, len(lines)//3) == 0 or (i+1) == len(lines):
                st.write(f"Generated scene {i+1} / {len(lines)}")
        st.success("Scenes created ✅")

        st.markdown("#### 4) Rendering Video")
        if clip_paths:
            out_path = assemble_video_from_clips(clip_paths, audio_path, total_duration=duration)
            vid_dur = duration
        else:
            out_path = assemble_video_from_images(images, lines, audio_path, beats, total_duration=duration)
            vid_dur = duration
        st.success(f"Video ready ✅  (length: {human_seconds(vid_dur)})")
        with open(out_path, "rb") as f: video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("⬇️ Download MP4", data=video_bytes, file_name="muvidgen_freeanim_video.mp4", mime="video/mp4", use_container_width=True)
