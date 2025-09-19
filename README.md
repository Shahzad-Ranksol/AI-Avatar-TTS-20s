================================================
FILE: README.md
================================================
# AI-Avatar-TTS-20s
AI Avatar TTS 20s



================================================
FILE: AI_Avatar_TTS_20s.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# 🎯 20‑Second Talking Avatar (Open‑Source, Colab Ready)

This notebook will:
1) Take a **local 30s video** you upload  
2) **Clone** its voice (using a short reference from the video’s audio) with **Coqui XTTS v2** (open‑source)  
3) Generate a **20s text‑to‑speech** clip in that style  
4) Create a **realistic talking avatar** (head poses, eye blinks, lip-sync) from a still frame using **SadTalker**  
5) Output a **final 20s MP4** with audio + avatar

> **Ethics & Rights:** Only use voices & faces you own or have permission to use. Respect local laws and platform policies.
"""

#@title ⛏️ Check GPU
import torch, platform
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Python:", platform.python_version())

#@title 📦 Install dependencies (takes ~2–5 minutes)
!sudo apt-get -qq update
!sudo apt-get -qq install -y ffmpeg git-lfs
!git lfs install

# Core python deps
!pip -q install --upgrade pip
!pip -q install TTS==0.22.0  # Coqui TTS with XTTS v2
# Pin numpy/numba to compatible releases so librosa installs cleanly
!pip -q install moviepy==1.0.3 librosa==0.10.1 soundfile==0.12.1 numpy==1.23.5 numba==0.57.1
!pip -q install opencv-python-headless==4.8.0.74
!pip -q install tqdm==4.66.5

# Confirm the pinned versions (librosa depends on numba, which constrains numpy)
import numpy, numba, librosa
print("numpy:", numpy.__version__)
print("numba:", numba.__version__)
print("librosa:", librosa.__version__)

# SadTalker (talking head)
!git clone -q https://github.com/OpenTalker/SadTalker.git
%cd SadTalker
!pip -q install -r requirements.txt

# Download SadTalker pretrained models
!bash scripts/download_models.sh

# Return to root
%cd /content

#@title ⬆️ Upload your 30‑second source video
#@markdown Upload a video with a clear single speaker (frontal if possible). Duration can be >30s; we will trim automatically.
from google.colab import files
uploaded = files.upload()  # pick your local video
assert len(uploaded) > 0, "No file uploaded"
src_video_path = list(uploaded.keys())[0]
print("Uploaded:", src_video_path)

#@title 🎬 Extract mid‑frame portrait and a clean 10‑second voice reference
#@markdown We pick a mid-frame as the avatar's face; and extract a 10s audio slice for cloning.
import subprocess, json, os, cv2
import numpy as np

os.makedirs("work", exist_ok=True)
mid_frame_path = "work/reference_face.jpg"
ref_wav_path = "work/ref_voice_10s.wav"
trimmed_src_path = "work/source_30s.mp4"

# 5a) Normalize/trim to 30s (from 00:00)
!ffmpeg -y -i "$src_video_path" -t 30 -r 25 -vf "scale=768:-2" -an "$trimmed_src_path" -loglevel error
print("Normalized/trimmed:", trimmed_src_path)

# 5b) Extract audio (full) then cut a clean 10s window (starting at 5s to avoid intros)
!ffmpeg -y -i "$src_video_path" -vn -ac 1 -ar 16000 "work/full.wav" -loglevel error
!ffmpeg -y -ss 5 -t 10 -i "work/full.wav" -ac 1 -ar 16000 "$ref_wav_path" -loglevel error
print("Ref voice:", ref_wav_path)

# 5c) Grab a middle frame as portrait
# Use ffprobe to find duration and pick mid timestamp
import subprocess, json, math
dur_json = subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json','-i',src_video_path])
duration = float(json.loads(dur_json)['format']['duration'])
mid_ts = max(0.0, duration/2.0)

# Extract a frame image
!ffmpeg -y -ss {mid_ts} -i "$src_video_path" -frames:v 1 "work/raw_mid.jpg" -loglevel error

# Face crop (simple center-crop fallback if face not detected)
img = cv2.imread("work/raw_mid.jpg")
h,w = img.shape[:2]
# center crop to square
side = min(h,w)
y0 = (h-side)//2; x0 = (w-side)//2
square = img[y0:y0+side, x0:x0+side]
# Resize to 512 for SadTalker
square = cv2.resize(square, (512,512), interpolation=cv2.INTER_AREA)
cv2.imwrite(mid_frame_path, square)
print("Reference face saved to:", mid_frame_path)

#@title 🗣️ Generate 20‑second TTS in cloned style (Coqui XTTS v2)
#@markdown Enter the exact text (≈45–55 words for ~20s at natural pace).
from TTS.api import TTS
import soundfile as sf
import numpy as np

tts_text = "Hello! This is a demo of an open source AI avatar. My voice was cloned from the short reference, and I am speaking naturally with expressive prosody and clear articulation. Welcome to your fully local, privacy friendly pipeline powered by open models." #@param {type:"string"}
out_tts_wav = "work/tts_20s.wav"

# Load multilingual XTTS v2
avail = TTS.list_models()
model_name = "tts_models/multilingual/multi-dataset/xtts_v2" if any("xtts_v2" in m for m in avail) else avail[0]
tts = TTS(model_name)

# Synthesize (language auto; set to 'en' unless you need another)
audio = tts.tts(text=tts_text, speaker_wav="work/ref_voice_10s.wav", language="en")
sf.write(out_tts_wav, np.array(audio), 24000)
print("TTS saved:", out_tts_wav)

# Optional: force‑trim/pad to ~20s
!ffmpeg -y -i "$out_tts_wav" -t 20 -ac 1 -ar 24000 "work/tts_20s_fixed.wav" -loglevel error
out_tts_wav = "work/tts_20s_fixed.wav"
print("TTS (20s) saved:", out_tts_wav)

#@title 🧑‍🎤 Generate talking avatar with SadTalker
#@markdown Produces a head-posed, expressive talking head driven by TTS audio.
import os, subprocess, shlex, pathlib
img = "work/reference_face.jpg"
aud = "work/tts_20s_fixed.wav"
out_dir = "work/sadtalker_out"
os.makedirs(out_dir, exist_ok=True)

cmd = f"""python /content/SadTalker/inference.py   --driven_audio "{aud}"   --source_image "{img}"   --enhancer gfpgan   --preprocess full   --result_dir "{out_dir}"   --still   --expression_scale 1.2   --pose_scale 1.1   --ref_eyeblink "video"   --ref_pose "audio"   --use_idle   --batch_size 1
"""
print("Running:", cmd)
ret = subprocess.run(cmd, shell=True, check=False)
print("Return code:", ret.returncode)

# Find the generated video
from glob import glob
candidates = sorted(glob(os.path.join(out_dir, "*.mp4")), key=os.path.getmtime)
assert candidates, "No output video produced by SadTalker."
gen_video = candidates[-1]
print("SadTalker video:", gen_video)

#@title 🎚️ Remux clean audio & export final 20s MP4
from glob import glob
import os

gen_video = sorted(glob("work/sadtalker_out/*.mp4"), key=os.path.getmtime)[-1]
final_mp4 = "AI_Avatar_20s.mp4"

# Replace audio track with our clean tts_20s_fixed.wav at 24kHz; set fps 25
!ffmpeg -y -i "$gen_video" -i "work/tts_20s_fixed.wav" -map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 25 -c:a aac -b:a 128k -shortest "$final_mp4" -loglevel error
print("Final video:", final_mp4)

"""
## ✅ Outputs
- `AI_Avatar_20s.mp4` — final 20s talking avatar video  
- `work/reference_face.jpg` — portrait used by SadTalker  
- `work/tts_20s_fixed.wav` — cloned‑style TTS
"""

#@title ⬇️ Download final video
from google.colab import files
files.download("AI_Avatar_20s.mp4")

"""
## 🛠️ Troubleshooting

- **No GPU** → Runtime ▸ Change runtime type ▸ **T4/A100 GPU**.  
- **SadTalker "No output video"** → Use a **clear portrait**; try another frame; set `--preprocess crop` and `--still` flags.  
- **Voice not close** → Ensure the 10s reference has **clean speech**; extend `-t 12` in the ref‑audio cut.  
- **Laggy lips** → Lower `--pose_scale` to `1.0`, try `--expression_scale 1.0`.  
- **Choppy audio** → Re‑encode: `ffmpeg -i in.wav -ac 1 -ar 24000 out.wav`.
"""


