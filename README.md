# 🎞 FlowSculpt

A fast interactive tool for drawing custom timewarp curves on video playback — with audio sync, waveform/spectrogram display, and export to MP4.

---

## ⚙️ Setup (Windows & Linux)

### ✅ 1. Clone & enter project

```bash
git clone https://github.com/nikto123/imgfader
cd flowsculpt
```

### ✅ 2. Create & activate virtual environment

#### Windows (CMD or PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### ✅ 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pygame numpy opencv-python soundfile sounddevice
```

---

## 🔧 FFmpeg (required for audio decoding + export)

### Windows

1. Download FFmpeg static build: https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your **system PATH**  
   (System Properties → Environment Variables → Path → Add new)

### Linux

```bash
sudo apt install ffmpeg
```

---

## 🚀 Running the App

```bash
python flowsculptor_app.py
```

Choose:
- Load saved project (`.json`)  
  — or —
- Select **video file** and **audio file**

---

## 🎮 Controls & Features

### ▶️ Playback
- **SPACE** — Play / pause
- **Right Click** — Jump to clicked time

### 🖌️ Draw frame curve
- **Ctrl + Left Click** — Add polyline segments to time curve
- **Left Click + Drag** — Directly paint timewarp values (lower pane)
- **Shift + Hover** — Preview frame at that curve index

### ⏺️ Record mode
- **R** — Live record mouse curve while playing

### 📈 Zoom & Pan
- **Mouse Wheel** — Zoom in/out
- **Middle Click / Shift + Left Click** — Pan view
- **Backspace** — Reset zoom & pan

### 🎚️ Spectrogram toggle
- **S** — Toggle between waveform and spectrogram

### 🔁 Undo
- **Ctrl + Z** — Undo last curve change

### 💾 Project I/O
- **Ctrl + S** — Save `.json` project (includes paths + curve)
- **Ctrl + L** — Load `.json` project

### 🎥 Export to video
- **Ctrl + X** — Export timewarped video with synced audio
  - Includes progress bar & final merged `.mp4`

---

## 🧠 Tips

- 📦 Use `.wav` or `.flac` for audio (instant loading)
- ⚡ Transcode videos to all-keyframe `.mp4` for faster scrubbing:

```bash
ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 18 -g 1 -an output_scrub.mp4
```

---

## 📂 Project Format

Saved `.json`:
```json
{
  "video_path": "path/to/video.mp4",
  "audio_path": "path/to/audio.wav",
  "curve": [ ... 2000 timewarp values ... ]
}
```

---

## ✅ Compatibility

- Tested on **Python 3.10–3.12**
- ⚠️ Python 3.13 not recommended (threading/subprocess crash with ffmpeg)

---

## 📃 License

MIT (or your preferred license)

