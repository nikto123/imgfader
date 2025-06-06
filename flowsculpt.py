import os
import sys
import json
import time
import tempfile
import threading
import subprocess

import cv2
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

from tkinter import Tk, filedialog, messagebox

# Constants
FRAME_RATE = 30
CURVE_RES = 2000
FREQ_BINS = 256
CACHE_SIZE = 100
WINDOW_MIN_HEIGHT = 200
WINDOW_MIN_WIDTH = 200
ZOOM_STEP = 1.1


class VideoScrubber:
    def __init__(self):
        pygame.init()
        self.wave_env = None
        self.spec_matrix = None
        self.spec_surf0 = None
        self.frame_curve = np.full(CURVE_RES, 0.5, dtype=np.float32)
        self.frame_cache = {}
        self.undo_stack = []

        self.spec_mode = False
        self.record_mode = False
        self.playing = True
        self.running = True

        self.playback_time = 0.0
        self.divider_pos = 520

        # Zoom factors and offsets (world units)
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.offset_x = 0.0  # horizontal pan
        self.offset_y = 0.0  # vertical pan

        self.divider_dragging = False
        self.panning = False
        self.zooming = False

        # For polyline drawing (ctrl + click)
        self.ctrl_down = False
        self.line_pts = []

        self.hover_index = None
        self.hovering = False

        self.last_mouse_idx = None
        self.idx_prev = None
        self.dur_prev = None

        # Tkinter setup for file dialogs
        self.root = Tk()
        self.root.withdraw()
        self.root.attributes("-topmost", True)

        # Initialize screen placeholders
        self.screen = None
        self.window_width = 0
        self.window_height = 0

        # Launch and load media
        self.VIDEO_PATH, self.AUDIO_PATH = self.launcher()
        self.load_media()

    def launcher(self):
        choice = messagebox.askquestion("Video Scrubber", "Load existing project?")
        if choice == "yes":
            path = filedialog.askopenfilename(
                title="Load project", filetypes=[("JSON files", "*.json")]
            )
            if not path:
                sys.exit(0)
            with open(path, "r") as f:
                data = json.load(f)
            curve = np.array(data["curve"][:CURVE_RES], dtype=np.float32)
            if curve.size < CURVE_RES:
                curve = np.pad(curve, (0, CURVE_RES - curve.size), constant_values=0.5)
            self.frame_curve = curve
            return data["video_path"], data["audio_path"]
        else:
            video_path = filedialog.askopenfilename(
                title="Select video file",
                filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi")],
            )
            audio_path = filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[("Audio files", "*.wav *.flac *.ogg *.mp3")],
            )
            if not video_path or not audio_path:
                sys.exit(0)
            return video_path, audio_path

    def load_audio(self, path):
        try:
            data, sr = sf.read(path)
            return data, sr
        except RuntimeError:
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmpf.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, tmpf.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            data, sr = sf.read(tmpf.name)
            os.unlink(tmpf.name)
            return data, sr

    def load_media(self):
        # Load audio
        self.audio_data, self.audio_sr = self.load_audio(self.AUDIO_PATH)
        self.audio_duration = len(self.audio_data) / self.audio_sr
        self.mono_audio = (
            self.audio_data.mean(axis=1)
            if self.audio_data.ndim > 1
            else self.audio_data
        )

        # Load video
        self.cap = cv2.VideoCapture(self.VIDEO_PATH)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or FRAME_RATE
        self.video_duration = self.total_frames / self.video_fps

        # Precompute waveform envelope
        self.wave_env = np.zeros(CURVE_RES, dtype=np.float32)
        segment = len(self.mono_audio) // CURVE_RES
        for i in range(CURVE_RES):
            start = i * segment
            end = start + segment
            chunk = self.mono_audio[start:end]
            self.wave_env[i] = (
                max(abs(chunk.max()), abs(chunk.min())) if chunk.size else 0.0
            )

        # Precompute spectrogram matrix
        self.spec_matrix = np.zeros((FREQ_BINS, CURVE_RES), dtype=np.float32)
        seg_len = max(1, len(self.mono_audio) // CURVE_RES)
        for i in range(CURVE_RES):
            start = i * seg_len
            end = start + seg_len
            chunk = self.mono_audio[start:end]
            if chunk.size < seg_len:
                chunk = np.pad(chunk, (0, seg_len - chunk.size), mode="constant")
            fft_full = np.fft.rfft(chunk, n=FREQ_BINS * 2)
            mag = np.abs(fft_full[:FREQ_BINS])
            self.spec_matrix[:, i] = np.log1p(mag)

        maxv = self.spec_matrix.max()
        if maxv > 0:
            self.spec_matrix /= maxv

        # Create spectrogram surface
        arr8 = (self.spec_matrix * 255).astype(np.uint8)
        arr8 = np.flipud(arr8)
        arr_rgb = np.repeat(arr8[:, :, None], 3, axis=2)
        arr_rgb = np.transpose(arr_rgb, (1, 0, 2))
        self.spec_surf0 = pygame.surfarray.make_surface(arr_rgb)

    def save_project(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", title="Save project", filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        data = {
            "video_path": self.VIDEO_PATH,
            "audio_path": self.AUDIO_PATH,
            "curve": self.frame_curve.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_project(self):
        path = filedialog.askopenfilename(
            title="Load project", filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.VIDEO_PATH = data["video_path"]
        self.AUDIO_PATH = data["audio_path"]
        loaded = np.array(data.get("curve", [])[:CURVE_RES], dtype=np.float32)
        self.frame_curve[:] = np.pad(
            loaded,
            (0, max(0, CURVE_RES - len(loaded))),
            constant_values=0.5,
        )
        self.load_media()
        self.frame_cache.clear()
        self.undo_stack.clear()

    def update_layout(self):
        self.window_width, self.window_height = self.screen.get_size()
        self.window_width = max(self.window_width, WINDOW_MIN_WIDTH)
        self.window_height = max(self.window_height, WINDOW_MIN_HEIGHT)

    def audio_thread(self):
        try:
            channels = 1 if self.audio_data.ndim == 1 else self.audio_data.shape[1]

            def callback(outdata, frames, time_info, status):
                if not self.playing:
                    outdata[:] = np.zeros((frames, channels))
                    return
                start = int(self.playback_time * self.audio_sr)
                end = start + frames
                chunk = self.audio_data[start:end]
                if len(chunk) < frames:
                    pad_shape = ((0, frames - len(chunk)), (0, 0)) if chunk.ndim > 1 else ((0, frames - len(chunk)),)
                    chunk = np.pad(chunk, pad_shape, mode="constant")
                if channels == 1 and chunk.ndim == 1:
                    outdata[:] = chunk[:, None]
                else:
                    outdata[:] = chunk
                self.playback_time += frames / self.audio_sr
                if self.playback_time >= self.audio_duration:
                    self.playing = False

            stream = sd.OutputStream(
                callback=callback, samplerate=self.audio_sr, channels=channels
            )
            with stream:
                while self.running:
                    time.sleep(0.01)
        except Exception as e:
            print("Audio thread failed:", e)

    def get_frame_at_time(self, t):
        length = len(self.frame_curve)
        idx = int((t / self.audio_duration) * (length - 1))
        idx = max(0, min(length - 1, idx))
        return int(self.frame_curve[idx] * self.total_frames)

    def get_video_frame(self, frame_idx):
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        h_target = self.divider_pos
        h, w, _ = frame.shape
        scale = min(self.window_width / w, h_target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        x_offset = (self.window_width - new_w) // 2
        y_offset = (h_target - new_h) // 2
        surface = pygame.Surface((self.window_width, h_target))
        surface.fill((0, 0, 0))
        frame = cv2.cvtColor(np.fliplr(frame), cv2.COLOR_BGR2RGB)
        frame_surf = pygame.surfarray.make_surface(np.rot90(frame))
        surface.blit(frame_surf, (x_offset, y_offset))
        if len(self.frame_cache) >= CACHE_SIZE:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[frame_idx] = surface
        return surface
    

    def draw_curve_bar(self, surf):
        height = self.window_height - self.divider_pos
        bar_top = self.divider_pos
        pygame.draw.rect(surf, (30, 30, 30), (0, bar_top, self.window_width, height))
        length = CURVE_RES

        # Compute visible index range
        world_min = self.offset_x
        world_max = (self.window_width / self.zoom_x) + self.offset_x
        i_min = int(np.floor((world_min / self.window_width) * (length - 1)))
        i_max = int(np.ceil((world_max / self.window_width) * (length - 1)))
        i_min = max(0, min(length - 1, i_min))
        i_max = max(0, min(length - 1, i_max))

        if self.spec_mode and self.spec_surf0 is not None:
            if i_max >= i_min:
                slice_width = i_max - i_min + 1
                sub = self.spec_surf0.subsurface((i_min, 0, slice_width, FREQ_BINS))
                scaled = pygame.transform.scale(sub, (self.window_width, height))
                surf.blit(scaled, (0, bar_top))
        else:
            midpoint = bar_top + (height // 2)
            for i in range(i_min, i_max + 1):
                i_norm = i / (length - 1)
                world_x = i_norm * self.window_width
                sx = (world_x - self.offset_x) * self.zoom_x

                amp = self.wave_env[i] if self.wave_env is not None else 0.0
                world_y = amp * (height / 2)
                sy1 = midpoint - (world_y * self.zoom_y) + self.offset_y
                sy2 = midpoint + (world_y * self.zoom_y) + self.offset_y

                if np.isfinite(sy1) and np.isfinite(sy2):
                    pygame.draw.line(
                        surf,
                        (100, 100, 255),
                        (int(sx), int(sy1)),
                        (int(sx), int(sy2)),
                    )

        # Draw red frame_curve overlay
        for i in range(i_min, i_max):
            i_norm = i / (length - 1)
            i1_norm = (i + 1) / (length - 1)
            world_x1 = i_norm * self.window_width
            world_x2 = i1_norm * self.window_width
            sx1 = (world_x1 - self.offset_x) * self.zoom_x
            sx2 = (world_x2 - self.offset_x) * self.zoom_x

            world_y1 = (1.0 - self.frame_curve[i]) * height
            world_y2 = (1.0 - self.frame_curve[i + 1]) * height
            center_world = height / 2
            dy1 = (world_y1 - center_world) * self.zoom_y
            dy2 = (world_y2 - center_world) * self.zoom_y
            sy1 = bar_top + center_world + dy1 + self.offset_y
            sy2 = bar_top + center_world + dy2 + self.offset_y

            if not (
                np.isfinite(sx1) and np.isfinite(sy1) and
                np.isfinite(sx2) and np.isfinite(sy2)
            ):
                continue

            if (0 <= sx1 < self.window_width or 0 <= sx2 < self.window_width):
                pygame.draw.line(
                    surf,
                    (255, 0, 0),
                    (int(sx1), int(sy1)),
                    (int(sx2), int(sy2)),
                    2,
                )

        # Draw green playback cursor
        world_xc = (self.playback_time / self.audio_duration) * self.window_width
        cx = (world_xc - self.offset_x) * self.zoom_x
        if np.isfinite(cx) and 0 <= cx < self.window_width:
            pygame.draw.line(
                surf,
                (0, 255, 0),
                (int(cx), bar_top),
                (int(cx), bar_top + height),
                2,
            )

        height = self.window_height - self.divider_pos
        bar_top = self.divider_pos
        pygame.draw.rect(surf, (30, 30, 30), (0, bar_top, self.window_width, height))
        length = CURVE_RES

        # Compute visible index range
        world_min = self.offset_x
        world_max = (self.window_width / self.zoom_x) + self.offset_x
        i_min = int(np.floor((world_min / self.window_width) * (length - 1)))
        i_max = int(np.ceil((world_max / self.window_width) * (length - 1)))
        i_min = max(0, min(length - 1, i_min))
        i_max = max(0, min(length - 1, i_max))

        if self.spec_mode and self.spec_surf0 is not None:
            if i_max >= i_min:
                slice_width = i_max - i_min + 1
                sub = self.spec_surf0.subsurface((i_min, 0, slice_width, FREQ_BINS))
                scaled = pygame.transform.scale(sub, (self.window_width, height))
                surf.blit(scaled, (0, bar_top))
        else:
            midpoint = bar_top + (height // 2)
            for i in range(i_min, i_max + 1):
                i_norm = i / (length - 1)
                world_x = i_norm * self.window_width
                sx = (world_x - self.offset_x) * self.zoom_x

                amp = self.wave_env[i] if self.wave_env is not None else 0.0
                # world_y is half-height of waveform at this index
                world_y = amp * (height / 2)

                # Apply zoom first, then pan in screen coords
                scaled_y = world_y * self.zoom_y
                sy1 = midpoint - scaled_y + self.offset_y
                sy2 = midpoint + scaled_y + self.offset_y

                if np.isfinite(sy1) and np.isfinite(sy2):
                    pygame.draw.line(
                        surf,
                        (100, 100, 255),
                        (int(sx), int(sy1)),
                        (int(sx), int(sy2)),
                    )

        # Draw red frame_curve overlay over same visible range
        for i in range(i_min, i_max):
            i_norm = i / (length - 1)
            i1_norm = (i + 1) / (length - 1)
            world_x1 = i_norm * self.window_width
            world_x2 = i1_norm * self.window_width
            sx1 = (world_x1 - self.offset_x) * self.zoom_x
            sx2 = (world_x2 - self.offset_x) * self.zoom_x

            # world_curve_y measured from top of bar
            world_y1 = (1.0 - self.frame_curve[i]) * height
            world_y2 = (1.0 - self.frame_curve[i + 1]) * height

            # center-based zoom
            center_world = height / 2
            dy1 = (world_y1 - center_world) * self.zoom_y
            dy2 = (world_y2 - center_world) * self.zoom_y
            sy1 = bar_top + center_world + dy1 + self.offset_y
            sy2 = bar_top + center_world + dy2 + self.offset_y

            if not (
                np.isfinite(sx1)
                and np.isfinite(sy1)
                and np.isfinite(sx2)
                and np.isfinite(sy2)
            ):
                continue
            if (0 <= sx1 < self.window_width or 0 <= sx2 < self.window_width):
                pygame.draw.line(
                    surf,
                    (255, 0, 0),
                    (int(sx1), int(sy1)),
                    (int(sx2), int(sy2)),
                    2,
                )

        # Draw green playback cursor
        world_xc = (self.playback_time / self.audio_duration) * self.window_width
        cx = (world_xc - self.offset_x) * self.zoom_x
        if np.isfinite(cx) and 0 <= cx < self.window_width:
            pygame.draw.line(
                surf,
                (0, 255, 0),
                (int(cx), bar_top),
                (int(cx), bar_top + height),
                2,
            )

    def set_curve_at_mouse(self, x, y):
        if y < self.divider_pos:
            return

        height = self.window_height - self.divider_pos
        bar_top = self.divider_pos
        center = height / 2

        # compute world_x as before...
        world_x = (x / self.zoom_x) + self.offset_x

        # fix vertical: invert center‐based zoom
        world_y = ((y - bar_top - center - self.offset_y) / self.zoom_y) + center

        norm_x = world_x / self.window_width
        idx = int(norm_x * (CURVE_RES - 1))
        idx = max(0, min(CURVE_RES - 1, idx))

        val = 1.0 - (world_y / height)
        val = np.clip(val, 0.0, 1.0)

        if self.last_mouse_idx is None:
            self.undo_stack.append(self.frame_curve.copy())
            if len(self.undo_stack) > 100:
                self.undo_stack.pop(0)

        if self.last_mouse_idx is not None and self.last_mouse_idx != idx:
            prev_i = self.last_mouse_idx
            prev_val = self.frame_curve[prev_i]
            cur_i = idx
            start_i = min(prev_i, cur_i)
            end_i = max(prev_i, cur_i)
            for i in range(start_i, end_i + 1):
                t = (i - start_i) / (end_i - start_i) if end_i != start_i else 0.0
                v = (1 - t) * prev_val + t * val if prev_i <= cur_i else (1 - t) * val + t * prev_val
                self.frame_curve[i] = v
        else:
            self.frame_curve[idx] = val

        self.last_mouse_idx = idx



    def export_video(self, fps=60):
        from tkinter import Toplevel, ttk

        # Prompt for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            title="Export Full Video"
        )
        if not save_path:
            return

        # Compute number of output frames
        num_out = int(round(self.audio_duration * fps))
        if num_out <= 0:
            return

        # Create a simple Tkinter progress window
        progress_win = Toplevel(self.root)
        progress_win.title("Exporting Video...")
        progress_win.geometry("450x120")
        progress_win.resizable(False, False)

        ttk.Label(progress_win, text="Rendering frames:").pack(pady=(10, 0))
        pb = ttk.Progressbar(
            progress_win,
            orient="horizontal",
            length=400,
            mode="determinate",
            maximum=num_out
        )
        pb.pack(pady=(5, 5))
        status_label = ttk.Label(progress_win, text=f"Frame 0 / {num_out}")
        status_label.pack()

        # Temporary raw video (no audio)
        temp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_vid.close()
        temp_path = temp_vid.name

        # Open fresh VideoCapture and get dimensions
        cap_export = cv2.VideoCapture(self.VIDEO_PATH)
        total_w = int(cap_export.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_h = int(cap_export.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_export.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare VideoWriter for raw frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (total_w, total_h))

    
        for frame_i in range(num_out):
            t = frame_i / fps
            idx_curve = int((t / self.audio_duration) * (CURVE_RES - 1))
            idx_curve = max(0, min(CURVE_RES - 1, idx_curve))
            src_frame_i = int(self.frame_curve[idx_curve] * (total_frames - 1))
            src_frame_i = max(0, min(total_frames - 1, src_frame_i))

            cap_export.set(cv2.CAP_PROP_POS_FRAMES, src_frame_i)
            ret, frame = cap_export.read()
            if ret:
                writer.write(frame)
            # if ret is False, skip but continue instead of breaking
            else:
                # still update progress so UI doesn't freeze
                pass

            # Update progress bar and status label
            pb['value'] = frame_i + 1
            status_label.config(text=f"Frame {frame_i + 1} / {num_out}")
            progress_win.update()

        writer.release()
        cap_export.release()

        status_label.config(text="Merging audio…")
        progress_win.update()

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-i", self.AUDIO_PATH,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                save_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.unlink(temp_path)
            status_label.config(text="Export complete!")
        except Exception as e:
            print("ffmpeg failed:", e)
            try: os.unlink(temp_path)
            except: pass
            status_label.config(text="Export failed!")

        progress_win.update()
        progress_win.after(1500, progress_win.destroy)


    def run(self):
        self.screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        self.update_layout()
        threading.Thread(target=self.audio_thread, daemon=True).start()
        clock = pygame.time.Clock()

        while self.running:
            self.update_layout()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.VIDEORESIZE:
                    self.frame_cache.clear()
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
                        self.ctrl_down = True
                        self.line_pts.clear()
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_r:
                        self.record_mode = not self.record_mode
                        self.idx_prev = None
                        self.dur_prev = None
                    elif event.key == pygame.K_s and not (mods & pygame.KMOD_CTRL):
                        self.spec_mode = not self.spec_mode
                    elif event.key == pygame.K_z and (mods & pygame.KMOD_CTRL):
                        if self.undo_stack:
                            self.frame_curve[:] = self.undo_stack.pop()
                    elif event.key == pygame.K_SPACE:
                        if self.playback_time >= self.audio_duration:
                            self.playback_time = 0.0
                        self.playing = not self.playing
                    elif event.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
                        self.save_project()
                    elif event.key == pygame.K_l and (mods & pygame.KMOD_CTRL):
                        self.load_project()
                    elif event.key == pygame.K_BACKSPACE:
                        # Reset zoom & pan
                        self.zoom_x = 1.0
                        self.zoom_y = 1.0
                        self.offset_x = 0.0
                        self.offset_y = 0.0
                    elif event.key == pygame.K_x and (mods & pygame.KMOD_CTRL):
                        self.export_video()        # ← call export here

                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
                        self.ctrl_down = False
                        self.line_pts.clear()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mods = pygame.key.get_mods()
                    if event.button == 1 and self.ctrl_down:
                        # Ctrl+left: accumulate polyline points
                        self.line_pts.append(event.pos)
                        if len(self.line_pts) >= 2:
                            x1, y1 = self.line_pts[-2]
                            x2, y2 = self.line_pts[-1]
                            height = self.window_height - self.divider_pos
                            self.undo_stack.append(self.frame_curve.copy())
                            if len(self.undo_stack) > 100:
                                self.undo_stack.pop(0)
                            for i in range(CURVE_RES):
                                world_x = (i / (CURVE_RES - 1)) * self.window_width
                                sx = (world_x - self.offset_x) * self.zoom_x
                                if (x1 <= sx <= x2) or (x2 <= sx <= x1):
                                    t = (sx - x1) / (x2 - x1) if x2 != x1 else 0
                                    interp_y = (1 - t) * y1 + t * y2
                                    bar_top = self.divider_pos
                                    height  = self.window_height - bar_top
                                    center  = height / 2
                                    world_y = ((interp_y - bar_top - center - self.offset_y) / self.zoom_y) + center
                                    v = 1.0 - (world_y / height)
                                    self.frame_curve[i] = np.clip(v, 0.0, 1.0)

                    elif event.button == 1:
                        if abs(event.pos[1] - self.divider_pos) < 5:
                            self.divider_dragging = True
                        else:
                            self.last_mouse_idx = None
                            self.set_curve_at_mouse(*event.pos)

                    elif event.button == 3:
                        if mods & pygame.KMOD_SHIFT:
                            self.zooming = True
                            self.zoom_origin = event.pos
                        else:
                            world_x = (event.pos[0] / self.zoom_x) + self.offset_x
                            norm = world_x / self.window_width
                            self.playback_time = norm * self.audio_duration

                    elif event.button == 2 or (event.button == 1 and (mods & pygame.KMOD_SHIFT)):
                        self.panning = True
                        self.pan_origin = event.pos

                    elif event.button in (4, 5):
                        # Mouse wheel zoom centered on mouse
                        mx, my = event.pos
                        old_zoom_x = self.zoom_x
                        old_zoom_y = self.zoom_y
                        scale = ZOOM_STEP if event.button == 4 else (1 / ZOOM_STEP)

                        world_mouse_x = (mx / old_zoom_x) + self.offset_x
                        world_mouse_y = ((my - self.divider_pos) / old_zoom_y) + self.offset_y

                        self.zoom_x *= scale
                        self.zoom_y *= scale

                        self.offset_x = world_mouse_x - (mx / self.zoom_x)
                        self.offset_y = world_mouse_y - ((my - self.divider_pos) / self.zoom_y)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.divider_dragging = False
                        self.last_mouse_idx = None
                    if event.button in (2, 3):
                        self.panning = False
                        self.zooming = False

                elif event.type == pygame.MOUSEMOTION:
                    mods = pygame.key.get_mods()
                    if self.panning:
                        dx, dy = event.rel
                        self.offset_x -= dx / self.zoom_x
                        self.offset_y += dy
                    elif self.zooming:
                        dx, dy = event.rel
                        mx, my = event.pos
                        rel_x = (mx / self.zoom_x) + self.offset_x
                        rel_y = ((my - self.divider_pos) / self.zoom_y) + self.offset_y
                        self.zoom_x *= (1 + dx * 0.01)
                        self.zoom_y *= (1 - dy * 0.01)
                        self.offset_x = rel_x - (mx / self.zoom_x)
                        self.offset_y = rel_y - ((my - self.divider_pos) / self.zoom_y)
                    elif mods & pygame.KMOD_SHIFT:
                        x, _ = event.pos
                        raw_norm = ((x / self.zoom_x + self.offset_x) / self.window_width)
                        idx = int(raw_norm * (CURVE_RES - 1))
                        idx = max(0, min(CURVE_RES - 1, idx))
                        if self.hover_index is None or idx != self.hover_index:
                            self.hover_index = idx
                            self.hovering = True

            if self.divider_dragging:
                y = pygame.mouse.get_pos()[1]
                new_div = max(100, min(y, self.window_height - 100))
                if new_div != self.divider_pos:
                    self.divider_pos = new_div
                    self.frame_cache.clear()
                self.update_layout()

            if pygame.mouse.get_pressed()[0] and not self.divider_dragging and not self.panning and not (
                pygame.key.get_mods() & pygame.KMOD_CTRL
            ):
                self.set_curve_at_mouse(*pygame.mouse.get_pos())

            # Record mode interpolation
            if self.record_mode:
                length = CURVE_RES
                world_x = (self.playback_time / self.audio_duration) * self.window_width
                idx = int((world_x / self.window_width) * (length - 1))
                idx = max(0, min(length - 1, idx))
                mx, my = pygame.mouse.get_pos()
                if my >= self.divider_pos:
                    bar_h = self.window_height - self.divider_pos
                    bar_top = self.divider_pos
                    height  = self.window_height - bar_top
                    center  = height / 2
                    world_y = ((my - bar_top - center - self.offset_y) / self.zoom_y) + center
                    val = 1.0 - (world_y / bar_h)
                    val = max(0.0, min(1.0, val))
                    if self.idx_prev is None:
                        self.undo_stack.append(self.frame_curve.copy())
                        if len(self.undo_stack) > 100:
                            self.undo_stack.pop(0)
                        self.frame_curve[idx] = val
                        self.idx_prev = idx
                        self.dur_prev = self.playback_time
                    else:
                        prev_i = self.idx_prev
                        prev_val = self.frame_curve[prev_i]
                        cur_i = idx
                        start_i = min(prev_i, cur_i)
                        end_i = max(prev_i, cur_i)
                        if end_i != start_i:
                            for i in range(start_i, end_i + 1):
                                t = (i - start_i) / (end_i - start_i)
                                if prev_i <= cur_i:
                                    v = (1 - t) * prev_val + t * val
                                else:
                                    v = (1 - t) * val + t * prev_val
                                self.frame_curve[i] = v
                        else:
                            self.frame_curve[start_i] = val
                        self.idx_prev = cur_i
                        self.dur_prev = self.playback_time

            # Determine which video frame to display
            if self.hovering:
                curve_val = self.frame_curve[self.hover_index]
                frame_idx = int(curve_val * self.total_frames)
                display_frame = self.get_video_frame(frame_idx)
            else:
                display_frame = self.get_video_frame(self.get_frame_at_time(self.playback_time))

            if display_frame is not None:
                self.screen.blit(display_frame, (0, 0))

            if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                self.hovering = False
                self.hover_index = None

            self.draw_curve_bar(self.screen)
            pygame.display.flip()

            idle = (
                not self.playing
                and not pygame.mouse.get_pressed()[0]
                and not pygame.event.peek()
            )
            if idle:
                pygame.time.wait(10)
            else:
                clock.tick(30)

        self.cap.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    VideoScrubber().run()
