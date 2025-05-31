import os
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Label
from PIL import Image, ImageTk

class TimelineEditor:
    def __init__(self, root):
        self.framerate = 30
        self.current_position = 0.0
        self.playing = False
        self.start_time = None  # for playback reference
        self.zoom_default = True  # if True, auto-fit zoom
        self.timeline_zoom = 1.0  # pixels per second
        
        self.root = root
        self.root.title("Timeline Editor")

        self.image_dir = "input"
        self.list_file = os.path.join(self.image_dir, "list.txt")
        self.total_duration = 5.0
        self.entries = []  # list of (name, delay)
        self.normalized = []  # list of {'name':..., 'time':...}
        self.image_cache = {}
        self.selected_indices = set()  # set of indices into self.normalized
        self.hover_index = None
        self.drag_origin = None  # logical x for dragging
        self.selection_box = None  # (x0_screen,y0,x1_screen,y1_screen)
        self.drag_mode = None  # 'move', 'scale_left', 'scale_right', 'select_box'
        self.clicked_on_point = False

        self.setup_ui()
        self.load_list()

    def setup_ui(self):
        fr = tk.Frame(self.root)
        fr.pack(fill=tk.X)
        tk.Label(fr, text="Framerate:").pack(side=tk.LEFT)
        self.fps_entry = tk.Entry(fr, width=5)
        self.fps_entry.insert(0, str(self.framerate))
        self.fps_entry.pack(side=tk.LEFT)
        tk.Button(fr, text="Set FPS", command=self.set_fps).pack(side=tk.LEFT)

        top = tk.Frame(self.root)
        top.pack(fill=tk.X)

        tk.Button(top, text="Open Dir", command=self.pick_directory).pack(side=tk.LEFT)
        tk.Button(top, text="Open list.txt", command=self.pick_list).pack(side=tk.LEFT)
        self.play_button = tk.Button(top, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        tk.Button(top, text="Save", command=self.save_list).pack(side=tk.LEFT)
        tk.Button(top, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT)

        tk.Label(top, text="Duration (s):").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(top, width=5)
        self.duration_entry.insert(0, str(self.total_duration))
        self.duration_entry.pack(side=tk.LEFT)
        tk.Button(top, text="Set", command=self.set_duration).pack(side=tk.LEFT)

        # Zoom controls
        tk.Button(top, text="Zoom+", command=self.zoom_in).pack(side=tk.LEFT)
        tk.Button(top, text="Zoom-", command=self.zoom_out).pack(side=tk.LEFT)
        tk.Button(top, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT)

        # Scrollable timeline area
        timeline_frame = tk.Frame(self.root)
        timeline_frame.pack(fill=tk.X, padx=10, pady=10)
        self.timeline = Canvas(timeline_frame, bg="gray", height=60)
        self.timeline.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.h_scroll = tk.Scrollbar(timeline_frame, orient=tk.HORIZONTAL, command=self.timeline.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.timeline.config(xscrollcommand=self.h_scroll.set)

        self.timeline.bind("<Button-1>", self.on_timeline_click)
        self.timeline.bind("<B1-Motion>", self.on_timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self.on_timeline_release)
        self.timeline.bind("<Motion>", self.on_timeline_hover)
        self.timeline.bind("<Leave>", self.on_timeline_leave)
        self.timeline.bind("<Configure>", lambda e: self.on_timeline_configure())

        self.image_canvas = Canvas(self.root, bg="black")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_canvas.bind("<Configure>", self.on_image_canvas_resize)

        # Status bar
        self.status = Label(self.root, text="Ready", anchor=tk.W)
        self.status.pack(fill=tk.X)

    def pick_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.image_dir = path
            self.list_file = os.path.join(self.image_dir, "list.txt")
            self.load_list()

    def pick_list(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            self.list_file = path
            self.load_list()

    def load_list(self):
        if not os.path.exists(self.list_file):
            messagebox.showerror("Error", f"list.txt not found in {self.image_dir}")
            return

        self.entries.clear()
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    name, delay = parts
                    self.entries.append((name, float(delay)))

        self.build_normalized()
        self.reset_zoom()
        self.draw_timeline()

    def build_normalized(self):
        total_units = sum(delay for _, delay in self.entries)
        self.normalized = []
        current_time = 0.0
        for name, delay in self.entries:
            self.normalized.append({'name': name, 'time': current_time})
            current_time += (delay / total_units) * self.total_duration
        self.selected_indices.clear()

    def set_duration(self):
        try:
            self.total_duration = float(self.duration_entry.get())
        except ValueError:
            self.total_duration = 5.0
            self.duration_entry.delete(0, tk.END)
            self.duration_entry.insert(0, "5.0")
        self.build_normalized()
        self.zoom_default = True
        self.reset_zoom()
        self.draw_timeline()

    def on_timeline_configure(self):
        # Adjust scrollregion and redraw when resized
        self.reset_zoom() if self.zoom_default else None
        self.draw_timeline()

    def draw_timeline(self):
        self.timeline.delete("all")
        # Total width in px based on zoom
        total_width = int(self.total_duration * self.timeline_zoom)
        h = self.timeline.winfo_height()
        # set scrollregion
        self.timeline.config(scrollregion=(0, 0, total_width, h))

        # draw playhead at logical x
        x_pos = self.current_position * self.timeline_zoom
        self.timeline.create_line(x_pos, 0, x_pos, h, fill="red", width=2)

        center_y = h // 2
        for idx, item in enumerate(self.normalized):
            x = item['time'] * self.timeline_zoom
            color = "green" if idx in self.selected_indices else "blue"
            self.timeline.create_oval(x - 5, center_y - 5, x + 5, center_y + 5,
                                     fill=color, tags=(f"point{idx}",))
        if len(self.selected_indices) > 1:
            times = [self.normalized[i]['time'] for i in self.selected_indices]
            x0 = min(times) * self.timeline_zoom
            x1 = max(times) * self.timeline_zoom
            self.timeline.create_rectangle(x0, center_y - 10, x1, center_y + 10,
                                           outline="white", dash=(2,2), tags=("group_box",))
            self.timeline.create_rectangle(x0 - 5, center_y - 15, x0 + 5, center_y + 15,
                                           fill="white", tags=("scale_left",))
            self.timeline.create_rectangle(x1 - 5, center_y - 15, x1 + 5, center_y + 15,
                                           fill="white", tags=("scale_right",))
        if self.selection_box:
            x0, y0, x1, y1 = self.selection_box
            self.timeline.create_rectangle(x0, y0, x1, y1, outline="yellow", dash=(2,2))

    def on_timeline_click(self, event):
        w = self.timeline.winfo_width()
        h = self.timeline.winfo_height()
        center_y = h // 2
        # logical x from screen x
        lx = self.timeline.canvasx(event.x)
        # check scale handles
        if len(self.selected_indices) > 1:
            times = [self.normalized[i]['time'] for i in self.selected_indices]
            x0 = min(times) * self.timeline_zoom
            x1 = max(times) * self.timeline_zoom
            if abs(lx - x0) <= 5 and center_y - 15 <= event.y <= center_y + 15:
                self.drag_mode = 'scale_left'
                self.clicked_on_point = True
                return
            if abs(lx - x1) <= 5 and center_y - 15 <= event.y <= center_y + 15:
                self.drag_mode = 'scale_right'
                self.clicked_on_point = True
                return
        self.drag_origin = lx
        self.clicked_on_point = False
        idx = None
        if abs(event.y - center_y) <= 6:
            idx = self.get_item_at_x(lx)
        if idx is not None:
            self.clicked_on_point = True
            if event.state & 0x0004:  # Ctrl pressed
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                else:
                    self.selected_indices.add(idx)
            else:
                if idx not in self.selected_indices:
                    self.selected_indices = {idx}
            self.show_image(idx)
            self.drag_mode = 'move'
        else:
            self.selected_indices.clear()
            self.drag_mode = 'select_box'
            self.selection_box = (event.x, event.y, event.x, event.y)
        if not self.clicked_on_point:
            self.current_position = lx / self.timeline_zoom if self.timeline_zoom > 0 else 0
            if self.playing:
                import time
                self.start_time = time.time() - self.current_position
            self.draw_timeline()

    def on_timeline_drag(self, event):
        if self.drag_mode == 'select_box':
            x0, y0 = self.drag_origin, event.y  # drag_origin here was lx, but we need y0 separately
            # Actually, keep selection_box in screen coords
            sx0, sy0, sx1, sy1 = self.selection_box
            self.selection_box = (sx0, sy0, event.x, event.y)
            self.draw_timeline()
            return
        if self.drag_mode == 'move' and self.selected_indices:
            lx = self.timeline.canvasx(event.x)
            dx_time = (lx - self.drag_origin) / self.timeline_zoom if self.timeline_zoom > 0 else 0
            for i in list(self.selected_indices):
                new_time = self.normalized[i]['time'] + dx_time
                self.normalized[i]['time'] = max(0.0, min(new_time, self.total_duration))
            self.drag_origin = lx
            self.draw_timeline()
            return
        if self.drag_mode in ('scale_left', 'scale_right'):
            lx = self.timeline.canvasx(event.x)
            times = [self.normalized[i]['time'] for i in self.selected_indices]
            min_t = min(times)
            max_t = max(times)
            span = max_t - min_t if max_t > min_t else 1.0
            new_time = lx / self.timeline_zoom if self.timeline_zoom > 0 else 0
            if self.drag_mode == 'scale_left':
                right_ref = max_t
                new_span = right_ref - new_time
                if new_span <= 0:
                    return
                scale = new_span / span
                for i in self.selected_indices:
                    t = self.normalized[i]['time']
                    self.normalized[i]['time'] = right_ref - (right_ref - t) * scale
            else:
                left_ref = min_t
                new_span = new_time - left_ref
                if new_span <= 0:
                    return
                scale = new_span / span
                for i in self.selected_indices:
                    t = self.normalized[i]['time']
                    self.normalized[i]['time'] = left_ref + (t - left_ref) * scale
            self.selection_box = None
            self.draw_timeline()
            return

    def on_timeline_release(self, event):
        if self.drag_mode == 'select_box':
            sx0, sy0, sx1, sy1 = self.selection_box
            x0 = min(self.timeline.canvasx(sx0), self.timeline.canvasx(sx1))
            x1 = max(self.timeline.canvasx(sx0), self.timeline.canvasx(sx1))
            selected = set()
            for idx, item in enumerate(self.normalized):
                x_item = item['time'] * self.timeline_zoom
                if x0 <= x_item <= x1:
                    selected.add(idx)
            self.selected_indices = selected
            self.selection_box = None
            self.draw_timeline()
        self.drag_mode = None

    def on_timeline_hover(self, event):
        w = self.timeline.winfo_width()
        lx = self.timeline.canvasx(event.x)
        mouse_time = lx / self.timeline_zoom if self.timeline_zoom > 0 else 0
        if self.normalized:
            closest_idx = min(range(len(self.normalized)), key=lambda i: abs(self.normalized[i]['time'] - mouse_time))
            if closest_idx != self.hover_index:
                self.hover_index = closest_idx
                self.show_image(closest_idx)
                t = self.normalized[closest_idx]['time']
                name = self.normalized[closest_idx]['name']
                self.status.config(text=f"{name}  |  Time: {t:.2f}s")

    def on_timeline_leave(self, event):
        if self.normalized:
            mouse_time = self.current_position
            closest_idx = min(range(len(self.normalized)), key=lambda i: abs(self.normalized[i]['time'] - mouse_time))
            self.show_image(closest_idx)
            t = self.normalized[closest_idx]['time']
            name = self.normalized[closest_idx]['name']
            self.status.config(text=f"{name}  |  Time: {t:.2f}s")
        else:
            self.status.config(text="")

    def on_image_canvas_resize(self, event):
        if self.selected_indices:
            idx = next(iter(self.selected_indices))
            self.show_image(idx)

    def get_item_at_x(self, lx):
        for idx, item in enumerate(self.normalized):
            x_item = item['time'] * self.timeline_zoom
            if abs(x_item - lx) < 6:
                return idx
        return None

    def show_image(self, idx):
        name = self.normalized[idx]['name']
        path = os.path.join(self.image_dir, name)
        if path not in self.image_cache:
            img = Image.open(path).convert("RGB")
            self.image_cache[path] = img

        img = self.image_cache[path].copy()
        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        img.thumbnail((cw * 1000, ch), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(img)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

    def set_fps(self):
        try:
            self.framerate = int(self.fps_entry.get())
        except ValueError:
            self.framerate = 30
            self.fps_entry.delete(0, tk.END)
            self.fps_entry.insert(0, "30")

    def zoom_in(self):
        w = self.timeline.winfo_width()
        if not self.zoom_default:
            self.timeline_zoom *= 1.25
        else:
            # initial zoom: keep existing and mark default off
            self.zoom_default = False
            self.timeline_zoom *= 1.25
        self.draw_timeline()

    def zoom_out(self):
        if not self.zoom_default:
            self.timeline_zoom /= 1.25
            self.draw_timeline()

    def reset_zoom(self):
        w = self.timeline.winfo_width()
        if self.total_duration > 0:
            self.timeline_zoom = w / self.total_duration
        else:
            self.timeline_zoom = 1
        self.zoom_default = True
        self.timeline.config(scrollregion=(0,0, self.total_duration*self.timeline_zoom, self.timeline.winfo_height()))

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")
        else:
            self.playing = True
            self.play_button.config(text="Stop")
            import time
            self.start_time = time.time() - self.current_position
            self.simulate_play()

    def simulate_play(self):
        import time
        self.root.update()
        self.image_canvas.delete("all")
        while self.playing:
            now = time.time()
            self.current_position = now - self.start_time
            if self.current_position > self.total_duration:
                break
            self.draw_timeline()
            closest_idx = min(range(len(self.normalized)), key=lambda i: abs(self.normalized[i]['time'] - self.current_position))
            item = self.normalized[closest_idx]
            path = os.path.join(self.image_dir, item['name'])
            if path not in self.image_cache:
                img = Image.open(path).convert("RGB")
                self.image_cache[path] = img
            img = self.image_cache[path].copy()
            cw = self.image_canvas.winfo_width()
            ch = self.image_canvas.winfo_height()
            img.thumbnail((cw, ch), Image.Resampling.NEAREST)
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(cw // 2, ch // 2, image=self.tk_img)
            self.root.update()
            time.sleep(1 / self.framerate)

        self.playing = False
        self.play_button.config(text="Play")
        if self.current_position > self.total_duration:
            self.current_position = self.total_duration
        self.draw_timeline()

    def delete_selected(self):
        if not self.selected_indices:
            return
        self.normalized = [item for idx, item in enumerate(self.normalized) if idx not in self.selected_indices]
        self.selected_indices.clear()
        self.draw_timeline()

    def save_list(self):
        sorted_items = sorted(self.normalized, key=lambda x: x['time'])
        delays = []
        times = [item['time'] for item in sorted_items]
        for i in range(1, len(times)):
            delays.append(times[i] - times[i - 1])
        delays.append(1.0)

        default_name = os.path.splitext(os.path.basename(self.list_file))[0] + "_modified.txt"
        new_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name,
                                                filetypes=[("Text Files", "*.txt")])
        if not new_path:
            return

        with open(new_path, 'w') as f:
            for item, d in zip(sorted_items, delays):
                f.write(f"{item['name']} {d}\n")
        messagebox.showinfo("Saved", f"Saved to {new_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TimelineEditor(root)
    root.mainloop()

