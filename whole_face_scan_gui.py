import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import threading, queue
import os
import shutil
import csv
import sys
# Ensure WORKING code directory is in path
ROOT = Path(__file__).parent.resolve()
WORKING = ROOT / 'WORKING CODE'
if str(WORKING) not in sys.path:
    sys.path.insert(0, str(WORKING))

# Import external modules (assumed to be correct)
from arcface_pipeline import load_yunet_model, load_arcface_model
from face_alignment import align_and_crop
from arcface_embedder import generate_arcface_embedding, calculate_distance
# --- Setup Directory ---
ENROLL_DIR = Path('face_pipeline') / 'enrollments'
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UI THEME / PALETTES (FINALIZED DARK MODE)
# =============================================================================

PALETTE_LIGHT = {
    'bg': '#f5f7fb',
    'panel': '#ffffff',
    'muted': '#6b7280',
    'accent': '#2563eb',
    'accent_hover': '#1d4ed8',
    'accent_light': '#eff6ff',
    'accent_alt': '#10b981',
    'accent_alt_hover': '#059669',
    'danger': '#ef4444',
    'danger_hover': '#dc2626',
    'card_border': '#e6edf3',
    'dark': '#0f172a',
    'text': '#0f172a'
}

PALETTE_DARK = {
    # 030812 (Deep Black-Blue)
    'bg': '#030812', 
    # 020764 (Rich Dark Blue)
    'panel': '#020764',
    # 043780 (Mid-Tone Blue / Light Surface)
    'accent_light': '#043780',
    # 043780 (Card/Border)
    'card_border': '#043780',
    # 025EC4 (Bright Blue / Primary Action)
    'accent': '#025EC4',
    # 043780 (Accent Hover)
    'accent_hover': '#043780',
    # 0ECCED (Vibrant Teal / Secondary Action & Success)
    'accent_alt': '#0ECCED',
    # 0BB6D1 (Slightly darker teal for hover)
    'accent_alt_hover': '#0BB6D1',
    # Errors
    'danger': '#E74C3C',
    'danger_hover': '#c0392b',
    # General light text on dark
    'text': '#E0E0E0', 
    # Muted helper text
    'muted': '#888888',
    'dark': '#E0E0E0'
}

# Default palette starts as DARK for modern aesthetic
PALETTE = dict(PALETTE_DARK)

# Font presets
HEADER_FONT = ('Segoe UI', 14, 'bold')
LABEL_FONT = ('Segoe UI', 10)
SMALL_FONT = ('Segoe UI', 9)
BTN_FONT = ('Segoe UI', 10, 'bold')


class WholeFaceScanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Whole-Face Multi-View Enrollment')
        self.root.geometry('1100x700')
        self.theme = 'dark' # Start in dark mode
        self.palette = PALETTE
        
        # models
        self.yunet = None
        self.arcface = None

        # enrollment state
        self.user_id = tk.StringVar()
        self.paths = {'front': None, 'left': None, 'right': None}
        self.thumbs = {}

        # matching state
        self.test_path = None
        self.test_thumb = None

        # build UI
        self.create_ui()
        self.load_models()
        self.apply_theme() # Apply initial dark theme after widgets are created

    # ------------- Theming helpers -------------
    def apply_theme(self):
        P = self.palette
        try:
            self.root.configure(bg=P['bg'])
            
            # Topbar
            if hasattr(self, 'topbar'):
                self.topbar.configure(bg=P['panel'], highlightbackground=P['card_border'])
                self.title_lbl.configure(bg=P['panel'], fg=P['text'])
                self.theme_btn.configure(bg=P['accent_light'], fg=P['text'], activebackground=P['accent_light'])
            
            # Main frames
            if hasattr(self, 'main'): self.main.configure(bg=P['bg'])
            if hasattr(self, 'left_frame'): self.left_frame.configure(bg=P['panel'], highlightbackground=P['card_border'])
            if hasattr(self, 'right_frame'): self.right_frame.configure(bg=P['panel'], highlightbackground=P['card_border'])
            
            # Labels, Listbox, Menu, Entry styling
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Frame) or isinstance(widget, tk.Label):
                    self._recolor_recursive(widget)

            # Specific Listbox and Entry coloring (Tkinter styling is limited)
            self.results_list.configure(bg=P['card_border'], fg=P['text'], selectbackground=P['accent_hover'], selectforeground=P['text'])
            self.user_entry.configure(bg=P['accent_light'], fg=P['text'], insertbackground=P['text'], highlightbackground=P['card_border'])
            self.session_menu.configure(bg=P['accent_light'], fg=P['text'], highlightbackground=P['card_border'])
            self.threshold_slider.configure(bg=P['panel'], troughcolor=P['card_border'], fg=P['text'])
            self.test_canvas.configure(bg=P['card_border'])
            
            if hasattr(self, 'statusbar') and self.statusbar is not None:
                self.statusbar.configure(bg=P['bg'], fg=P['muted'])
                
        except Exception as e:
            print(f"Error applying theme: {e}")

    def _recolor_recursive(self, parent):
        """Recursively recolor widgets that accept bg/fg."""
        P = self.palette
        try:
            if parent['bg']: parent.configure(bg=P['panel'])
        except Exception: pass
        try:
            if parent['fg']: parent.configure(fg=P['text'])
        except Exception: pass
        
        for child in parent.winfo_children():
            # Handle specific element types
            if isinstance(child, (tk.Canvas, tk.Listbox)):
                 child.configure(bg=P['card_border'])
            if isinstance(child, tk.Label):
                if child['fg'] == '#6b7280': # Recolor muted text
                    child.configure(fg=P['muted'])
                else:
                    child.configure(fg=P['text'])
                child.configure(bg=P['panel'])
            elif isinstance(child, tk.Frame):
                child.configure(bg=P['panel'])
                self._recolor_recursive(child)
            elif isinstance(child, tk.Button):
                 # Buttons are handled by the make_button function with hover states
                 pass

    def toggle_theme(self):
        # Switch between light and dark palettes
        if self.theme == 'light':
            self.theme = 'dark'
            PALETTE.update(PALETTE_DARK)
        else:
            self.theme = 'light'
            PALETTE.update(PALETTE_LIGHT)
        self.palette = PALETTE
        self.apply_theme()

    def make_button(self, parent, text, command=None, kind='accent'):
        """Create a styled button with hover effects."""
        P = self.palette
        if kind == 'accent':
            bg = P['accent']; hover = P['accent_hover']; fg = 'white'
        elif kind == 'alt':
            bg = P['accent_alt']; hover = P['accent_alt_hover']; fg = 'white'
        elif kind == 'danger':
            bg = P['danger']; hover = P['danger_hover']; fg = 'white'
        else:
            bg = P['accent_light']; hover = P['accent_hover']; fg = P['text']
            
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, font=BTN_FONT, relief='flat', padx=10, pady=6, cursor='hand2')
        # hover effects
        def _enter(e):
            e.widget.configure(bg=hover)
        def _leave(e):
            e.widget.configure(bg=bg)
        btn.bind('<Enter>', _enter)
        btn.bind('<Leave>', _leave)
        return btn

    def status(self, txt: str):
        """Set the statusbar text if available, else print to console."""
        try:
            if hasattr(self, 'statusbar') and self.statusbar is not None:
                self.statusbar.configure(text=txt)
            else:
                print(f"STATUS: {txt}")
        except Exception:
            try:
                print(f"STATUS: {txt}")
            except Exception:
                pass

    def load_models(self):
        self.status('Loading models...')
        try:
            self.yunet = load_yunet_model()
            self.arcface = load_arcface_model()
            self.status('Models loaded')
        except Exception as e:
            messagebox.showerror('Model error', f'Failed to load models: {e}')
            self.status('Model load failed')

    def create_ui(self):
        # Top App Bar
        self.topbar = tk.Frame(self.root, bg=PALETTE['panel'], height=48, highlightthickness=1, highlightbackground=PALETTE['card_border'])
        self.topbar.pack(fill='x', side='top')
        self.title_lbl = tk.Label(self.topbar, text='😎 Whole-Face Enrollment & Matching', font=('Segoe UI', 13, 'bold'), bg=PALETTE['panel'], fg=PALETTE['text'])
        self.title_lbl.pack(side='left', padx=12, pady=8)
        self.theme_btn = self.make_button(self.topbar, text='Toggle Theme', command=self.toggle_theme, kind='light')
        self.theme_btn.pack(side='right', padx=12, pady=6)

        main = tk.Frame(self.root, bg=PALETTE['bg'])
        main.pack(fill='both', expand=True, padx=16, pady=16)
        self.main = main

        left = tk.Frame(main, width=520, bg=PALETTE['panel'], bd=0, highlightthickness=1, highlightbackground=PALETTE['card_border'])
        left.pack(side='left', fill='y', padx=(0,12), pady=0)
        right = tk.Frame(main, width=520, bg=PALETTE['panel'], bd=0, highlightthickness=1, highlightbackground=PALETTE['card_border'])
        right.pack(side='left', fill='both', expand=True, padx=(0,0), pady=0)
        self.left_frame = left
        self.right_frame = right

        # Enrollment box
        tk.Label(left, text='Enrollment', font=HEADER_FONT, bg=PALETTE['panel'], fg=PALETTE['text']).pack(anchor='w', padx=12, pady=(12,6))
        tk.Label(left, text='User ID:', font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(anchor='w', pady=(6,0), padx=12)
        ent = tk.Entry(left, textvariable=self.user_id, width=30, font=LABEL_FONT)
        ent.pack(anchor='w', padx=12, pady=(4,8))
        self.user_entry = ent

        imgs_frame = tk.Frame(left, bg=PALETTE['panel'])
        imgs_frame.pack(fill='x', pady=8, padx=8)

        # three columns for front/left/right
        for i, key in enumerate(['front', 'left', 'right']):
            col = tk.Frame(imgs_frame, bd=1, relief='flat', width=160, height=200, bg=PALETTE['panel'])
            col.grid(row=0, column=i, padx=6)
            tk.Label(col, text=key.capitalize(), font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(pady=(6,0))
            canvas = tk.Canvas(col, width=150, height=150, bg=PALETTE['bg'], highlightthickness=0)
            canvas.pack(padx=6, pady=6)
            btn = self.make_button(col, text='Load', command=lambda k=key: self.load_enroll_image(k), kind='light')
            btn.pack(pady=(0,8))
            # store canvas for thumbnail
            self.thumbs[key] = {'canvas': canvas, 'image': None, 'path': None}

        self.make_button(left, text='Enroll User', command=self.enroll_user, kind='accent').pack(pady=12, padx=12, fill='x')
        self.make_button(left, text='Batch Enroll (CSV)', command=self.batch_enroll_csv, kind='alt').pack(pady=6, padx=12, fill='x')
        self.make_button(left, text='Export Enrollments', command=self.export_enrollments, kind='alt').pack(pady=6, padx=12, fill='x')

        # Enrollment status
        self.enroll_status = tk.Label(left, text='Ready', anchor='w', bg=PALETTE['panel'], fg=PALETTE['text'])
        self.enroll_status.pack(fill='x', padx=12)

        # Session user selector (only one user is referenced per session)
        tk.Label(left, text='Session user:', font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(anchor='w', pady=(8,0), padx=12)
        self.session_var = tk.StringVar(value='')
        self.session_menu = tk.OptionMenu(left, self.session_var, '')
        self.session_menu.config(width=26, bg=PALETTE['accent_light'], fg=PALETTE['text'], activebackground=PALETTE['accent_hover'], activeforeground=PALETTE['text'])
        self.session_menu.pack(anchor='w', padx=12, pady=(4,6))
        sess_ctl = tk.Frame(left, bg=PALETTE['panel'])
        sess_ctl.pack(fill='x', padx=12)
        self.make_button(sess_ctl, text='Set Session', command=self.set_session_user, kind='alt').pack(side='left', padx=4, pady=6)
        self.make_button(sess_ctl, text='Refresh Users', command=self.refresh_enrolled_users, kind='light').pack(side='left', padx=4)

        # per-view display (front/left/right) moved to Enrollment panel (left)
        pv_left = tk.Frame(left, bg=PALETTE['panel'])
        pv_left.pack(fill='x', pady=(12,0), padx=12)
        self.view_canvases = {}
        for i, key in enumerate(['front','left','right']):
            f = tk.Frame(pv_left, bg=PALETTE['panel'])
            f.grid(row=0, column=i, padx=6)
            lbl = tk.Label(f, text=key.capitalize(), bg=PALETTE['panel'], fg=PALETTE['text'])
            lbl.pack()
            c = tk.Canvas(f, width=120, height=120, bg=PALETTE['card_border'], highlightthickness=0)
            c.pack()
            dlab = tk.Label(f, text='dist: -', bg=PALETTE['panel'], fg=PALETTE['muted'])
            dlab.pack()
            self.view_canvases[key] = {'canvas': c, 'label': dlab, 'image': None}

        # Matching panel
        tk.Label(right, text='Matching', font=HEADER_FONT, bg=PALETTE['panel'], fg=PALETTE['text']).pack(anchor='w', padx=12, pady=(12,6))
        match_frame = tk.Frame(right, bg=PALETTE['panel'])
        match_frame.pack(fill='x', pady=6, padx=12)
        self.make_button(match_frame, text='Load Test Image', command=self.load_test_image, kind='light').pack(side='left')
        self.make_button(match_frame, text='🔎 Match', command=self.match_test, kind='accent').pack(side='left', padx=8)
        self.make_button(match_frame, text='👥 Match All Faces', command=self.match_all_faces, kind='danger').pack(side='left', padx=8)
        # threshold control
        tk.Label(match_frame, text='Threshold:', font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(side='left', padx=(16,4))
        self.match_threshold = tk.DoubleVar(value=1.25)
        def _on_thresh(val):
            try:
                self.threshold_val_lbl.configure(text=f"{float(val):.2f}")
            except Exception:
                pass
        self.threshold_slider = tk.Scale(match_frame, from_=0.8, to=1.8, resolution=0.01, orient='horizontal', length=160, variable=self.match_threshold, command=_on_thresh, showvalue=False, bg=PALETTE['panel'], highlightthickness=0, troughcolor=PALETTE['card_border'])
        self.threshold_slider.pack(side='left', padx=6)
        self.threshold_val_lbl = tk.Label(match_frame, text=f"{self.match_threshold.get():.2f}", font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted'])
        self.threshold_val_lbl.pack(side='left')

        # test image preview
        self.test_canvas = tk.Canvas(right, width=360, height=360, bg=PALETTE['card_border'], highlightthickness=0)
        self.test_canvas.pack(pady=8, padx=12)

        # results list
        self.results_list = tk.Listbox(right, height=8, font=SMALL_FONT, bg=PALETTE['card_border'], fg=PALETTE['text'], selectbackground=PALETTE['accent_hover'], selectforeground=PALETTE['text'])
        self.results_list.pack(fill='both', expand=True, padx=12)
        self.results_list.bind('<<ListboxSelect>>', lambda e: self.on_result_select())

        

        # statusbar
        try:
            self.statusbar = tk.Label(self.root, text='Ready', anchor='w', bg=PALETTE['bg'], fg=PALETTE['muted'])
            self.statusbar.pack(side='bottom', fill='x')
        except Exception:
            self.statusbar = None

        # Call initial theme application to ensure all dynamic elements are colored
        # initialize batch UI (right-side panel)
        try:
            self.init_batch_ui(self.right_frame)
        except Exception:
            pass
        self.apply_theme()


    def _parse_detector_faces(self, det_out, img_shape):
        """Normalize detector output into list of dicts with box(x,y,w,h), landmarks(np.array 5x2) and score"""
        # This function contains complex logic for parsing YuNet's raw output.
        # It handles scaling for normalized vs. pixel coordinates.
        faces_arr = None
        if isinstance(det_out, tuple) or isinstance(det_out, list):
            faces_arr = det_out[1] if len(det_out) >= 2 and hasattr(det_out[1], 'shape') else det_out
        else:
            faces_arr = det_out

        parsed = []
        h, w = img_shape[:2]
        
        # Simplified parsing for the expected YuNet NumPy array output
        if hasattr(faces_arr, 'ndim') and faces_arr.ndim >= 2 and faces_arr.shape[0] > 0:
            for row in faces_arr:
                row = np.asarray(row)
                if row.size >= 15:
                    x, y, bw, bh = row[0:4].astype(int)
                    lm = row[4:14].reshape(5,2).astype(float)
                    score = float(row[14])
                    
                    # Assume landmarks are normalized or scaled correctly by YuNet's wrapper 
                    # and check bounding box sanity
                    
                    if lm.max() <= 1.01:
                        # Rescale normalized landmarks
                        lm[:,0] = lm[:,0] * w
                        lm[:,1] = lm[:,1] * h
                    
                    parsed.append({'box': (int(x), int(y), int(bw), int(bh)), 'landmarks': lm, 'score': score})
        return parsed

    # ---------------- Batch Search UI & helpers ----------------
    def init_batch_ui(self, parent_frame):
        """
        Create a right-side Batch panel inside `parent_frame`.
        Call this from your GUI setup after creating other panes.
        """
        self.batch_container = tk.Frame(parent_frame, bd=2, relief="groove")
        # make the batch panel fill the parent right frame so it occupies the full right panel
        self.batch_container.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        title = tk.Label(self.batch_container, text="Batch Search", font=("Segoe UI", 10, "bold"))
        title.pack(anchor="nw", padx=6, pady=(4,2))

        btn_frame = tk.Frame(self.batch_container)
        btn_frame.pack(anchor="nw", padx=6, pady=2, fill="x")

        self.load_batch_btn = tk.Button(btn_frame, text="📂 Load Batch", command=self.load_batch_images)
        self.load_batch_btn.pack(side="left", padx=(0,4))
        self.run_batch_btn = tk.Button(btn_frame, text="🔎 Run Batch Search", command=self.match_batch_search)
        self.run_batch_btn.pack(side="left")

        # status label
        self.batch_status_var = tk.StringVar(value="No batch loaded")
        self.batch_status_label = tk.Label(self.batch_container, textvariable=self.batch_status_var, anchor="w")
        self.batch_status_label.pack(fill="x", padx=6, pady=(4,6))

        # scrollable thumbnail area
        canvas_frame = tk.Frame(self.batch_container)
        canvas_frame.pack(fill="both", expand=False, padx=6, pady=(0,6))

        # taller canvas to allow more thumbnails and preview area
        self.batch_canvas = tk.Canvas(canvas_frame, height=360)
        self.batch_hscroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.batch_canvas.xview)
        self.batch_canvas.configure(xscrollcommand=self.batch_hscroll.set)
        self.batch_canvas.pack(side="top", fill="x")
        self.batch_hscroll.pack(side="top", fill="x")

        # inner frame inside canvas to hold thumbnail canvases
        self.batch_thumb_frame = tk.Frame(self.batch_canvas)
        self.batch_canvas.create_window((0,0), window=self.batch_thumb_frame, anchor="nw")
        self.batch_thumb_frame.bind("<Configure>", lambda e: self.batch_canvas.configure(scrollregion=self.batch_canvas.bbox("all")))

        # internal storage
        self.batch_paths = []
        self.batch_thumb_images = []        # keep ImageTk objects to avoid GC
        self.batch_thumb_canvases = []      # Canvas widgets for each thumb (to draw border)
        self.batch_queue = queue.Queue()    # for thread -> mainloop updates
        self.batch_worker = None
        # polling for background thread updates
        self._start_batch_queue_poller()

    def _start_batch_queue_poller(self):
        def poll():
            try:
                while True:
                    action, payload = self.batch_queue.get_nowait()
                    if action == "update_status":
                        self.batch_status_var.set(payload)
                    elif action == "append_result":
                        idx, text = payload
                        try:
                            self.results_list.insert(tk.END, text)
                        except Exception:
                            print(text)
                    elif action == "highlight_thumb":
                        idx, matched_faces = payload
                        self._draw_thumb_border(idx, highlight=True)
                    elif action == "unhighlight_thumb":
                        idx = payload
                        self._draw_thumb_border(idx, highlight=False)
                    elif action == "set_status_text":
                        self.batch_status_var.set(payload)
            except queue.Empty:
                pass
            self.root.after(150, poll)
        self.root.after(150, poll)

    def load_batch_images(self):
        """
        Let user pick multiple images and populate the thumbnail panel.
        """
        paths = filedialog.askopenfilenames(title="Select images for batch",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not paths:
            return

        # clear prior thumbnails
        for c in getattr(self, "batch_thumb_canvases", []):
            c.destroy()
        self.batch_thumb_canvases = []
        self.batch_thumb_images = []
        self.batch_paths = list(paths)

        thumb_h = 100
        pad_x = 6

        for idx, p in enumerate(self.batch_paths):
            try:
                pil_img = Image.open(p).convert("RGB")
                w, h = pil_img.size
                new_h = thumb_h
                new_w = int(w * (new_h / h))
                pil_img.thumbnail((new_w, new_h), Image.LANCZOS)

                cw = new_w + 8
                ch = new_h + 8
                c = tk.Canvas(self.batch_thumb_frame, width=cw, height=ch, bd=0, highlightthickness=0)
                c.grid(row=0, column=idx, padx=(0 if idx == 0 else pad_x, 0))
                photo = ImageTk.PhotoImage(pil_img)
                self.batch_thumb_images.append(photo)
                img_x = 4
                img_y = 4
                c.create_image(img_x, img_y, image=photo, anchor="nw")
                c.image_path = p
                c.image_index = idx
                c.bind("<Button-1>", lambda ev, i=idx: self.on_select_thumbnail(i))
                self.batch_thumb_canvases.append(c)
                self._draw_thumb_border(idx, highlight=False)
            except Exception as e:
                print(f"Failed to load thumbnail {p}: {e}")

        self.batch_status_var.set(f"{len(self.batch_paths)} images loaded")
        try:
            self.results_list.delete(0, tk.END)
        except Exception:
            pass

    def _draw_thumb_border(self, idx, highlight=False):
        """
        Draw or clear a green border for thumbnail `idx`.
        """
        if idx < 0 or idx >= len(self.batch_thumb_canvases):
            return
        c = self.batch_thumb_canvases[idx]
        c.delete("border_rect")
        if highlight:
            w = int(c.cget("width"))
            h = int(c.cget("height"))
            c.create_rectangle(2, 2, w-2, h-2, outline="#00aa00", width=4, tags=("border_rect",))
        else:
            w = int(c.cget("width"))
            h = int(c.cget("height"))
            c.create_rectangle(2, 2, w-2, h-2, outline="#444444", width=1, tags=("border_rect",))

    def on_select_thumbnail(self, idx):
        """
        Called when user clicks a thumbnail. Select the corresponding result row.
        """
        try:
            path = self.batch_paths[idx]
        except Exception:
            return
        if hasattr(self, "load_test_image"):
            try:
                self.load_test_image(path)
            except Exception:
                pass
        try:
            self.results_list.selection_clear(0, tk.END)
            self.results_list.selection_set(idx)
            self.results_list.see(idx)
        except Exception:
            pass

    def _detect_and_match_single(self, path):
        """
        Run full pipeline on one image path. Return a dict:
          {
            "path": path,
            "matches": [ { "bbox": (x,y,w,h), "distance": float, "matched_name": str|None, "face_index": int }, ... ],
            "best_match": same-structure-or-None
          }
        """
        # load image
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        try:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
        except Exception:
            pass
        det_out = self.yunet.detect(img)
        faces = self._parse_detector_faces(det_out, img.shape)

        # require a session user
        sess = getattr(self, 'current_session_user', '') or self.session_var.get().strip()
        users = {}
        user_dir = ENROLL_DIR / sess
        if user_dir.exists() and user_dir.is_dir():
            npz = user_dir / 'embeddings.npz'
            if npz.exists():
                data = np.load(str(npz))
                users[sess] = data['embeddings']

        threshold = float(self.match_threshold.get()) if hasattr(self, 'match_threshold') else 1.25

        matches = []
        for i, f in enumerate(faces):
            bx, by, bw, bh = f['box']
            lm = f.get('landmarks')
            if lm is None:
                lm = np.array([[bx + bw/4, by + bh/4],[bx + 3*bw/4, by + bh/4],[bx + bw/2, by + bh/2],[bx + bw/4, by + 3*bh/4],[bx + 3*bw/4, by + 3*bh/4]])
            aligned = align_and_crop(img, lm.flatten(), output_size=112)
            emb = generate_arcface_embedding(aligned)

            matched_name = None
            distance = None
            if users:
                embs = users.get(sess)
                if embs is not None:
                    dists = [calculate_distance(emb, e) for e in embs]
                    min_dist = float(np.min(dists))
                    distance = min_dist
                    if min_dist <= threshold:
                        matched_name = sess

            matches.append({"bbox": (bx, by, bw, bh), "distance": distance, "matched_name": matched_name, "face_index": i})

        valid = [m for m in matches if m['distance'] is not None]
        best = min(valid, key=lambda m: m['distance']) if valid else None
        return {"path": path, "matches": matches, "best_match": best}

    def match_batch_search(self, threshold=None):
        """
        Run batch search over loaded images. Runs in background thread and updates UI as each image is processed.
        """
        if not getattr(self, "batch_paths", None):
            messagebox.showinfo("Batch Search", "No batch loaded. Click '📂 Load Batch' first.")
            return

        if not getattr(self, "current_session_user", None):
            messagebox.showwarning("Batch Search", "No session user selected / enrolled. Please set the session or enroll a user.")
            return

        if threshold is None:
            threshold = getattr(self, "match_threshold", None)
        # if threshold is a Tk variable (DoubleVar), convert to float
        if hasattr(threshold, 'get'):
            try:
                threshold = float(threshold.get())
            except Exception:
                # fallback: try direct float conversion
                threshold = float(threshold)
        if threshold is None:
            threshold = 1.25

        try:
            self.load_batch_btn.config(state="disabled")
            self.run_batch_btn.config(state="disabled")
        except Exception:
            pass

        def worker(paths, thr, out_q):
            # ensure thr is a float (handle Tk DoubleVar)
            try:
                if hasattr(thr, 'get'):
                    thr = float(thr.get())
                else:
                    thr = float(thr)
            except Exception:
                thr = 1.25
            out_q.put(("set_status_text", "Batch search running..."))
            for idx, p in enumerate(paths):
                try:
                    info = self._detect_and_match_single(p)
                    best = info.get("best_match")
                    text = f"Image {idx+1}: "
                    matched_any = False
                    if best and (best["distance"] is not None) and best["distance"] < thr:
                        matched_any = True
                        text += f"Matched {best.get('matched_name') or 'SESSION'} at Dist={best['distance']:.3f}"
                    else:
                        if info["matches"]:
                            text += "No match"
                        else:
                            text += "No faces found"
                    out_q.put(("append_result", (idx, text)))
                    if matched_any:
                        out_q.put(("highlight_thumb", (idx, info["matches"])))
                    else:
                        out_q.put(("unhighlight_thumb", idx))
                except Exception as e:
                    out_q.put(("append_result", (idx, f"Image {idx+1}: Error: {e}")))
            out_q.put(("set_status_text", "Batch search complete"))
            out_q.put(("update_status", "Batch finished"))

        self.batch_worker = threading.Thread(target=worker, args=(list(self.batch_paths), threshold, self.batch_queue), daemon=True)
        self.batch_worker.start()

        def reenable_buttons_when_done():
            if self.batch_worker and self.batch_worker.is_alive():
                self.root.after(500, reenable_buttons_when_done)
            else:
                try:
                    self.load_batch_btn.config(state="normal")
                    self.run_batch_btn.config(state="normal")
                except Exception:
                    pass
        reenable_buttons_when_done()

    def refresh_enrolled_users(self):
        """Refresh the OptionMenu entries from enrolled users on disk"""
        opts = ['']
        if ENROLL_DIR.exists():
            for d in ENROLL_DIR.iterdir():
                if d.is_dir():
                    opts.append(d.name)
        # rebuild menu
        menu = self.session_menu['menu']
        menu.delete(0, 'end')
        for o in opts:
            menu.add_command(label=o, command=lambda v=o: self.session_var.set(v))

    def set_session_user(self):
        val = self.session_var.get().strip()
        if val == '':
            messagebox.showwarning('Session', 'Please select a session user')
            return
        self.current_session_user = val
        messagebox.showinfo('Session', f'Session user set to: {val}')
        # display user's enrolled views if available
        try:
            self.show_session_user_views(val)
        except Exception as e:
             self.status(f"Error showing views: {e}")

    def _detect_first_face(self, detector, img_bgr):
        """Detect and return the first face dict (box,landmarks) or None."""
        h,w = img_bgr.shape[:2]
        try:
            detector.setInputSize((w,h))
        except Exception:
            pass
        out = detector.detect(img_bgr)
        faces = None
        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and hasattr(out[1], 'shape'):
                faces = out[1]
            else:
                faces = out
        else:
            faces = out
        if faces is None:
            return None
        # if matrix output
        if hasattr(faces, 'ndim') and faces.ndim >= 2 and faces.shape[0] > 0:
            row = faces[0]
            if row.size >= 15:
                b = row[0:4].astype(int)
                lm = row[4:14].reshape(5,2)
                return {'box': tuple(b.tolist()), 'landmarks': lm}
        # dict-list style
        if isinstance(faces, (list,)) and len(faces) > 0 and isinstance(faces[0], dict):
            d = faces[0]
            b = d.get('box')
            lm = d.get('landmarks')
            return {'box': tuple(b), 'landmarks': np.asarray(lm).reshape(5,2) if lm is not None else None}
        return None

    def load_test_image(self, path=None):
        """Open a test image (or use provided path), display preview, and clear previous overlays."""
        if path is None:
            path = filedialog.askopenfilename(title='Select test image', filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.tif *.tiff'), ('All','*.*')])
        if not path:
            return
        self.test_path = Path(path)
        # load and create thumbnail for canvas
        try:
            pil = Image.open(str(self.test_path)).convert('RGB')
            # scale to fit canvas (approx)
            max_w, max_h = 360, 360
            pil.thumbnail((max_w, max_h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil)
            self.test_thumb = imgtk
            self.test_canvas.delete('all')
            cw = self.test_canvas.winfo_width() or max_w
            ch = self.test_canvas.winfo_height() or max_h
            self.test_canvas.create_image(cw//2, ch//2, image=imgtk, anchor='center')
        except Exception as e:
            messagebox.showerror('Image', f'Failed to load test image: {e}')
            return
        # clear previous results
        try:
            self.results_list.delete(0, tk.END)
        except Exception:
            pass

    def match_test(self):
        """Match a single test image against the current session user (first detected face)."""
        if self.test_path is None:
            messagebox.showwarning('Missing', 'Load a test image first')
            return
        img = cv2.imread(str(self.test_path))
        if img is None:
            messagebox.showerror('Error', 'Failed to read test image')
            return

        info = self._detect_first_face(self.yunet, img)
        if info is None or info.get('landmarks') is None:
            messagebox.showinfo('No face', 'No face detected in test image')
            return
        lm = np.asarray(info['landmarks']).astype(np.float32)
        aligned = align_and_crop(img, lm.flatten(), output_size=112)
        emb = generate_arcface_embedding(aligned)

        sess = getattr(self, 'current_session_user', '') or self.session_var.get().strip()
        if not sess:
            messagebox.showwarning('Session', 'Please set a session user before matching')
            return

        # load session embeddings
        user_dir = ENROLL_DIR / sess
        if not user_dir.exists():
            messagebox.showinfo('No enrollments', f'No enrollment found for session user: {sess}')
            return
        npz = user_dir / 'embeddings.npz'
        if not npz.exists():
            messagebox.showinfo('No enrollments', f'No embeddings file for user: {sess}')
            return
        data = np.load(str(npz))
        embs = data['embeddings']

        dists = [calculate_distance(emb, e) for e in embs]
        min_dist = float(np.min(dists))
        matched = min_dist <= float(self.match_threshold.get())

        # update results list and per-view displays
        try:
            self.results_list.delete(0, tk.END)
            self.results_list.insert(tk.END, f'User: {sess} min_dist={min_dist:.3f} matched={matched} per_view={ [round(x,3) for x in dists] }')
        except Exception:
            pass

        # show aligned crop in view canvases and labels
        try:
            thumb_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(thumb_rgb)
            pil.thumbnail((120,120))
            imgtk = ImageTk.PhotoImage(pil)
            for i, key in enumerate(['front','left','right']):
                cinfo = self.view_canvases.get(key)
                if cinfo is None:
                    continue
                c = cinfo['canvas']
                c.delete('all')
                c.create_image(c.winfo_width()//2 or 60, c.winfo_height()//2 or 60, image=imgtk, anchor='center')
                cinfo['image'] = imgtk
                # set label text to corresponding dist if available
                lbl = cinfo.get('label')
                if lbl is not None:
                    try:
                        lbl.config(text=f"Dist: {dists[i]:.3f}\nBest: {['Front','Left','Right'][i]}")
                    except Exception:
                        lbl.config(text=f"Dist: -")
        except Exception:
            pass

    def match_all_faces(self):
        """Detect all faces in the loaded test image and match each against enrollments."""
        if self.test_path is None:
            messagebox.showwarning('Missing', 'Load a test image first')
            return
        # read original image
        img = cv2.imread(str(self.test_path))
        if img is None:
            messagebox.showerror('Error', 'Failed to read test image')
            return

        try:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
        except Exception:
            pass
        det_out = self.yunet.detect(img)
        faces = self._parse_detector_faces(det_out, img.shape)
        if len(faces) == 0:
            messagebox.showinfo('No faces', 'No faces detected')
            return

        # require a session user
        sess = getattr(self, 'current_session_user', '') or self.session_var.get().strip()
        if not sess:
            messagebox.showwarning('Session', 'Please set a session user before matching')
            return
        # load only the session user's enrollments
        users = {}
        user_dir = ENROLL_DIR / sess
        if user_dir.exists() and user_dir.is_dir():
            npz = user_dir / 'embeddings.npz'
            if npz.exists():
                data = np.load(str(npz))
                users[sess] = data['embeddings']

        if len(users) == 0:
            messagebox.showinfo('No enrollments', f'No enrollment found for session user: {sess}')
            return

        threshold = float(self.match_threshold.get()) if hasattr(self, 'match_threshold') else 1.25

        results = []
        for i, f in enumerate(faces):
            bx, by, bw, bh = f['box']
            lm = f.get('landmarks')
            if lm is None:
                # fallback to bbox center landmarks
                lm = np.array([[bx + bw/4, by + bh/4],[bx + 3*bw/4, by + bh/4],[bx + bw/2, by + bh/2],[bx + bw/4, by + 3*bh/4],[bx + 3*bw/4, by + 3*bh/4]])
            aligned = align_and_crop(img, lm.flatten(), output_size=112)
            emb = generate_arcface_embedding(aligned)

            # compare only against the session user's views
            per_user = []
            for user, embs in users.items():
                dists = [calculate_distance(emb, e) for e in embs]
                min_dist = float(np.min(dists))
                per_user.append({'user': user, 'min_dist': min_dist, 'dists': dists})
            per_user = sorted(per_user, key=lambda x: x['min_dist'])
            best = per_user[0]
            matched = best['min_dist'] <= threshold

            # create thumbnail for UI
            try:
                thumb_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(thumb_rgb)
                pil.thumbnail((120,120))
                imgtk = ImageTk.PhotoImage(pil)
            except Exception:
                imgtk = None

            results.append({'index': i, 'box': (bx, by, bw, bh), 'landmarks': lm, 'aligned': aligned, 'embedding': emb, 'best': best, 'matched': matched, 'thumb': imgtk, 'per_user': per_user})

        # store and display results
        self.multi_results = results
        self._draw_test_overlays()
        self.results_list.delete(0, tk.END)
        for r in results:
            # only display the session user's result
            self.results_list.insert(tk.END, f"Face #{r['index']+1}  {sess}  min={r['best']['min_dist']:.3f} matched={r['matched']}")

    def _draw_test_overlays(self):
        # remove previous overlays
        if hasattr(self, 'test_overlay_items'):
            for iid in self.test_overlay_items:
                try:
                    self.test_canvas.delete(iid)
                except Exception:
                    pass
        self.test_overlay_items = []

        # need scale: compute from displayed thumbnail to original
        if not hasattr(self, 'test_thumb') or self.test_thumb is None:
            return
        # get original size
        img_bgr = cv2.imread(str(self.test_path))
        if img_bgr is None:
            return
        orig_h, orig_w = img_bgr.shape[:2]
        disp_w = self.test_thumb.width()
        disp_h = self.test_thumb.height()
        sx = disp_w / float(orig_w)
        sy = disp_h / float(orig_h)

        # canvas center pos
        cw = self.test_canvas.winfo_width() or disp_w
        ch = self.test_canvas.winfo_height() or disp_h
        cx = cw // 2
        cy = ch // 2
        # compute top-left of displayed image on canvas
        img_x0 = cx - disp_w // 2
        img_y0 = cy - disp_h // 2

        for i, r in enumerate(self.multi_results):
            bx, by, bw, bh = r['box']
            x1 = int(img_x0 + bx * sx)
            y1 = int(img_y0 + by * sy)
            x2 = int(img_x0 + (bx + bw) * sx)
            y2 = int(img_y0 + (by + bh) * sy)
            color = PALETTE['accent_alt'] if r['matched'] else PALETTE['danger']
            rect = self.test_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
            txt = self.test_canvas.create_text(x1 + 4, max(4, y1 - 10), text=f"{r['best']['user']} {r['best']['min_dist']:.3f}", anchor='nw', fill=color, font=('Arial', 10, 'bold'))
            self.test_overlay_items.extend([rect, txt])

    def on_result_select(self):
        sel = self.results_list.curselection()
        if not sel:
            return
        idx = sel[0]
        r = self.multi_results[idx]
        # show thumbnail in the three view canvases (for quick inspection)
        for key in ['front','left','right']:
            cinfo = self.view_canvases.get(key)
            if cinfo is None:
                continue
            c = cinfo['canvas']
            c.delete('all')
            if r['thumb'] is not None:
                cw = c.winfo_width() or 120
                ch = c.winfo_height() or 120
                c.create_image(cw//2, ch//2, image=r['thumb'], anchor='center')
                cinfo['image'] = r['thumb']
            # set label to show best dist for this selected face
            dists = [round(d, 3) for d in r['best']['dists']]
            best_idx = r['best']['dists'].index(r['best']['min_dist'])
            best_pose = ['Front','Left','Right'][best_idx]
            
            cinfo['label'].config(text=f"Dist: {dists[best_idx]:.3f}\nBest: {best_pose}")


    # (rest of the functions remain the same)
    
    def show_session_user_views(self, uid: str):
        """Load and display the enrolled aligned images (front/left/right) for the given user.
        Looks for face_pipeline/enrollments/<uid>/(front|left|right).png saved during enrollment.
        """
        user_dir = ENROLL_DIR / uid
        for key in ['front','left','right']:
            cinfo = self.view_canvases.get(key)
            if cinfo is None:
                continue
            c = cinfo['canvas']
            c.delete('all')
            img_path = user_dir / f"{key}.png"
            if img_path.exists():
                try:
                    pil = Image.open(str(img_path)).convert('RGB')
                    pil.thumbnail((120,120))
                    imgtk = ImageTk.PhotoImage(pil)
                    cw = c.winfo_width() or 120
                    ch = c.winfo_height() or 120
                    c.create_image(cw//2, ch//2, image=imgtk, anchor='center')
                    cinfo['image'] = imgtk
                    cinfo['label'].config(text=f"{key.capitalize()}")
                except Exception:
                    cinfo['label'].config(text=f"{key}: (error)")
            else:
                cinfo['label'].config(text=f"{key}: (missing)")

    def _enroll_user_from_files(self, uid, front_path, left_path, right_path, verbose=True):
        """Enroll a user given explicit file paths for three views. Returns True on success."""
        user_dir = ENROLL_DIR / uid
        ensure_dir(user_dir)
        detector = self.yunet
        arc = self.arcface
        embeddings = []
        for key, p in [('front', front_path), ('left', left_path), ('right', right_path)]:
            if p is None:
                embeddings.append(np.zeros((512,), dtype=np.float32))
                continue
            p = Path(p)
            if not p.exists():
                if verbose:
                    print(f"[WARN] {key} image missing: {p}")
                embeddings.append(np.zeros((512,), dtype=np.float32))
                continue
            img = cv2.imread(str(p))
            if img is None:
                if verbose:
                    print(f"[ERROR] failed to read {p}")
                return False
            info = self._detect_first_face(detector, img)
            if info is None or info.get('landmarks') is None:
                h,w = img.shape[:2]
                box = (max(0,w//2-56), max(0,h//2-56), 112,112)
                lm = np.array([[box[0]+28, box[1]+28],[box[0]+84, box[1]+28],[box[0]+56, box[1]+56],[box[0]+28, box[1]+84],[box[0]+84, box[1]+84]])
            else:
                lm = np.asarray(info['landmarks']).astype(np.float32)
            aligned = align_and_crop(img, lm.flatten(), output_size=112)
            emb = generate_arcface_embedding(aligned)
            embeddings.append(emb)
            outp = user_dir / f"{key}.png"
            # save aligned image for verification display
            cv2.imwrite(str(outp), aligned) 
        embeddings = np.stack(embeddings, axis=0)
        npz = user_dir / 'embeddings.npz'
        np.savez_compressed(str(npz), embeddings=embeddings)
        if verbose:
            print(f"Saved embeddings for {uid} -> {npz}")
        return True

    def enroll_user(self):
        """GUI handler to enroll a user by selecting three view images."""
        uid = self.user_id.get().strip()
        if not uid:
            messagebox.showwarning('Enroll', 'Please enter a User ID before enrolling')
            return

        # ask for three images if not already loaded in self.paths
        front = self.paths.get('front')
        left = self.paths.get('left')
        right = self.paths.get('right')

        if not front:
            front = filedialog.askopenfilename(title='Select FRONT image')
        if not left:
            left = filedialog.askopenfilename(title='Select LEFT image')
        if not right:
            right = filedialog.askopenfilename(title='Select RIGHT image')

        if not any([front, left, right]):
            messagebox.showinfo('Enroll', 'Enrollment cancelled — no images selected')
            return

        ok = self._enroll_user_from_files(uid, front, left, right, verbose=True)
        if ok:
            messagebox.showinfo('Enroll', f'Enrolled user: {uid}')
            # refresh session list and set current session
            try:
                self.refresh_enrolled_users()
                self.session_var.set(uid)
                self.set_session_user()
            except Exception:
                pass

    def batch_enroll_csv(self):
        """Prompt for a CSV file with rows: user_id,front_path,left_path,right_path and enroll each user."""
        csv_path = filedialog.askopenfilename(title='Select batch enroll CSV', filetypes=[('CSV','*.csv'),('All files','*.*')])
        if not csv_path:
            return
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                count = 0
                for row in reader:
                    if not row:
                        continue
                    # allow header or variable-length rows
                    uid = row[0].strip()
                    front = row[1].strip() if len(row) > 1 else ''
                    left = row[2].strip() if len(row) > 2 else ''
                    right = row[3].strip() if len(row) > 3 else ''
                    if not uid:
                        continue
                    ok = self._enroll_user_from_files(uid, front or None, left or None, right or None, verbose=False)
                    if ok:
                        count += 1
            messagebox.showinfo('Batch Enroll', f'Enrolled {count} users from CSV')
            self.refresh_enrolled_users()
        except Exception as e:
            messagebox.showerror('Batch Enroll', f'Failed to process CSV: {e}')

    def export_enrollments(self):
        """Export all enrollment folders to a chosen directory (copies folders)."""
        if not ENROLL_DIR.exists():
            messagebox.showinfo('Export', 'No enrollments to export')
            return
        out_dir = filedialog.askdirectory(title='Select export directory')
        if not out_dir:
            return
        out_dir = Path(out_dir)
        try:
            # copy each enrollment subfolder
            count = 0
            for d in ENROLL_DIR.iterdir():
                if d.is_dir():
                    dest = out_dir / d.name
                    # copytree requires dest not exist
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(d, dest)
                    count += 1
            messagebox.showinfo('Export', f'Exported {count} enrollment(s) to {out_dir}')
        except Exception as e:
            messagebox.showerror('Export', f'Failed to export enrollments: {e}')


def main():
    root = tk.Tk()
    app = WholeFaceScanGUI(root)
    app.refresh_enrolled_users() # Initial load of user list
    root.mainloop()


if __name__ == '__main__':
    main()