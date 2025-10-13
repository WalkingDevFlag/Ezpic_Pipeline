"""
whole_face_scan_gui.py

Tkinter GUI for multi-view enrollment (front/left/right) and matching.
Stores enrollments at face_pipeline/enrollments/<user_id>/embeddings.npz

Controls:
- Enrollment panel: user id input, three load buttons (front/left/right), thumbnails, Enroll button
- Matching panel: load test image, Match button, results list

"""
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

import sys
ROOT = Path(__file__).parent.resolve()
WORKING = ROOT / 'WORKING CODE'
if str(WORKING) not in sys.path:
    sys.path.insert(0, str(WORKING))

from arcface_pipeline import load_yunet_model, load_arcface_model
from face_alignment import align_and_crop
from arcface_embedder import generate_arcface_embedding, calculate_distance
import shutil
import csv

ENROLL_DIR = Path('face_pipeline') / 'enrollments'


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# UI themes / palettes
PALETTE_LIGHT = {
    'bg': '#f5f7fb',         # window background
    'panel': '#ffffff',      # panel background
    'muted': '#6b7280',      # secondary text
    'accent': '#2563eb',     # primary blue
    'accent_hover': '#1d4ed8',
    'accent_light': '#eff6ff',
    'accent_alt': '#10b981', # green
    'accent_alt_hover': '#059669',
    'danger': '#ef4444',
    'danger_hover': '#dc2626',
    'card_border': '#e6edf3',
    'dark': '#0f172a',
    'text': '#0f172a'
}

PALETTE_DARK = {
    # Based on provided palette mapping for dark mode
    # PRIMARY_DARK_BACKGROUND
    'bg': '#030812',
    # SECONDARY_BACKGROUND / PANEL_BG
    'panel': '#020764',
    # Muted helper text
    'muted': '#888888',
    # PRIMARY_ACCENT / ACTIVE_STATE
    'accent': '#025EC4',
    # CARD_BACKGROUND / ACCENT_HOVER
    'accent_hover': '#043780',
    # Neutral/secondary surfaces for light-styled buttons
    'accent_light': '#043780',
    # SECONDARY_ACCENT / HIGHLIGHT (teal)
    'accent_alt': '#0ECCED',
    # Slightly darker teal for hover
    'accent_alt_hover': '#0BB6D1',
    # Errors
    'danger': '#E74C3C',
    'danger_hover': '#c0392b',
    # Card/border separators
    'card_border': '#043780',
    # General light text on dark
    'dark': '#E0E0E0',
    'text': '#E0E0E0'
}

# Default palette
PALETTE = dict(PALETTE_LIGHT)

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
        self.theme = 'light'
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

        # now load models (statusbar created by create_ui)
        self.load_models()

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
            if hasattr(self, 'main'):
                self.main.configure(bg=P['bg'])
            if hasattr(self, 'left_frame'):
                self.left_frame.configure(bg=P['panel'], highlightbackground=P['card_border'])
            if hasattr(self, 'right_frame'):
                self.right_frame.configure(bg=P['panel'], highlightbackground=P['card_border'])
            # Entry and dropdown styling (best-effort on Tk)
            if hasattr(self, 'user_entry'):
                try:
                    self.user_entry.configure(bg=P['accent_light'], fg=P['text'], insertbackground=P['text'], highlightthickness=1, highlightbackground=P['card_border'], relief='flat')
                except Exception:
                    pass
            if hasattr(self, 'session_menu'):
                try:
                    self.session_menu.configure(bg=P['accent_light'], fg=P['text'], activebackground=P['accent_hover'], activeforeground=P['text'], highlightthickness=1)
                except Exception:
                    pass
            # Statusbar
            if hasattr(self, 'statusbar') and self.statusbar is not None:
                self.statusbar.configure(bg=P['bg'], fg=P['muted'])
        except Exception:
            pass

    def toggle_theme(self):
        # Switch between light and dark palettes
        if self.theme == 'light':
            self.theme = 'dark'
            for k in list(PALETTE.keys()):
                PALETTE[k] = PALETTE_DARK.get(k, PALETTE[k])
        else:
            self.theme = 'light'
            for k in list(PALETTE.keys()):
                PALETTE[k] = PALETTE_LIGHT.get(k, PALETTE[k])
        self.palette = PALETTE
        self.apply_theme()

    def make_button(self, parent, text, command=None, kind='accent'):
        """Create a styled button with hover effects.
        kind: 'accent' | 'alt' | 'danger' | 'light'
        """
        P = self.palette
        if kind == 'accent':
            bg = P['accent']; hover = P['accent_hover']; fg = 'white'
        elif kind == 'alt':
            bg = P['accent_alt']; hover = P['accent_alt_hover']; fg = 'white'
        elif kind == 'danger':
            bg = P['danger']; hover = P['danger_hover']; fg = 'white'
        else:
            bg = P['accent_light']; hover = P['accent_light']; fg = P['text']
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, font=BTN_FONT, relief='flat', padx=10, pady=6, activebackground=hover, activeforeground=fg, cursor='hand2')
        # hover effects
        def _enter(e):
            e.widget.configure(bg=hover)
        def _leave(e):
            e.widget.configure(bg=bg)
        btn.bind('<Enter>', _enter)
        btn.bind('<Leave>', _leave)
        return btn

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
        # root background
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
        self.enroll_status = tk.Label(left, text='Ready', anchor='w', bg=PALETTE['panel'])
        self.enroll_status.pack(fill='x', padx=12)

        # Session user selector (only one user is referenced per session)
        tk.Label(left, text='Session user:', font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(anchor='w', pady=(8,0), padx=12)
        self.session_var = tk.StringVar(value='')
        self.session_menu = tk.OptionMenu(left, self.session_var, '')
        self.session_menu.config(width=26)
        self.session_menu.pack(anchor='w', padx=12, pady=(4,6))
        sess_ctl = tk.Frame(left, bg=PALETTE['panel'])
        sess_ctl.pack(fill='x', padx=12)
        self.make_button(sess_ctl, text='Set Session', command=self.set_session_user, kind='alt').pack(side='left', padx=4, pady=6)
        self.make_button(sess_ctl, text='Refresh Users', command=self.refresh_enrolled_users, kind='light').pack(side='left', padx=4)
        # populate menu
        self.current_session_user = ''
        self.refresh_enrolled_users()

        # Matching panel
        tk.Label(right, text='Matching', font=HEADER_FONT, bg=PALETTE['panel'], fg=PALETTE['text']).pack(anchor='w', padx=12, pady=(12,6))
        match_frame = tk.Frame(right, bg=PALETTE['panel'])
        match_frame.pack(fill='x', pady=6, padx=12)
        self.make_button(match_frame, text='Load Test Image', command=self.load_test_image, kind='light').pack(side='left')
        self.make_button(match_frame, text='🔎 Match', command=self.match_test, kind='accent').pack(side='left', padx=8)
        self.make_button(match_frame, text='👥 Match All', command=self.match_all_faces, kind='danger').pack(side='left', padx=8)
        # threshold control
        tk.Label(match_frame, text='Threshold:', font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted']).pack(side='left', padx=(16,4))
        self.match_threshold = tk.DoubleVar(value=1.25)
        def _on_thresh(val):
            try:
                self.threshold_val_lbl.configure(text=f"{float(val):.2f}")
            except Exception:
                pass
        self.threshold_slider = tk.Scale(match_frame, from_=0.8, to=1.8, resolution=0.01, orient='horizontal', length=160, variable=self.match_threshold, command=_on_thresh, showvalue=False, bg=PALETTE['panel'], highlightthickness=0)
        self.threshold_slider.pack(side='left', padx=6)
        self.threshold_val_lbl = tk.Label(match_frame, text=f"{self.match_threshold.get():.2f}", font=LABEL_FONT, bg=PALETTE['panel'], fg=PALETTE['muted'])
        self.threshold_val_lbl.pack(side='left')

        # test image preview
        self.test_canvas = tk.Canvas(right, width=360, height=360, bg=PALETTE['bg'], highlightthickness=0)
        self.test_canvas.pack(pady=8, padx=12)

        # results list
        self.results_list = tk.Listbox(right, height=8, font=SMALL_FONT)
        self.results_list.pack(fill='both', expand=True, padx=12)
        self.results_list.bind('<<ListboxSelect>>', lambda e: self.on_result_select())

        # per-view display (front/left/right)
        pv = tk.Frame(right, bg=PALETTE['panel'])
        pv.pack(fill='x', pady=(6,0), padx=12)
        self.view_canvases = {}
        for i, key in enumerate(['front','left','right']):
            f = tk.Frame(pv)
            f.grid(row=0, column=i, padx=6)
            lbl = tk.Label(f, text=key.capitalize())
            lbl.pack()
            c = tk.Canvas(f, width=120, height=120, bg=PALETTE['panel'])
            c.pack()
            dlab = tk.Label(f, text='dist: -')
            dlab.pack()
            self.view_canvases[key] = {'canvas': c, 'label': dlab, 'image': None}

        # statusbar
        try:
            self.statusbar = tk.Label(self.root, text='Ready', anchor='w', bg=PALETTE['bg'], fg=PALETTE['muted'])
            self.statusbar.pack(side='bottom', fill='x')
        except Exception:
            self.statusbar = None

        # Apply theme styling to top-level elements
        self.apply_theme()

    def _parse_detector_faces(self, det_out, img_shape):
        """Normalize detector output into list of dicts with box(x,y,w,h), landmarks(np.array 5x2) and score"""
        faces_arr = None
        if isinstance(det_out, tuple) or isinstance(det_out, list):
            if len(det_out) >= 2 and hasattr(det_out[1], 'shape'):
                faces_arr = det_out[1]
            else:
                faces_arr = det_out
        else:
            faces_arr = det_out

        parsed = []
        h, w = img_shape[:2]
        # dict-style
        if isinstance(faces_arr, (list,)) and len(faces_arr) > 0 and isinstance(faces_arr[0], dict):
            for d in faces_arr:
                b = d.get('box')
                score = float(d.get('score', 0))
                lm = d.get('landmarks')
                if b is None:
                    continue
                # normalize box to x,y,w,h
                try:
                    if len(b) == 4:
                        x0, y0, x1, y1 = map(float, b)
                        # if values look normalized <=1
                        if max(x0, y0, x1, y1) <= 1.01:
                            x = int(x0 * w)
                            y = int(y0 * h)
                            x2 = int(x1 * w)
                            y2 = int(y1 * h)
                            bw = x2 - x
                            bh = y2 - y
                        else:
                            # could be xywh or xyxy; heuristics
                            if x1 > x0 and y1 > y0 and x1 <= w and y1 <= h:
                                # xyxy
                                x = int(x0); y = int(y0); bw = int(x1 - x0); bh = int(y1 - y0)
                            else:
                                x = int(x0); y = int(y0); bw = int(x1); bh = int(y1)
                except Exception:
                    continue
                landmarks = None
                if lm is not None:
                    try:
                        arr = np.asarray(lm).reshape(-1,2).astype(float)
                        if arr.max() <= 1.01:
                            arr[:,0] = arr[:,0] * w
                            arr[:,1] = arr[:,1] * h
                        landmarks = arr
                    except Exception:
                        landmarks = None
                parsed.append({'box': (int(x), int(y), int(bw), int(bh)), 'landmarks': landmarks, 'score': score})
        elif hasattr(faces_arr, 'ndim') and faces_arr.ndim >= 2:
            for row in faces_arr:
                row = np.asarray(row)
                if row.size >= 15:
                    x, y, bw, bh = row[0:4].astype(int)
                    lm = row[4:14].reshape(5,2).astype(float)
                    # if normalized
                    if lm.max() <= 1.01:
                        lm[:,0] = lm[:,0] * w
                        lm[:,1] = lm[:,1] * h
                    parsed.append({'box': (int(x), int(y), int(bw), int(bh)), 'landmarks': lm, 'score': float(row[14])})
        return parsed

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
            color = '#16c79a' if r['matched'] else '#e74c3c'
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
            cinfo['label'].config(text=f"best: {r['best']['user']}\n{r['best']['min_dist']:.3f}")

    # (statusbar is created in create_ui)

    def status(self, text: str):
        if hasattr(self, 'statusbar') and self.statusbar is not None:
            try:
                self.statusbar.config(text=text)
                return
            except Exception:
                pass
        print(text)

    def load_enroll_image(self, key: str):
        path = filedialog.askopenfilename(title=f'Select {key} image', filetypes=[('Images','*.jpg *.jpeg *.png')])
        if not path:
            return
        img = Image.open(path).convert('RGB')
        img.thumbnail((150,150))
        imgtk = ImageTk.PhotoImage(img)
        c = self.thumbs[key]['canvas']
        c.delete('all')
        # center
        cw = c.winfo_width() or 150
        ch = c.winfo_height() or 150
        c.create_image(cw//2, ch//2, image=imgtk, anchor='center')
        self.thumbs[key]['image'] = imgtk
        self.thumbs[key]['path'] = Path(path)
        self.enroll_status.config(text=f'Loaded {key} image')

    def enroll_user(self):
        uid = self.user_id.get().strip()
        if uid == '':
            messagebox.showwarning('Missing', 'Please enter a user id')
            return
        # require at least one view, but prefer three
        provided = [k for k in ['front','left','right'] if self.thumbs[k]['path'] is not None]
        if len(provided) == 0:
            messagebox.showwarning('Missing', 'Please load at least one view image')
            return
        self.enroll_status.config(text='Enrolling...')
        self.root.update()

        # delegate to helper using selected paths
        front = self.thumbs['front']['path']
        left = self.thumbs['left']['path']
        right = self.thumbs['right']['path']
        ok = self._enroll_user_from_files(uid, front, left, right)
        if ok:
            self.enroll_status.config(text=f'Enrolled {uid}')
            # automatically set this user as the session user
            self.current_session_user = uid
            try:
                self.session_var.set(uid)
            except Exception:
                pass
            self.refresh_enrolled_users()
            messagebox.showinfo('Enrolled', f'User {uid} enrolled with views: {provided}\nSet as session user')
            # show saved views on the right panel
            try:
                self.show_session_user_views(uid)
            except Exception:
                pass

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
                    cinfo['label'].config(text=f"{key}")
                except Exception:
                    cinfo['label'].config(text=f"{key}: (error)")
            else:
                cinfo['label'].config(text=f"{key}: (missing)")

    def _detect_first_face(self, detector, img_bgr):
        h,w = img_bgr.shape[:2]
        try:
            detector.setInputSize((w,h))
        except Exception:
            pass
        out = detector.detect(img_bgr)
        faces = None
        if isinstance(out, tuple) or isinstance(out, list):
            if len(out) >= 2 and hasattr(out[1], 'shape'):
                faces = out[1]
            else:
                faces = out
        else:
            faces = out
        if faces is None:
            return None
        if hasattr(faces, 'ndim') and faces.ndim >= 2 and faces.shape[0] > 0:
            row = faces[0]
            if row.size >= 15:
                b = row[0:4].astype(int)
                lm = row[4:14].reshape(5,2)
                return {'box': tuple(b.tolist()), 'landmarks': lm}
        if isinstance(faces, (list,)) and len(faces) > 0 and isinstance(faces[0], dict):
            d = faces[0]
            b = d.get('box')
            lm = d.get('landmarks')
            return {'box': tuple(b), 'landmarks': np.asarray(lm).reshape(5,2) if lm is not None else None}
        return None

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
            cv2.imwrite(str(outp), aligned)
        embeddings = np.stack(embeddings, axis=0)
        npz = user_dir / 'embeddings.npz'
        np.savez_compressed(str(npz), embeddings=embeddings)
        if verbose:
            print(f"Saved embeddings for {uid} -> {npz}")
        return True

    def batch_enroll_csv(self):
        path = filedialog.askopenfilename(title='Select CSV', filetypes=[('CSV','*.csv')])
        if not path:
            return
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                uid = row.get('user_id') or row.get('id') or row.get('user')
                front = row.get('front')
                left = row.get('left')
                right = row.get('right')
                if not uid:
                    continue
                self._enroll_user_from_files(uid, front, left, right, verbose=True)
        messagebox.showinfo('Batch', 'Batch enroll finished')

    def export_enrollments(self):
        # choose destination zip
        dst = filedialog.asksaveasfilename(title='Export enrollments to ZIP', defaultextension='.zip', filetypes=[('Zip','*.zip')])
        if not dst:
            return
        dst = Path(dst)
        import zipfile
        with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            if ENROLL_DIR.exists():
                for root, dirs, files in os.walk(ENROLL_DIR):
                    for f in files:
                        fp = Path(root) / f
                        zf.write(str(fp), arcname=str(fp.relative_to(ENROLL_DIR.parent)))
        messagebox.showinfo('Export', f'Exported enrollments -> {dst}')

    def load_test_image(self):
        path = filedialog.askopenfilename(title='Select test image', filetypes=[('Images','*.jpg *.jpeg *.png')])
        if not path:
            return
        self.test_path = Path(path)
        pil = Image.open(path).convert('RGB')
        pil.thumbnail((360,360))
        imgtk = ImageTk.PhotoImage(pil)
        self.test_thumb = imgtk
        self.test_canvas.delete('all')
        cw = self.test_canvas.winfo_width() or 360
        ch = self.test_canvas.winfo_height() or 360
        self.test_canvas.create_image(cw//2, ch//2, image=imgtk, anchor='center')
        self.status('Loaded test image')

    def match_test(self):
        if self.test_path is None:
            messagebox.showwarning('Missing', 'Load a test image first')
            return
        # detect
        img = cv2.imread(str(self.test_path))
        if img is None:
            messagebox.showerror('Error', 'Failed to read test image')
            return
        info = self._detect_first_face(self.yunet, img)
        if info is None or info.get('landmarks') is None:
            messagebox.showwarning('No face', 'No face detected in test image')
            return
        lm = np.asarray(info['landmarks']).astype(np.float32)
        aligned = align_and_crop(img, lm.flatten(), output_size=112)
        query_emb = generate_arcface_embedding(aligned)

        # require session user
        sess = getattr(self, 'current_session_user', '') or self.session_var.get().strip()
        if not sess:
            messagebox.showwarning('Session', 'Please set a session user before matching')
            return
        # load only session user embeddings
        user_dir = ENROLL_DIR / sess
        if not user_dir.exists() or not user_dir.is_dir():
            messagebox.showinfo('No enrollments', f'No enrollment found for session user: {sess}')
            return
        npz = user_dir / 'embeddings.npz'
        if not npz.exists():
            messagebox.showinfo('No enrollments', f'No embeddings found for session user: {sess}')
            return
        data = np.load(str(npz))
        embs = data['embeddings']

        dists = [calculate_distance(query_emb, e) for e in embs]
        min_dist = float(np.min(dists))
        matched = min_dist <= float(self.match_threshold.get())

        self.results_list.delete(0, tk.END)
        line = f"{sess}  min={min_dist:.3f}  matched={matched}  dists={[round(x,3) for x in dists]}"
        self.results_list.insert(tk.END, line)
        if matched:
            messagebox.showinfo('Match', f"Recognized as {sess} (dist={min_dist:.3f})")
        else:
            messagebox.showinfo('No match', f"No match for session user {sess} (best dist={min_dist:.3f})")


if __name__ == '__main__':
    root = tk.Tk()
    app = WholeFaceScanGUI(root)
    root.mainloop()
