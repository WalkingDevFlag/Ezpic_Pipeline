"""
ArcFace Multi-Face Search System
==================================

Enhanced face recognition for finding a target person in group photos.

Pipeline:
1. Load Reference: Single face of target person
2. Load Test Image: Group photo with multiple faces
3. Multi-Face Detection: Detect ALL faces in test image using YuNet
4. Generate All Embeddings: Create 512D vector for each detected face
5. Compare Against All: Calculate distance from reference to each face
6. Find Best Match: Return the face with minimum distance
7. Threshold Decision: Check if best match passes threshold (1.25)

Features:
- Detects and processes ALL faces in group photos
- Visual indication of all detected faces
- Highlights best match with confidence indicator
- LFW-validated threshold (95.20% accuracy)
- Handles 1-to-many comparisons efficiently

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from pathlib import Path

# Import pipeline modules
from arcface_pipeline import load_yunet_model, load_arcface_model, process_single_face_arcface
from arcface_embedder import calculate_distance, generate_arcface_embedding
from face_alignment import align_and_crop


class MultiFaceSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Multi-Face Search System")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1a1a2e')
        
        # Data storage
        self.reference_path = None
        self.reference_embedding = None
        self.test_path = None
        self.test_faces = []  # List of detected faces with embeddings
        self.best_match_idx = None
        self.distance_threshold = 1.25  # LFW optimal
        
        # Photo references
        self.photos = {}
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_ui()
    
    def load_models(self):
        """Pre-load models for faster processing"""
        self.root.title("Loading models...")
        self.root.update()
        
        try:
            self.yunet = load_yunet_model()
            self.arcface = load_arcface_model()
            print("✅ YuNet and ArcFace models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def create_ui(self):
        """Create the GUI layout"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#0f3460', height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🔍 Multi-Face Search System", 
                              font=('Arial', 22, 'bold'), bg='#0f3460', fg='white')
        title_label.pack(side='left', padx=20, pady=20)
        
        subtitle = tk.Label(title_frame, text="Find Target Person in Group Photos • 1-to-Many Comparison", 
                           font=('Arial', 11), bg='#0f3460', fg='#16c79a')
        subtitle.pack(side='left', padx=20)
        
        # LFW validation badge
        badge = tk.Label(title_frame, text="✅ LFW Validated: 95.20% Accuracy",
                        font=('Arial', 9, 'bold'), bg='#27ae60', fg='white',
                        padx=10, pady=5, relief='raised', bd=2)
        badge.pack(side='right', padx=20)
        
        # Control bar
        control_frame = tk.Frame(self.root, bg='#0f3460', height=60)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)
        
        # Buttons
        btn_container = tk.Frame(control_frame, bg='#0f3460')
        btn_container.pack(side='left', padx=20, pady=10)
        
        tk.Button(btn_container, text="👤 Load Target Person", 
                 command=self.load_reference,
                 font=('Arial', 11, 'bold'), bg='#16c79a', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="👥 Load Group Photo", 
                 command=self.load_test_image,
                 font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="🔍 Search All Faces", 
                 command=self.search_all_faces,
                 font=('Arial', 11, 'bold'), bg='#e74c3c', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        tk.Button(btn_container, text="🗑️ Clear All", 
                 command=self.clear_all,
                 font=('Arial', 11, 'bold'), bg='#95a5a6', fg='white',
                 padx=20, pady=10, cursor='hand2', relief='flat').pack(side='left', padx=5)
        
        # Threshold control
        threshold_frame = tk.Frame(control_frame, bg='#0f3460')
        threshold_frame.pack(side='right', padx=20, pady=10)
        
        tk.Label(threshold_frame, text="Distance Threshold:", 
                font=('Arial', 10, 'bold'), bg='#0f3460', fg='white').pack(side='left', padx=5)
        
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.80, to=1.50, resolution=0.05,
                                        orient='horizontal', length=200,
                                        command=self.update_threshold,
                                        font=('Arial', 9), bg='#0f3460', fg='white',
                                        highlightbackground='#0f3460', troughcolor='#16c79a',
                                        showvalue=False)
        self.threshold_slider.set(1.25)
        self.threshold_slider.pack(side='left', padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="1.25",
                                       font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a',
                                       width=5)
        self.threshold_label.pack(side='left', padx=5)
        
        # Main content area
        content = tk.Frame(self.root, bg='#16213e')
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Reference face
        self.reference_panel = self.create_reference_panel(content)
        self.reference_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Middle: Test image with all faces
        self.test_panel = self.create_test_panel(content)
        self.test_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right: Results and analysis
        self.results_panel = self.create_results_panel(content)
        self.results_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready • Load target person and group photo to begin search",
                                     font=('Arial', 10), bg='#0f3460', fg='white',
                                     anchor='w', padx=20, pady=8)
        self.status_label.pack(side='bottom', fill='x')
    
    def create_reference_panel(self, parent):
        """Create reference face panel"""
        panel = tk.Frame(parent, bg='#0f3460', width=400)
        panel.pack_propagate(False)
        
        tk.Label(panel, text="👤 Target Person", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')
        
        self.reference_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0, height=400)
        self.reference_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(self.reference_canvas, text="Click 'Load Target Person'\nto select reference image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d',
                justify='center').place(relx=0.5, rely=0.5, anchor='center')
        
        return panel

    def create_test_panel(self, parent):
        panel = tk.Frame(parent, bg='#0f3460', width=700)
        panel.pack_propagate(False)

        tk.Label(panel, text="👥 Group Photo", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')

        self.test_canvas = tk.Canvas(panel, bg='#1a1a2e', highlightthickness=0, height=600)
        self.test_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        tk.Label(self.test_canvas, text="Click 'Load Group Photo' to select an image",
                font=('Arial', 12), bg='#1a1a2e', fg='#7f8c8d', justify='center').place(relx=0.5, rely=0.5, anchor='center')

        return panel

    def create_results_panel(self, parent):
        panel = tk.Frame(parent, bg='#0f3460', width=420)
        panel.pack_propagate(False)

        tk.Label(panel, text="🔎 Results", font=('Arial', 13, 'bold'),
                bg='#0f3460', fg='white', pady=12).pack(fill='x')

        # Best match card
        self.best_card = tk.Frame(panel, bg='#0f3460', height=220)
        self.best_card.pack(fill='x', padx=12, pady=(6,6))
        self.best_card.pack_propagate(False)
        tk.Label(self.best_card, text="Best Match: ", font=('Arial', 12, 'bold'), bg='#0f3460', fg='#16c79a').pack(anchor='w', padx=8, pady=4)
        self.best_thumb = tk.Label(self.best_card, bg='#0f3460')
        self.best_thumb.pack(padx=8)
        self.best_text = tk.Label(self.best_card, text='No match', font=('Arial', 11), bg='#0f3460', fg='white')
        self.best_text.pack(pady=6)

        # All detected faces list
        list_frame = tk.Frame(panel, bg='#16213e')
        list_frame.pack(fill='both', expand=True, padx=12, pady=(6,6))
        tk.Label(list_frame, text='All Detected Faces', font=('Arial', 12, 'bold'), bg='#16213e', fg='#16c79a').pack(anchor='w', pady=(4,6))
        self.faces_listbox = tk.Listbox(list_frame, bg='#0f1b2d', fg='white', activestyle='none', selectbackground='#16c79a', font=('Arial', 11), height=6)
        self.faces_listbox.pack(fill='x', padx=4, pady=4)
        self.faces_listbox.bind('<<ListboxSelect>>', lambda e: self.on_face_select())

        # LFW context box
        ctx = tk.Frame(panel, bg='#0f3460')
        ctx.pack(fill='x', padx=12, pady=(6,12))
        tk.Label(ctx, text='LFW Benchmark Context', font=('Arial', 11, 'bold'), bg='#0f3460', fg='#16c79a').pack(anchor='w')
        ctx_text = "Same person avg: 1.017 ± 0.163\nDifferent person avg: 1.387 ± 0.069\nOptimal threshold: 1.25 (95.2% accuracy)"
        tk.Label(ctx, text=ctx_text, font=('Arial', 10), bg='#0f3460', fg='white', justify='left').pack(anchor='w', pady=6)

        # store references
        self.faces_list = []

        return panel

    def on_face_select(self):
        sel = self.faces_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.select_result(idx)

    def select_result(self, idx):
        """Highlight selected face and update best card."""
        if idx < 0 or idx >= len(self.test_faces):
            return
        r = self.test_faces[idx]
        bx, by, bw, bh = r['box']
        crop = cv2.imread(str(self.test_path))[int(by):int(by+bh), int(bx):int(bx+bw)]
        if crop is None or crop.size == 0:
            return
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_pil.thumbnail((120,120))
        thumb = ImageTk.PhotoImage(crop_pil)
        self.best_thumb.config(image=thumb)
        self.best_thumb.image = thumb
        self.best_text.config(text=f"Face #{idx+1} \nDistance: {r['distance']:.3f} | Threshold: {self.distance_threshold:.2f}")

    def update_results_ui(self):
        """Populate listbox and show best result."""
        self.faces_listbox.delete(0, tk.END)
        for i, r in enumerate(self.test_faces):
            self.faces_listbox.insert(tk.END, f"Face #{i+1}    Distance: {r['distance']:.3f}")
        if len(self.test_faces) > 0:
            # select best by default
            self.faces_listbox.select_set(0)
            self.select_result(0)

    def load_reference(self):
        path = filedialog.askopenfilename(title='Select reference image', filetypes=[('Images', '*.jpg *.jpeg *.png')])
        if not path:
            return
        self.reference_path = Path(path)
        # display — remove any placeholder widgets and canvas items, center the image
        pil = Image.open(path).convert('RGB')
        pil.thumbnail((360, 360))
        imgtk = ImageTk.PhotoImage(pil)
        self.photos['reference'] = imgtk
        # destroy any child widgets placed inside the canvas (placeholder labels)
        for w in self.reference_canvas.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        # clear canvas items
        try:
            self.reference_canvas.delete('all')
        except Exception:
            pass
        # center image based on current canvas size
        self.reference_canvas.update_idletasks()
        cw = self.reference_canvas.winfo_width()
        ch = self.reference_canvas.winfo_height()
        cx = cw // 2
        cy = ch // 2
        self.reference_canvas.create_image(cx, cy, image=imgtk, anchor='center')
        self.status_label.config(text=f'Loaded reference: {self.reference_path.name}')

    def load_test_image(self):
        path = filedialog.askopenfilename(title='Select group photo', filetypes=[('Images', '*.jpg *.jpeg *.png')])
        if not path:
            return
        self.test_path = Path(path)
        pil = Image.open(path).convert('RGB')
        pil.thumbnail((900, 900))
        imgtk = ImageTk.PhotoImage(pil)
        self.photos['test'] = imgtk
        # remove any placeholder widgets inside canvas
        for w in self.test_canvas.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        try:
            self.test_canvas.delete('all')
        except Exception:
            pass
        # center image using canvas size
        self.test_canvas.update_idletasks()
        tw = self.test_canvas.winfo_width()
        th = self.test_canvas.winfo_height()
        tx = tw // 2
        ty = th // 2
        self.test_canvas.create_image(tx, ty, image=imgtk, anchor='center')
        self.status_label.config(text=f'Loaded test image: {self.test_path.name}')

    def search_all_faces(self):
        if self.reference_path is None or self.test_path is None:
            messagebox.showwarning('Missing data', 'Please load both reference and group photo first.')
            return

        self.status_label.config(text='Running search...')
        self.root.update()

        # Load models
        detector = self.yunet
        embedder = self.arcface

        # --- Reference embedding ---
        ref_img = cv2.imread(str(self.reference_path))
        if ref_img is None:
            messagebox.showerror('Error', 'Failed to read reference image')
            return
        h, w = ref_img.shape[:2]
        try:
            detector.setInputSize((w, h))
        except Exception:
            pass
        det_out = detector.detect(ref_img)
        # detector.detect may return (retval, faces) or faces ndarray or list
        faces_arr = None
        if isinstance(det_out, tuple) or isinstance(det_out, list):
            if len(det_out) >= 2 and hasattr(det_out[1], 'shape'):
                faces_arr = det_out[1]
            else:
                # maybe list of dicts
                faces_arr = det_out
        else:
            faces_arr = det_out

        # parse first face
        ref_face = None
        if isinstance(faces_arr, (list,)) and len(faces_arr) > 0 and isinstance(faces_arr[0], dict):
            ref_face = faces_arr[0]
            # convert to bbox, landmarks
            box = ref_face.get('box')
            lm = ref_face.get('landmarks')
        elif hasattr(faces_arr, 'ndim') and faces_arr.ndim >= 2:
            # assume (N,15) Yunet style
            if faces_arr.shape[1] >= 15:
                ref_face = faces_arr[0]
                box = ref_face[0:4].astype(int)
                lm = ref_face[4:14].reshape(5,2)
            else:
                ref_face = None
        else:
            ref_face = None

        if ref_face is None:
            messagebox.showwarning('No face', 'No face detected in reference image')
            self.status_label.config(text='No face in reference')
            return

        # Ensure landmarks array
        try:
            if isinstance(lm, np.ndarray):
                landmarks = lm.reshape(5,2)
            else:
                landmarks = np.asarray(lm).reshape(5,2)
        except Exception:
            messagebox.showerror('Error', 'Invalid landmarks from detector')
            return

        # Align reference (use align_and_crop which returns BGR)
        aligned_ref = align_and_crop(ref_img, landmarks.flatten(), output_size=112)

        # Get embedding (support callable or .run)
        try:
            # Use canonical helper which handles sessions and preprocessing
            ref_emb = generate_arcface_embedding(aligned_ref)
        except Exception as e:
            messagebox.showerror('Embedder error', f'Failed to embed reference: {e}')
            return

        self.reference_embedding = ref_emb

        # --- Detect test image faces ---
        test_img = cv2.imread(str(self.test_path))
        if test_img is None:
            messagebox.showerror('Error', 'Failed to read test image')
            return
        th, tw = test_img.shape[:2]
        try:
            detector.setInputSize((tw, th))
        except Exception:
            pass
        det_out = detector.detect(test_img)
        faces_arr = None
        if isinstance(det_out, tuple) or isinstance(det_out, list):
            if len(det_out) >= 2 and hasattr(det_out[1], 'shape'):
                faces_arr = det_out[1]
            else:
                faces_arr = det_out
        else:
            faces_arr = det_out

        parsed_faces = []
        # normalize different detector outputs into dicts
        if isinstance(faces_arr, (list,)) and len(faces_arr) > 0 and isinstance(faces_arr[0], dict):
            for d in faces_arr:
                bx, by, bx2, by2 = d.get('box')
                score = d.get('score', 0)
                w = bx2 - bx + 1
                h = by2 - by + 1
                lm = d.get('landmarks')
                if lm is not None:
                    lm = np.asarray(lm).reshape(5,2)
                parsed_faces.append({'box': (int(bx), int(by), int(w), int(h)), 'score': float(score), 'landmarks': lm})
        elif hasattr(faces_arr, 'ndim') and faces_arr.ndim >= 2:
            for row in faces_arr:
                row = np.asarray(row)
                if row.size >= 15:
                    x, y, w0, h0 = row[0:4].astype(int)
                    lm = row[4:14].reshape(5,2).astype(int)
                    score = float(row[14])
                    parsed_faces.append({'box': (int(x), int(y), int(w0), int(h0)), 'score': score, 'landmarks': lm})
        else:
            messagebox.showwarning('No faces', 'No faces detected in test image')
            self.status_label.config(text='No faces in test image')
            return

        if len(parsed_faces) == 0:
            messagebox.showwarning('No faces', 'No faces detected in test image')
            self.status_label.config(text='No faces in test image')
            return

        # For each face compute embedding and distance
        results = []
        for pf in parsed_faces:
            bx, by, bw, bh = pf['box']
            lm = pf.get('landmarks')
            if lm is None:
                # fallback: create landmarks from bbox center
                lm = np.array([[bx + bw/4, by + bh/4],[bx + 3*bw/4, by + bh/4],[bx + bw/2, by + bh/2],[bx + bw/4, by + 3*bh/4],[bx + 3*bw/4, by + 3*bh/4]])
            aligned = align_and_crop(test_img, np.asarray(lm).flatten(), output_size=112)
            try:
                emb = generate_arcface_embedding(aligned)
            except Exception:
                emb = None
            if emb is None:
                continue
            dist = calculate_distance(ref_emb, emb)
            results.append({'box': (bx, by, bw, bh), 'score': pf.get('score', 0), 'landmarks': lm, 'embedding': emb, 'distance': dist})

        if len(results) == 0:
            messagebox.showwarning('No embeddings', 'Could not compute embeddings for detected faces')
            self.status_label.config(text='No embeddings')
            return

        # sort by distance (ascending)
        results.sort(key=lambda r: r['distance'])
        best = results[0]
        self.test_faces = results
        self.best_match_idx = 0

        # Prepare annotated image for display (PIL draw)
        pil = Image.open(str(self.test_path)).convert('RGB')
        draw = ImageDraw.Draw(pil)
        font = None
        try:
            font = ImageFont.truetype('arial.ttf', 20)
        except Exception:
            font = ImageFont.load_default()

        for i, r in enumerate(results):
            bx, by, bw, bh = r['box']
            x2 = bx + bw
            y2 = by + bh
            color = (0,255,0) if r['distance'] <= self.distance_threshold else (255,0,0)
            draw.rectangle([bx, by, x2, y2], outline=color, width=4)
            draw.text((bx, max(0, by-18)), f"{r['distance']:.3f}", fill=color, font=font)

        # highlight best
        bx, by, bw, bh = best['box']
        draw.rectangle([bx, by, bx+bw, by+bh], outline=(16,199,154), width=5)

        # display annotated image
        pil.thumbnail((900,900))
        imgtk = ImageTk.PhotoImage(pil)
        self.photos['test'] = imgtk
        self.test_canvas.delete('all')
        self.test_canvas.create_image(350, 300, image=imgtk, anchor='center')

        # populate results panel with thumbnails
        # update the richer results UI
        self.update_results_ui()

        self.status_label.config(text=f"Search finished • best distance={best['distance']:.3f}")

    def clear_all(self):
        self.reference_path = None
        self.reference_embedding = None
        self.test_path = None
        self.test_faces = []
        self.best_match_idx = None
        self.photos.clear()
        self.reference_canvas.delete('all')
        try:
            self.test_canvas.delete('all')
        except Exception:
            pass
        for widget in getattr(self, 'results_box', []):
            try:
                widget.destroy()
            except Exception:
                pass
        self.status_label.config(text='Cleared')

    def update_threshold(self, val):
        try:
            v = float(val)
            self.distance_threshold = v
            self.threshold_label.config(text=f'{v:.2f}')
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiFaceSearchGUI(root)
    root.mainloop()
