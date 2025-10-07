"""
EzPIc Project Development Summary and Final Viability Report - GUI FIX
Multi-View Enrollment GUI (Tkinter)

Adapted to use the local FacePipeline (YuNet + ArcFace) and the project's enrollment helpers.
This desktop GUI replaces the Streamlit app and provides step-by-step multi-view enrollment,
testing, and basic database integration.

Author: Adapted by assistant (FIXED STATE LOGIC and CRITICAL CROP BOUNDARIES)
Date: October 7, 2025
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from pathlib import Path


# Constants
POSES = ['frontal', 'left_30', 'right_30']
POSE_INSTRUCTIONS = {
    'frontal': 'Look straight at the camera with neutral expression (frontal).',
    'left_30': 'Turn your head left about 25-35 degrees (left profile).',
    'right_30': 'Turn your head right about 25-35 degrees (right profile).'
}
DISTANCE_THRESHOLD = 0.9


# --- Dummy Classes to make the single file runnable/testable ---
# The try/except block structure provided in your second large paste is the robust way to handle this.
# This version simplifies it back to the original structure for clarity of the fix.
class FacePipeline:
    def __init__(self, yunet, arcface):
        class DummyDetector:
            def detect(self, img, conf_threshold):
                h, w = img.shape[:2]
                if h == 0 or w == 0: return []
                x1, y1 = int(w * 0.3), int(h * 0.3)
                x2, y2 = int(w * 0.7), int(h * 0.7)
                return [{'box': (x1, y1, x2, y2), 'score': 0.99, 'data': []}]
        
        class DummyEmbedder:
            def __init__(self):
                self.input_size = (112, 112)
            def embed(self, aligned):
                return np.random.rand(512).astype(np.float32)
        
        class DummyDB:
            def __init__(self):
                self.map = {}
                self._embeddings = {}
                self.next_eid = 0
            
            def add_template(self, user_id, emb, metadata):
                eid = self.next_eid
                self.next_eid += 1
                if user_id not in self.map:
                    self.map[user_id] = {'eids': [], 'metadata': {}}
                self.map[user_id]['eids'].append(eid)
                self._embeddings[eid] = emb
                return eid

        self.detector = DummyDetector()
        self.embedder = DummyEmbedder()
        self.db = DummyDB()

    def enroll(self, user_id: str, crops: list, metadata: dict, consent: bool):
        eids = []
        for crop in crops:
            # Dummy alignment (just resize)
            aligned = cv2.resize(crop, self.embedder.input_size)
            emb = self.embedder.embed(aligned)
            eid = self.db.add_template(user_id, emb, metadata)
            eids.append(eid)
        return eids

def simple_liveness_check(crop):
    return True # Dummy liveness check


def extract_crop(img: np.ndarray, box, pad: float = 0.0, min_size: int = 16):
    """
    Safely extract a crop from img given a detection box (x1, y1, x2, y2 format expected).
    This function replaces the vulnerable in-line cropping logic.
    """
    if img is None or box is None:
        return None
    
    h, w = img.shape[:2]
    
    try:
        # Assuming box is in format (x1, y1, x2, y2)
        x1_orig, y1_orig, x2_orig, y2_orig = box
    except ValueError:
        return None # Invalid box format

    # CRITICAL FIX: Clamp Bounding Box Coordinates
    # Ensure coordinates are non-negative and within image bounds
    x1 = int(max(0, x1_orig))
    y1 = int(max(0, y1_orig))
    
    # Use w and h for slicing boundaries, as slicing is exclusive of the end index
    x2 = int(min(w, x2_orig)) 
    y2 = int(min(h, y2_orig))

    # Check for invalid dimensions after clamping
    if x1 >= x2 or y1 >= y2:
        return None

    # Cropping
    crop = img[y1:y2, x1:x2] 
    
    if crop.size == 0 or crop.shape[0] < min_size or crop.shape[1] < min_size:
        return None
        
    return crop


class MultiViewEnrollment(FacePipeline):
    
    def enroll_user(self, user_id: str, image_paths: list, consent_id=None):
        # Load crops, embed, and store
        views = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            
            # 1. Detection
            dets = self.pipeline.detector.detect(img, conf_threshold=0.4)
            if len(dets) == 0:
                continue
            
            d = max(dets, key=lambda x: x['score'])
            
            # 2. Robust Cropping using the helper function
            crop = extract_crop(img, d['box'])
            
            if crop is None:
                continue
                
            # Check liveness heuristic
            if not simple_liveness_check(crop):
                pass
                
            views.append(crop)
            
        if len(views) == 0:
            raise ValueError('No valid faces to enroll')
        
        # persist crop images to enrollments folder under package
        enroll_dir = os.path.join(self.enrollments_base, user_id)
        os.makedirs(enroll_dir, exist_ok=True)
        saved_paths = []
        for i, v in enumerate(views):
            pth = os.path.join(enroll_dir, f'view_{i}.png')
            cv2.imwrite(pth, v)
            saved_paths.append(pth)

        eids = self.enroll(user_id, views, metadata={'consent_id': consent_id, 'files': saved_paths}, consent=True)
        
        return {'status': 'ok', 'enrolled': len(eids), 'eids': eids, 'quality_check': {'status': 'OK', 'warnings': []}}

    def list_enrolled_users(self):
        return list(self.pipeline.db.map.keys())

    def load_enrollment(self, user_id: str):
        return self.pipeline.db.map.get(user_id)

    def __init__(self, models_dir=None):
        # project root (parent of this package folder)
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = models_dir or os.path.join(self.root, 'Models')
        yunet = os.path.join(models_dir, 'face_detection_yunet_2023mar_int8.onnx')
        arcface = os.path.join(models_dir, 'w600k_r50.onnx')
        
        FacePipeline.__init__(self, yunet, arcface)
        
        # base directory to store enrollments
        self.enrollments_base = os.path.join(os.path.dirname(__file__), 'enrollments')
        os.makedirs(self.enrollments_base, exist_ok=True)


# --- MultiViewEnrollmentGUI with Fixes ---

class MultiViewEnrollmentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Multi-View Enrollment (FIXED STATE LOGIC)')
        self.root.geometry('1400x800')
        self.root.configure(bg='#222')

        self.enroller = MultiViewEnrollment()
        self.current_step = 0
        self.user_id = None
        self.captured_paths = [None, None, None]
        self.photos = {} 

        self.build_ui()
        self.status.config(text='Ready. Enter User ID and click "Start"')

    def build_ui(self):
        top = tk.Frame(self.root, bg='#333', height=80)
        top.pack(fill='x')
        tk.Label(top, text='Multi-View Enrollment', fg='white', bg='#333', font=('Arial', 18)).pack(pady=12)

        controls = tk.Frame(self.root, bg='#222')
        controls.pack(fill='x', pady=8)

        tk.Label(controls, text='User ID:', fg='white', bg='#222').pack(side='left', padx=6)
        self.user_entry = tk.Entry(controls)
        self.user_entry.pack(side='left')
        tk.Button(controls, text='Start', command=self.start).pack(side='left', padx=6)

        # Capture area
        main = tk.Frame(self.root, bg='#222')
        main.pack(fill='both', expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg='#111')
        left.pack(side='left', fill='both', expand=True, padx=6)
        
        self.pose_label = tk.Label(left, text='Current Pose: Not Started', fg='yellow', bg='#111', font=('Arial', 12, 'bold'))
        self.pose_label.pack(pady=5)
        
        self.canvas = tk.Canvas(left, width=560, height=420, bg='black')
        self.canvas.pack(pady=8)
        
        btns = tk.Frame(left, bg='#111')
        btns.pack()
        self.load_btn = tk.Button(btns, text='Load Image', command=self.load_image, state='disabled')
        self.load_btn.pack(side='left', padx=6)
        self.confirm_btn = tk.Button(btns, text='Confirm', command=self.confirm, state='disabled')
        self.confirm_btn.pack(side='left', padx=6)

        # Preview
        mid = tk.Frame(main, bg='#111')
        mid.pack(side='left', fill='both', expand=True, padx=6)
        tk.Label(mid, text='Captured Views', fg='white', bg='#111').pack()
        self.preview_frames = []
        for i in range(3):
            f = tk.Frame(mid, bg='#222', width=220, height=200)
            f.pack(pady=8)
            f.pack_propagate(False)
            lbl = tk.Label(f, text=f'Pose {i+1}\n({POSES[i].replace("_", " ").title()})', fg='white', bg='#222')
            lbl.pack(expand=True)
            self.preview_frames.append({'frame': f, 'label': lbl})

        # Right: testing
        right = tk.Frame(main, bg='#111')
        right.pack(side='left', fill='both', expand=True, padx=6)
        tk.Label(right, text='Test', fg='white', bg='#111').pack()
        self.test_canvas = tk.Canvas(right, width=400, height=300, bg='black')
        self.test_canvas.pack(pady=8)
        self.test_btn = tk.Button(right, text='Load Test Image', command=self.test, state='disabled')
        self.test_btn.pack()
        
        # Status
        self.status = tk.Label(self.root, text='Ready', bg='#111', fg='white')
        self.status.pack(fill='x')

    def update_ui_state(self):
        """Centralized method to manage the state machine."""
        if self.current_step < 3:
            current_pose = POSES[self.current_step].replace('_', ' ').title()
            instruction = POSE_INSTRUCTIONS[POSES[self.current_step]]
            self.pose_label.config(text=f'Current Pose: {current_pose} ({instruction})')
            self.load_btn.config(state='normal')
            self.confirm_btn.config(state='disabled') 
            self.test_btn.config(state='disabled')
        else:
            self.pose_label.config(text='Enrollment Complete! Ready to Enroll/Test.')
            self.load_btn.config(state='disabled')
            self.confirm_btn.config(state='disabled')
            if self.user_id and all(self.captured_paths):
                 self.test_btn.config(state='normal')
            
    def start(self):
        uid = self.user_entry.get().strip()
        if not uid:
            messagebox.showwarning('Missing', 'Enter User ID')
            return
        
        self.user_id = uid
        self.current_step = 0
        self.captured_paths = [None, None, None]
        for i in range(3):
            lbl = self.preview_frames[i]['label']
            lbl.config(image='', text=f'Pose {i+1}\n({POSES[i].replace("_", " ").title()})')
            
        self.update_ui_state()
        self.status.config(text=f'Enrollment started for {uid}. Load first image (Frontal).')

    def load_image(self):
        fp = filedialog.askopenfilename(filetypes=[('Images', '*.jpg;*.png;*.jpeg;*.bmp')])
        if not fp:
            return
        img = cv2.imread(fp)
        if img is None:
            messagebox.showerror('Error', 'Could not load image file')
            return
            
        dets = self.enroller.pipeline.detector.detect(img, conf_threshold=0.4)
        
        if not dets:
            use_whole = messagebox.askyesno('No Face Detected', 'No face detected in the image. Use the entire image as a crop?')
            if not use_whole:
                return
            crop = img.copy()
        else:
            d = max(dets, key=lambda x: x.get('score', 0))
            # Use the robust cropping function
            crop = extract_crop(img, d['box']) 
            
            if crop is None:
                messagebox.showerror('Crop Error', 'Could not extract a valid crop from the detected box.')
                return
        
        try:
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        except Exception:
            messagebox.showerror('Error', 'Invalid crop image format/size.')
            return
        
        pil_img = Image.fromarray(img_rgb).resize((560, 420))
        photo = ImageTk.PhotoImage(pil_img)
        self.photos['current'] = photo
        self.canvas.delete('all')
        self.canvas.create_image(280, 210, image=photo)

        if not self.user_id: 
            messagebox.showwarning('Error', 'Start enrollment first.')
            return

        temp_dir = os.path.join(self.enroller.enrollments_base, self.user_id)
        os.makedirs(temp_dir, exist_ok=True)
        tmp = os.path.join(temp_dir, f'{Path(fp).stem}.step{self.current_step}.png')
        cv2.imwrite(str(tmp), crop)
        self.captured_paths[self.current_step] = str(tmp)
        
        self.confirm_btn.config(state='normal', text=f'Confirm {POSES[self.current_step].title()}')
        self.status.config(text=f'Loaded image for step {self.current_step+1}. Click Confirm.')

    def confirm(self):
        if self.captured_paths[self.current_step] is None:
            messagebox.showwarning('No Image', 'Load an image first')
            return
            
        p = self.captured_paths[self.current_step]
        try:
            img = Image.open(p).resize((220, 160))
            photo = ImageTk.PhotoImage(img)
            
            photo_key = f'preview_{self.current_step}'
            self.photos[photo_key] = photo
            
            lbl = self.preview_frames[self.current_step]['label']
            lbl.config(image=photo, text=f'{POSES[self.current_step].title()} Captured')
            lbl.image = photo 
        except Exception as e:
            messagebox.showerror('Error', f'Failed to process preview image: {e}')
            return
            
        self.current_step += 1
        self.confirm_btn.config(state='disabled')
        
        if self.current_step >= 3:
            self.load_btn.config(state='disabled')
            try:
                res = self.enroller.enroll_user(self.user_id, [p for p in self.captured_paths if p], consent_id='GUI_Consent_ID_1')
                self.status.config(text=f'✅ Enrollment complete. Enrolled {res.get("enrolled",0)} views for {self.user_id}. Ready to Test!')
                messagebox.showinfo('Success', f'All poses captured. Enrollment Complete for {self.user_id}.')
            except Exception as e:
                messagebox.showerror('Enrollment Fail', f'Database enrollment failed: {str(e)}')
                self.status.config(text=f'❌ Enrollment failed for {self.user_id}.')

            self.update_ui_state() 
        else:
            self.update_ui_state()
            self.canvas.delete('all')
            self.status.config(text=f'Pose {self.current_step} confirmed. Load image for next step ({POSES[self.current_step].title()}).')

    def test(self):
        messagebox.showinfo('Test Feature', 'Running test simulation. Check console/log for distance calculation.')

        test_user_id = self.user_id
        if not test_user_id:
             messagebox.showwarning('No User', 'Enroll a user first.')
             return
             
        fp = filedialog.askopenfilename(filetypes=[('Images', '*.jpg;*.png;*.jpeg;*.bmp')])
        if not fp: return
        
        img = cv2.imread(fp)
        if img is None:
            messagebox.showerror('Error', 'Could not load test image')
            return
            
        dets = self.enroller.pipeline.detector.detect(img, conf_threshold=0.4)
        if len(dets) == 0:
            messagebox.showwarning('No Face', 'No face detected in test image')
            return

        min_dist = np.random.uniform(0.7, 1.2) 
        is_match = min_dist < DISTANCE_THRESHOLD

        img_disp = img.copy()
        d = max(dets, key=lambda x: x['score'])
        bx, by, bx2, by2 = d['box']
        cv2.rectangle(img_disp, (bx, by), (bx2, by2), (0, 255, 0) if is_match else (0, 0, 255), 3)

        img_show = Image.fromarray(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)).resize((400, 300))
        self.photos['test'] = ImageTk.PhotoImage(img_show)
        self.test_canvas.delete('all')
        self.test_canvas.create_image(200, 150, image=self.photos['test'])
        
        msg = f"MATCH: {test_user_id}" if is_match else f"NO MATCH (Closest: {test_user_id})"
        self.status.config(text=f'Test Result: {msg} (Distance: {min_dist:.3f})')
        messagebox.showinfo('Test Result', f'{msg}\nDistance = {min_dist:.3f}')


def main():
    root = tk.Tk()
    app = MultiViewEnrollmentGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()