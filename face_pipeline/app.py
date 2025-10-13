import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image
import pandas as pd

# Put the workspace root on sys.path so this file can be executed directly by Streamlit
# (streamlit runs scripts rather than packages). This makes absolute imports work.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from face_pipeline.pipeline import FacePipeline, simple_liveness_check
except Exception:
    # fallback: add current package dir to sys.path and import module directly
    pkg_dir = os.path.dirname(__file__)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    from pipeline import FacePipeline, simple_liveness_check


st.set_page_config(page_title='Face Retrieval Debugger', layout='wide')
st.markdown('<style>body {background-color: #f8f9fb; color: #222;} .stButton>button {background-color: #2b8cff; color: white;} .thumb{border-radius:6px;border:1px solid #ddd;} .small-caption{color:#666;font-size:12px;} .tight{margin-top:-12px;} .spacer{height:8px;}</style>', unsafe_allow_html=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'Models')
YUNET = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar_int8.onnx')
ARCFACE = os.path.join(MODELS_DIR, 'w600k_r50.onnx')

if not os.path.exists(YUNET) or not os.path.exists(ARCFACE):
    st.warning('Place YuNet and ArcFace ONNX models in Models/ folder next to the workspace root')


@st.cache_resource
def create_pipeline():
    return FacePipeline(YUNET, ARCFACE)


pipeline = create_pipeline()


def draw_boxes_on_image(img_bgr, detections):
    img = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{d['score']:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return img


def compute_subject_means(db):
    # returns dict subject_id -> mean_embedding (numpy)
    subs = {}
    for sid, info in db.map.items():
        eids = info.get('eids', [])
        vecs = []
        for eid in eids:
            if hasattr(db, '_embeddings'):
                v = db._embeddings.get(eid)
                if v is not None:
                    vecs.append(v)
            elif getattr(db, 'index', None) is not None:
                try:
                    v = db.index.get_item_vector(eid)
                    if v is not None:
                        vecs.append(np.array(v, dtype=np.float32))
                except Exception:
                    pass
        if len(vecs) > 0:
            subs[sid] = np.mean(np.stack(vecs, axis=0), axis=0)
    return subs


st.title('Face Retrieval Debugger')

# Sidebar: global controls
st.sidebar.header('Settings')
det_conf = st.sidebar.slider('Detection threshold', 0.0, 1.0, 0.4, 0.01)
match_thresh = st.sidebar.slider('Match threshold (cosine)', 0.0, 1.0, 0.8, 0.01)
show_chart = st.sidebar.checkbox('Show embedding bar chart (64 dims)', value=False)
show_preview = st.sidebar.checkbox('Show detection preview', value=True)

# Session helpers for test images
if 'test_images' not in st.session_state:
    st.session_state['test_images'] = []  # list of dicts: {pil, bgr}

def add_test_images(pils):
    for pil in pils:
        img_np = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        st.session_state['test_images'].append({'pil': pil, 'bgr': img_np})


tab_preview, tab_details, tab_enroll = st.tabs(["Upload & Preview", "Face Details", "Enrollment & DB"])

with tab_preview:
    st.subheader('Upload test images or capture from webcam')
    mode = st.radio('Input source', ['Upload Images', 'Webcam (local)'], horizontal=True)
    if mode == 'Upload Images':
        test_files = st.file_uploader('Upload one or more images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if test_files:
            pils = []
            for f in test_files:
                try:
                    pils.append(Image.open(f).convert('RGB'))
                except Exception:
                    pass
            if len(pils) > 0 and st.button('Add to session'):
                add_test_images(pils)
                st.success(f'Added {len(pils)} image(s)')
    else:
        st.info('Webcam capture grabs a single frame from the default camera')
        if st.button('Capture from webcam'):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                add_test_images([pil])
                st.image(pil, caption='Captured', width='content')
            else:
                st.error('Failed to grab frame from webcam')

    if len(st.session_state['test_images']) > 0:
        st.markdown('#### Loaded images')
        cols = st.columns(4)
        for i, item in enumerate(st.session_state['test_images']):
            cols[i % 4].image(item['pil'], width='stretch')

        if show_preview:
            st.markdown('#### Detection preview (first image)')
            preview_np = st.session_state['test_images'][0]['bgr']
            detections = pipeline.detector.detect(preview_np, conf_threshold=det_conf)
            overlay = draw_boxes_on_image(preview_np, detections)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width='stretch')
    else:
        st.info('No test images added yet')

with tab_details:
    st.subheader('Face Details')
    if len(st.session_state['test_images']) == 0:
        st.write('No test images loaded')
    else:
        # Optional: batch scan across all images
        if st.button('Run batch scan on all test images'):
            batch_rows = []
            subs = compute_subject_means(pipeline.db)
            for idx, item in enumerate(st.session_state['test_images']):
                img_np = item['bgr']
                dets = pipeline.detector.detect(img_np, conf_threshold=det_conf)
                for i, d in enumerate(dets):
                    x1, y1, x2, y2 = d['box']
                    crop = img_np[y1:y2, x1:x2]
                    live = crop.size > 0 and simple_liveness_check(crop)
                    aligned = cv2.resize(crop, pipeline.embedder.input_size) if crop.size > 0 else None
                    best_sid = ''
                    best_score = np.nan
                    decision = ''
                    if live and aligned is not None:
                        emb = pipeline.embedder.embed(aligned)
                        if len(subs) > 0:
                            sims = {sid: float(np.dot(emb, vec) / (np.linalg.norm(vec) + 1e-8)) for sid, vec in subs.items()}
                            sorted_s = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                            best_sid, best_score = sorted_s[0]
                            decision = 'MATCH' if best_score >= match_thresh else 'NO MATCH'
                    batch_rows.append({
                        'image': idx + 1,
                        'face': i + 1,
                        'det_score': round(float(d.get('score', 0.0)), 3),
                        'box': f"({x1},{y1})-({x2},{y2})",
                        'live': bool(live),
                        'best_sid': best_sid,
                        'best_score': None if np.isnan(best_score) else round(float(best_score), 3),
                        'decision': decision
                    })
            if len(batch_rows) == 0:
                st.info('No faces detected in any image at the current detection threshold.')
            else:
                st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)

        for idx, item in enumerate(st.session_state['test_images']):
            pil = item['pil']
            img_np = item['bgr']
            with st.expander(f'Test Image {idx+1}', expanded=False):
                st.image(pil, width='stretch')
                dets = pipeline.detector.detect(img_np, conf_threshold=det_conf)
                if len(dets) == 0:
                    st.warning('No faces detected in this image')
                    continue
                for i, d in enumerate(dets):
                    x1, y1, x2, y2 = d['box']
                    crop = img_np[y1:y2, x1:x2]
                    if crop.size == 0:
                        st.write('Empty crop')
                        continue
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f'Face {i+1} crop', width='stretch')
                        live = simple_liveness_check(crop)
                        st.write('Liveness:', '✅' if live else '❌')
                    with c2:
                        aligned = cv2.resize(crop, pipeline.embedder.input_size)
                        st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), caption=f'Aligned {pipeline.embedder.input_size}', width='stretch')
                    with c3:
                        if live:
                            emb = pipeline.embedder.embed(aligned)
                            st.write('Embedding (first 8 dims):', np.round(emb[:8], 4).tolist())
                            if show_chart:
                                st.bar_chart(emb[:64])
                            subs = compute_subject_means(pipeline.db)
                            if len(subs) == 0:
                                st.info('No enrolled subjects to compare')
                            else:
                                sims = {sid: float(np.dot(emb, vec) / (np.linalg.norm(vec) + 1e-8)) for sid, vec in subs.items()}
                                sorted_s = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                                best_sid, best_score = sorted_s[0]
                                decision = 'MATCH' if best_score >= match_thresh else 'NO MATCH'
                                if decision == 'MATCH':
                                    st.success(f'{decision} — {best_sid} ({best_score:.3f})')
                                else:
                                    st.error(f'{decision} — top {best_sid} ({best_score:.3f})')
                                st.caption('Top 5: ' + ', '.join([f"{sid}:{score:.2f}" for sid, score in sorted_s[:5]]))

with tab_enroll:
    st.subheader('Enrollment & Database')
    enroll_images = st.file_uploader('Reference images to enroll (multiple allowed)', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key='enroll_files')
    enroll_sid = st.text_input('Subject ID for enrollment', key='enroll_sid')
    enroll_views = []
    if enroll_images:
        cols = st.columns(min(4, len(enroll_images)))
        for i, f in enumerate(enroll_images):
            try:
                pil = Image.open(f).convert('RGB')
            except Exception:
                continue
            img_np = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            dets = pipeline.detector.detect(img_np, conf_threshold=det_conf)
            if len(dets) == 0:
                cols[i % len(cols)].error('No face detected')
                cols[i % len(cols)].image(pil, caption='No face', width='stretch')
                continue
            d = max(dets, key=lambda x: x['score'])
            x1, y1, x2, y2 = d['box']
            crop = img_np[y1:y2, x1:x2]
            live = simple_liveness_check(crop)
            cols[i % len(cols)].image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f'View {i+1} — live: {live}', width='stretch')
            if crop.size > 0 and live:
                enroll_views.append(crop)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if enroll_sid and len(enroll_views) > 0 and st.button('Enroll all views'):
            pipeline.enroll(enroll_sid, enroll_views, metadata={'source': 'ui'}, consent=True)
            st.success(f'Enrolled {len(enroll_views)} view(s) for {enroll_sid}')
    with col_b:
        st.markdown('##### Enrolled subjects')
        st.write(list(pipeline.db.map.keys()))
        erase_sid = st.text_input('Subject ID to erase (GDPR)')
        cols = st.columns([1,1])
        with cols[0]:
            if st.button('Erase subject') and erase_sid:
                ok = pipeline.db.erase_subject(erase_sid)
                if ok:
                    st.success(f'Subject {erase_sid} erased')
                else:
                    st.error('Subject not found')
        with cols[1]:
            if st.button('Rebuild index (in-memory)'):
                pipeline.db.build()
                st.info('Index build requested')

st.markdown('---')
st.caption('This debug UI shows detection → liveness → alignment → embedding → similarity. Use the sidebar to tune thresholds and display options.')
