import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image

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
st.markdown('<style>body {background-color: #f8f9fb; color: #222;} .stButton>button {background-color: #2b8cff; color: white;}</style>', unsafe_allow_html=True)

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


st.title('Face Retrieval Debugger — Integrated View (Light)')

left, mid, right = st.columns([1.2, 1, 0.8])

with left:
    st.header('Input / Detection')
    mode = st.radio('Input', ['Upload Images', 'Webcam (local)'])
    # Support multiple test images (batch) to run queries against enrolled subjects
    test_files = []
    images = []
    if mode == 'Upload Images':
        test_files = st.file_uploader('Upload test images (multiple allowed)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if test_files:
            for f in test_files:
                try:
                    pil = Image.open(f).convert('RGB')
                except Exception:
                    continue
                img_np = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                images.append((pil, img_np))
            st.write(f'Loaded {len(images)} test image(s)')
            cols = st.columns(min(4, len(images)))
            for i, (pil, _) in enumerate(images):
                cols[i % len(cols)].image(pil, use_column_width=True)
    else:
        st.info('Webcam mode will capture frames when enabled.')
        if st.button('Capture from webcam'):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                images = [(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), frame)]
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Webcam capture', use_column_width=True)
            else:
                st.error('Failed to grab frame from webcam')
    if len(images) > 0:
        # show detections for the first test image as a quick preview
        preview_np = images[0][1]
        detections = pipeline.detector.detect(preview_np, conf_threshold=0.4)
        overlay = draw_boxes_on_image(preview_np, detections)
        st.subheader('Detections (preview of first image)')
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with mid:
    st.header('Face Details')
    st.write('Upload reference images (right) and test images (left) to populate this panel.')
    # Process each uploaded test image and show face details and similarity results
    if len(images) == 0:
        st.write('No test images loaded')
    else:
        for idx, (pil, img_np) in enumerate(images):
            st.markdown(f'## Test Image {idx+1}')
            dets = pipeline.detector.detect(img_np, conf_threshold=0.4)
            if len(dets) == 0:
                st.write('No faces detected in this image')
                continue
            for i, d in enumerate(dets):
                st.markdown(f'### Face {i+1}')
                x1, y1, x2, y2 = d['box']
                crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                st.write('Empty crop')
                continue
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption='Cropped face', width=200)
                live = simple_liveness_check(crop)
                st.write('Liveness:', '✅' if live else '❌')
                aligned = cv2.resize(crop, pipeline.embedder.input_size)
                st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), caption=f'Aligned {pipeline.embedder.input_size}', width=160)
                if live:
                    emb = pipeline.embedder.embed(aligned)
                    st.write('Embedding (first 8 dims):', np.round(emb[:8], 4).tolist())
                    chart = st.bar_chart(emb[:64])
                    # similarity against enrolled subjects
                    subs = compute_subject_means(pipeline.db)
                    if len(subs) == 0:
                        st.info('No enrolled subjects to compare')
                    else:
                        sims = {sid: float(np.dot(emb, vec) / (np.linalg.norm(vec) + 1e-8)) for sid, vec in subs.items()}
                        sorted_s = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                        st.write('Top matches:')
                        thresh = st.slider('Match threshold (cosine similarity)', 0.0, 1.0, 0.8, key=f'thresh_{idx}')
                        best_sid, best_score = sorted_s[0]
                        for sid, score in sorted_s[:5]:
                            st.write(f'Subject {sid}: similarity {score:.4f}')
                        decision = 'MATCH' if best_score >= thresh else 'NO MATCH'
                        if decision == 'MATCH':
                            st.success(f'{decision} — {best_sid} ({best_score:.3f})')
                        else:
                            st.error(f'{decision} — top {best_sid} ({best_score:.3f})')

with right:
    st.header('Enrollment & DB')
    # Support multiple reference images for multi-view enrollment
    enroll_images = st.file_uploader('Reference images to enroll (multiple allowed)', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key='enroll')
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
            dets = pipeline.detector.detect(img_np, conf_threshold=0.4)
            if len(dets) == 0:
                cols[i % len(cols)].error('No face detected')
                cols[i % len(cols)].image(pil, caption='No face', use_column_width=True)
                continue
            d = max(dets, key=lambda x: x['score'])
            x1, y1, x2, y2 = d['box']
            crop = img_np[y1:y2, x1:x2]
            live = simple_liveness_check(crop)
            cols[i % len(cols)].image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f'View {i+1} — live: {live}', use_column_width=True)
            if crop.size > 0 and live:
                enroll_views.append(crop)
    if enroll_sid and len(enroll_views) > 0 and st.button('Enroll all views'):
        pipeline.enroll(enroll_sid, enroll_views, metadata={'source': 'ui'}, consent=True)
        st.success(f'Enrolled {len(enroll_views)} view(s) for {enroll_sid}')

    st.markdown('---')
    st.write('Enrolled subjects:')
    st.write(list(pipeline.db.map.keys()))
    erase_sid = st.text_input('Subject ID to erase (GDPR)')
    if st.button('Erase subject') and erase_sid:
        ok = pipeline.db.erase_subject(erase_sid)
        if ok:
            st.success(f'Subject {erase_sid} erased')
        else:
            st.error('Subject not found')

    if st.button('Rebuild index (no-op for in-memory)'):
        pipeline.db.build()
        st.info('Index build requested')

st.markdown('---')
st.caption('This debug UI shows detection -> liveness -> alignment -> embedding -> similarity. For production replace the heuristic liveness with a trained model and use Annoy or a managed vector DB for large scale.')
