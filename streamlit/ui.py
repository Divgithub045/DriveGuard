import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import math
from collections import deque
import time
from playsound import playsound
from ultralytics import YOLO
import threading
import pythoncom
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

def speak_warning():
    import pyttsx3
    pythoncom.CoInitialize()

    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say("Please do not use phone while driving")
    engine.runAndWait()
    engine.stop()

    from gtts import gTTS
    import os

    file_path = "hindi_alert.mp3"

    if not os.path.exists(file_path):
        tts = gTTS(
            text="कृपया गाड़ी चलाते समय फोन का उपयोग न करें",
            lang='hi'
        )
        tts.save(file_path)

    playsound(file_path)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
NOSE_TIP = 1
CHIN = 152

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.6
PITCH_THRESHOLD = 20

def euclidean(p1, p2):
    return dist.euclidean(p1, p2)

def calculate_EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(landmarks, w, h):
    upper = (int(landmarks[MOUTH[0]].x * w),
             int(landmarks[MOUTH[0]].y * h))
    lower = (int(landmarks[MOUTH[1]].x * w),
             int(landmarks[MOUTH[1]].y * h))
    left = (int(landmarks[MOUTH[2]].x * w),
            int(landmarks[MOUTH[2]].y * h))
    right = (int(landmarks[MOUTH[3]].x * w),
             int(landmarks[MOUTH[3]].y * h))
    return euclidean(upper, lower) / euclidean(left, right)

def calculate_head_pitch(landmarks, w, h):
    nose = (int(landmarks[NOSE_TIP].x * w),
            int(landmarks[NOSE_TIP].y * h))
    chin = (int(landmarks[CHIN].x * w),
            int(landmarks[CHIN].y * h))
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    return math.degrees(math.atan2(dy, dx))

def compute_score(EAR, MAR, pitch):
    score = 0
    if EAR < EAR_THRESHOLD:
        score += 2
    if MAR > MAR_THRESHOLD:
        score += 1
    if pitch > PITCH_THRESHOLD:
        score += 1
    
    if score >= 3:
        return score, "High Drowsiness Risk", "🔴"
    elif score == 2:
        return score, "Moderate Risk", "🟡"
    else:
        return score, "Normal", "🟢"

def process_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    metrics = {
        'EAR': None,
        'MAR': None,
        'pitch': None,
        'score': 0,
        'status': 'No Face Detected',
        'emoji': '⚪'
    }
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            left_eye = [(int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            
            EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
            MAR = calculate_MAR(face_landmarks.landmark, w, h)
            pitch = calculate_head_pitch(face_landmarks.landmark, w, h)
            
            score, status, emoji = compute_score(EAR, MAR, pitch)
            
            metrics = {
                'EAR': EAR,
                'MAR': MAR,
                'pitch': pitch,
                'score': score,
                'status': status,
                'emoji': emoji
            }
            
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    metrics['phone'] = False

    results_yolo = yolo_model(frame,verbose=False)[0]

    for box in results_yolo.boxes:
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        if label == "cell phone" and box.conf[0] > 0.5:
            metrics['phone'] = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, "PHONE", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame, metrics

st.title("👁️ Real-Time Drowsiness Detection System")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Controls")
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start",  width="stretch"):
            st.session_state.running = True
    with col2:
        if st.button("⏹️ Stop",  width="stretch"):
            st.session_state.running = False
    
    st.markdown("---")
    st.header("📊 Current Metrics")
    
    ear_text = st.empty()
    mar_text = st.empty()
    pitch_text = st.empty()
    score_text = st.empty()
    status_text = st.empty()
    
    st.markdown("---")
    st.header("📈 Score History")
    score_chart = st.empty()
    
    st.markdown("---")
    st.header("⚠️ Alerts")
    alert_area = st.empty()

video_placeholder = st.empty()
metrics_text = st.empty()
fps_text = st.empty()

if 'score_history' not in st.session_state:
    st.session_state.score_history = deque(maxlen=50)
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'last_voice' not in st.session_state:
    st.session_state.last_voice = 0

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.running = False
    else:
        prev_time = time.time()
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, metrics = process_frame(frame, face_mesh)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            video_placeholder.image(processed_frame, channels="BGR",  width="stretch")
            fps_text.text(f"FPS: {fps:.2f}")

            if metrics.get('phone', False):
                alert_area.error("📱 PHONE USAGE DETECTED!")

                if time.time() - st.session_state.last_voice > 5:
                    st.session_state.last_voice = time.time()
                    threading.Thread(target=speak_warning).start()

            if metrics['EAR'] is not None:
                ear_text.metric("👁️ Eye Aspect Ratio (EAR)", f"{metrics['EAR']:.3f}")
                mar_text.metric("👄 Mouth Aspect Ratio (MAR)", f"{metrics['MAR']:.3f}")
                pitch_text.metric("📐 Head Pitch", f"{metrics['pitch']:.1f}")
                score_text.metric("🎯 Drowsiness Score", f"{metrics['score']}/4")
                status_text.markdown(f"### {metrics['emoji']} {metrics['status']}")

                st.session_state.score_history.append(metrics['score'])
                score_chart.line_chart(list(st.session_state.score_history))

                if metrics['score'] >= 3:
                    alert_area.error(f"🚨 HIGH DROWSINESS DETECTED! (Alert #{st.session_state.alert_count})")
                elif metrics['score'] == 2:
                    alert_area.warning(f"⚠️ Moderate drowsiness detected")
                else:
                    alert_area.success("✅ Driver is alert")

            time.sleep(0.01)

        cap.release()
else:
    st.info("▶️ Click 'Start' in the sidebar to begin monitoring")