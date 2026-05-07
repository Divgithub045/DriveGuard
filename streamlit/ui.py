import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import math
from collections import deque
import time
import json
import datetime
import os
from playsound import playsound
from ultralytics import YOLO
import threading
#import pythoncom
import matplotlib.pyplot as plt
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
   # pythoncom.CoInitialize()

   # engine = pyttsx3.init()
   # engine.setProperty('rate', 160)
   # engine.say("Please do not use phone while driving")
   # engine.runAndWait()
   # engine.stop()

    from gtts import gTTS
    import os

    file_path = "hindi_alert.mp3"

    if not os.path.exists(file_path):
        tts = gTTS(
            text="वाहन चलाते समय किसी भी वस्तु, फोन इत्यादि का प्रयोग न करें।",
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

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate width and height
            width = x2 - x1
            height = y2 - y1
            # Check phone dimensions
            if 0 <= width <= 150 and 150 <= height <= 250:
                metrics['phone'] = True
                text = "Phone"
                color = (0, 255, 0)

            else:
                metrics['foreign_object'] = True
                text = "Foreign Object"
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{text} ({width}x{height}px)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    return frame, metrics

st.title("👁️ Real-Time Drowsiness Detection System")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Controls")
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None
    if 'prev_time' not in st.session_state:
        st.session_state.prev_time = None
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'log_path' not in st.session_state:
        st.session_state.log_path = None
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start",  width="stretch"):
            st.session_state.running = True
            st.session_state.logs = []
            st.session_state.frame_count = 0
            st.session_state.report = None
            st.session_state.pdf_bytes = None
            st.session_state.prev_time = None
            if st.session_state.cap is not None:
                st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.log_path = None
            st.session_state.started_at = datetime.datetime.now()
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
if 'started_at' not in st.session_state:
    st.session_state.started_at = None
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'prev_time' not in st.session_state:
    st.session_state.prev_time = None
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'log_path' not in st.session_state:
    st.session_state.log_path = None

LOG_EVERY_N_FRAMES = 5
CONTINUOUS_PHONE_RUN = 5
def generate_report(logs):
    total = len(logs)
    if total == 0:
        return None

    drowsy_frames = sum(1 for item in logs if item.get('score', 0) >= 2)
    alert_frames = sum(1 for item in logs if item.get('score', 0) < 2)
    phone_frames = sum(1 for item in logs if item.get('phone', False))

    continuous_phone_frames = 0
    run = 0
    for item in logs:
        if item.get('phone', False):
            run += 1
        else:
            if run >= CONTINUOUS_PHONE_RUN:
                continuous_phone_frames += run
            run = 0
    if run >= CONTINUOUS_PHONE_RUN:
        continuous_phone_frames += run

    drowsy_pct = (drowsy_frames / total) * 100
    alert_pct = (alert_frames / total) * 100
    phone_pct = (phone_frames / total) * 100
    continuous_phone_pct = (continuous_phone_frames / total) * 100
    safe_pct = max(0.0, 100.0 - (drowsy_pct * 0.6 + phone_pct * 0.3 + alert_pct * 0.1))

    return {
        'total_samples': total,
        'drowsy_pct': drowsy_pct,
        'alert_pct': alert_pct,
        'phone_pct': phone_pct,
        'continuous_phone_pct': continuous_phone_pct,
        'safety_score': safe_pct
    }

def build_pdf_bytes(lines):
    def escape_pdf(text):
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    line_height = 14
    content_lines = []
    for line in lines:
        content_lines.append(f"{escape_pdf(line)}")

    content = "BT /F1 12 Tf 50 760 Td "
    for i, line in enumerate(content_lines):
        if i > 0:
            content += f"0 -{line_height} Td "
        content += f"({line}) Tj "
    content += "ET"

    objects = []
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n")
    objects.append("4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(f"5 0 obj\n<< /Length {len(content)} >>\nstream\n{content}\nendstream\nendobj\n")

    xref_positions = []
    pdf = "%PDF-1.4\n"
    for obj in objects:
        xref_positions.append(len(pdf))
        pdf += obj
    xref_start = len(pdf)
    pdf += "xref\n0 6\n0000000000 65535 f \n"
    for pos in xref_positions:
        pdf += f"{pos:010d} 00000 n \n"
    pdf += "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
    pdf += f"{xref_start}\n%%EOF\n"

    return pdf.encode("ascii", errors="ignore")

def render_report():
    if not st.session_state.report:
        return

    st.markdown("---")
    st.subheader("Driver Performance Report")
    report = st.session_state.report

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Drowsiness %", f"{report['drowsy_pct']:.1f}%")
        st.metric("Phone Usage %", f"{report['phone_pct']:.1f}%")
    with col2:
        st.metric("Alert %", f"{report['alert_pct']:.1f}%")
        st.metric("Continuous Phone %", f"{report['continuous_phone_pct']:.1f}%")
    with col3:
        st.metric("Safety Score", f"{report['safety_score']:.1f}/100")

    chart_data = {
        "Drowsiness %": report['drowsy_pct'],
        "Phone Usage %": report['phone_pct'],
        "Alert %": report['alert_pct'],
        "Continuous Phone %": report['continuous_phone_pct']
    }
    st.bar_chart(chart_data)

    if st.session_state.log_path:
        st.caption(f"Session log path: {st.session_state.log_path}")

    if st.session_state.pdf_bytes:
        st.download_button(
            label="Download Full Session Log (PDF)",
            data=st.session_state.pdf_bytes,
            file_name="driver_session_log.pdf",
            mime="application/pdf"
        )
    labels = ["Drowsiness %", "Alert %"]
    values = [
        report["drowsy_pct"],
        report["alert_pct"],
    ]

    fig, ax = plt.subplots(figsize=(2.2, 2.2), facecolor="black")  # or "none" for transparent
    ax.set_facecolor("none")  # or "none"

    colors = ["#1E6FD9", "#63B3FF"]  # two blues
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "white", "fontsize": 4, "weight": "regular","rotation":90,"ha":"center","va":"center"}
    )

    ax.axis("equal")

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.pyplot(fig, transparent=True)  
def finalize_report():
    if st.session_state.report or not st.session_state.logs:
        return

    st.session_state.report = generate_report(st.session_state.logs)
    if st.session_state.report:
        report = st.session_state.report
        log_lines = [
            "Driver Monitoring Session Log",
            f"Started: {st.session_state.started_at.isoformat(timespec='seconds') if st.session_state.started_at else 'N/A'}",
            f"Samples: {report['total_samples']}",
            "",
            "Report Summary:",
            f"Drowsiness %: {report['drowsy_pct']:.1f}",
            f"Alert %: {report['alert_pct']:.1f}",
            f"Phone Usage %: {report['phone_pct']:.1f}",
            f"Continuous Phone %: {report['continuous_phone_pct']:.1f}",
            f"Safety Score: {report['safety_score']:.1f}",
            "",
            "Full Session Log:",
        ]

        for item in st.session_state.logs:
            log_lines.append(json.dumps(item, ensure_ascii=True))

        st.session_state.pdf_bytes = build_pdf_bytes(log_lines)

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

            st.session_state.frame_count += 1
            if st.session_state.frame_count % LOG_EVERY_N_FRAMES == 0:
                log_item = {
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'EAR': metrics.get('EAR'),
                    'MAR': metrics.get('MAR'),
                    'pitch': metrics.get('pitch'),
                    'score': metrics.get('score', 0),
                    'phone': metrics.get('phone', False)
                }
                st.session_state.logs.append(log_item)

            time.sleep(0.02)

        cap.release()
        st.session_state.report = generate_report(st.session_state.logs)
        if st.session_state.report:
            report = st.session_state.report
            log_lines = [
                "Driver Monitoring Session Log",
                f"Started: {st.session_state.started_at.isoformat(timespec='seconds') if st.session_state.started_at else 'N/A'}",
                f"Samples: {report['total_samples']}",
                "",
                "Report Summary:",
                f"Drowsiness %: {report['drowsy_pct']:.1f}",
                f"Alert %: {report['alert_pct']:.1f}",
                f"Phone Usage %: {report['phone_pct']:.1f}",
                f"Continuous Phone %: {report['continuous_phone_pct']:.1f}",
                f"Safety Score: {report['safety_score']:.1f}",
                "",
                "Full Session Log:",
            ]

            for item in st.session_state.logs:
                log_lines.append(json.dumps(item, ensure_ascii=True))

            os.makedirs("logs", exist_ok=True)
            session_name = st.session_state.session_id or "session"
            log_path = os.path.join("logs", f"{session_name}.json")
            with open(log_path, "w", encoding="utf-8") as log_file:
                json.dump(st.session_state.logs, log_file, ensure_ascii=True, indent=2)
            st.session_state.log_path = log_path

            st.session_state.pdf_bytes = build_pdf_bytes(log_lines)
            render_report()
else:
    st.info("▶️ Click 'Start' in the sidebar to begin monitoring")
    finalize_report()
    render_report()