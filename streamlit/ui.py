import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import math
from collections import deque
import time

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="👁️",
    layout="wide"
)

# Initialize MediaPipe Face Mesh
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
NOSE_TIP = 1
CHIN = 152

# Thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.6
PITCH_THRESHOLD = 20

# Helper Functions
def euclidean(p1, p2):
    return dist.euclidean(p1, p2)

def calculate_EAR(eye):
    """Calculate Eye Aspect Ratio"""
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(landmarks, w, h):
    """Calculate Mouth Aspect Ratio"""
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
    """Calculate head pitch angle"""
    nose = (int(landmarks[NOSE_TIP].x * w),
            int(landmarks[NOSE_TIP].y * h))
    chin = (int(landmarks[CHIN].x * w),
            int(landmarks[CHIN].y * h))
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    return math.degrees(math.atan2(dy, dx))

def compute_score(EAR, MAR, pitch):
    """Compute drowsiness score"""
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
    """Process a single frame for drowsiness detection"""
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
            
            # Extract eye landmarks
            left_eye = [(int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            
            # Calculate metrics
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
            
            # Draw eye landmarks (optional - you can remove this if you don't want any overlay)
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
    
    return frame, metrics

# Streamlit UI
st.title("👁️ Real-Time Drowsiness Detection System")
st.markdown("---")

# Sidebar for controls and metrics
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Start/Stop button
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.running = False
    
    st.markdown("---")
    st.header("📊 Current Metrics")
    
    # Metric placeholders
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

# Main video display
video_placeholder = st.empty()
metrics_text = st.empty()
fps_text = st.empty()

# Initialize session state for history
if 'score_history' not in st.session_state:
    st.session_state.score_history = deque(maxlen=50)
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0

# Video capture loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera connection.")
        st.session_state.running = False
    else:
        prev_time = time.time()
        
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to capture frame")
                break
            
            # Process frame
            processed_frame, metrics = process_frame(frame, face_mesh)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Display video
            
            # Display metrics below video
            if metrics['EAR'] is not None:
                metrics_text.markdown(f"""
                **{metrics['emoji']} Status: {metrics['status']}**  
                👁️ Eye Aspect Ratio (EAR): `{metrics['EAR']:.3f}` | 
                👄 Mouth Aspect Ratio (MAR): `{metrics['MAR']:.3f}` | 
                📐 Head Pitch: `{metrics['pitch']:.1f}°` | 
                🎯 Drowsiness Score: `{metrics['score']}/4`
                """)
            else:
                metrics_text.markdown("**⚪ No Face Detected** - Please ensure your face is visible to the camera")
            
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            fps_text.text(f"FPS: {fps:.2f}")
            
            # Update sidebar metrics
            if metrics['EAR'] is not None:
                ear_text.metric("👁️ Eye Aspect Ratio (EAR)", f"{metrics['EAR']:.3f}", 
                               delta="Normal" if metrics['EAR'] >= EAR_THRESHOLD else "Low")
                mar_text.metric("👄 Mouth Aspect Ratio (MAR)", f"{metrics['MAR']:.3f}",
                               delta="Normal" if metrics['MAR'] <= MAR_THRESHOLD else "High")
                pitch_text.metric("📐 Head Pitch", f"{metrics['pitch']:.1f}°",
                                 delta="Normal" if abs(metrics['pitch']) <= PITCH_THRESHOLD else "Tilted")
                score_text.metric("🎯 Drowsiness Score", f"{metrics['score']}/4")
                status_text.markdown(f"### {metrics['emoji']} {metrics['status']}")
                
                # Update history
                st.session_state.score_history.append(metrics['score'])
                
                # Update chart
                if len(st.session_state.score_history) > 0:
                    score_chart.line_chart(list(st.session_state.score_history))
                
                # Alert handling
                if metrics['score'] >= 3:
                    st.session_state.alert_count += 1
                    alert_area.error(f"🚨 HIGH DROWSINESS DETECTED! (Alert #{st.session_state.alert_count})")
                elif metrics['score'] == 2:
                    alert_area.warning(f"⚠️ Moderate drowsiness detected")
                else:
                    alert_area.success("✅ Driver is alert")
            else:
                status_text.markdown("### ⚪ No Face Detected")
                alert_area.info("ℹ️ Please ensure your face is visible to the camera")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
        
        cap.release()
else:
    st.info("▶️ Click 'Start' in the sidebar to begin monitoring")
    
    # Display instructions
    st.markdown("""
    ### 📖 Instructions:
    1. Click the **Start** button in the sidebar
    2. Allow camera access when prompted
    3. Position your face in front of the camera
    4. The system will monitor for drowsiness indicators:
       - **Eye Aspect Ratio (EAR)**: Detects eye closure
       - **Mouth Aspect Ratio (MAR)**: Detects yawning
       - **Head Pitch**: Detects head nodding
    
    ### 🎯 Scoring System:
    - **Score 0-1**: 🟢 Normal - Driver is alert
    - **Score 2**: 🟡 Moderate Risk - Warning signs detected
    - **Score 3-4**: 🔴 High Risk - Immediate attention needed
    
    ### ⚠️ Thresholds:
    - EAR below {:.2f} → Eyes closing
    - MAR above {:.2f} → Yawning detected  
    - Pitch above {:.0f}° → Head nodding
    """.format(EAR_THRESHOLD, MAR_THRESHOLD, PITCH_THRESHOLD))
