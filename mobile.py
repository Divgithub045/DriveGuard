import streamlit as st
import cv2
import av
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pyttsx3

# Load YOLO model
model = YOLO("yolov8n.pt")

# Init TTS
engine = pyttsx3.init()
engine.setProperty('rate', 160)

st.title("🚗 Driver Phone Detection System")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_alert_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)[0]
        phone_found = False

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone" and box.conf[0] > 0.5:
                phone_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img, "PHONE", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Alert (cooldown 5 sec)
        if phone_found and (time.time() - self.last_alert_time > 5):
            self.last_alert_time = time.time()
            engine.say("Please do not use phone while driving")
            engine.say("कृपया गाड़ी चलाते समय फोन का उपयोग न करें")
            engine.runAndWait()

        return img

# Start webcam
webrtc_streamer(
    key="driver-monitor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)