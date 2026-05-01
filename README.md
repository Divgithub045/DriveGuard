Summary:

* A real-time AI-based driver monitoring system that detects drowsiness and mobile phone usage using computer vision, providing alerts to improve road safety.

Features:

* Real-time webcam-based monitoring using OpenCV
* Drowsiness detection using facial metrics (EAR, MAR, head movement) via MediaPipe
* Mobile phone detection using YOLOv8
* Live display of driver metrics and status using Streamlit
* Audio and visual alerts for unsafe behavior
* Multilingual voice alerts (English and Hindi)

Plan:

* Analyze input video stream from webcam using OpenCV
* Use MediaPipe to extract facial landmarks and compute EAR, MAR, and head pose
* Calculate drowsiness score based on thresholds
* Integrate YOLOv8 model to detect mobile phone presence in frames
* Combine detection outputs to determine unsafe conditions
* Trigger alerts (visual + audio) when thresholds are exceeded
* Display real-time analytics dashboard using Streamlit

Tech Stack:

* Python
* OpenCV
* MediaPipe
* YOLOv8
* Streamlit
* Text-to-Speech (for alerts)
