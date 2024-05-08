import cv2
import numpy as np
import math
from ultralytics import YOLO
import winsound
import streamlit as st

# Load YOLO model
model = YOLO("C:/Users/94801/OneDrive/Desktop/project 1/runs/detect/train4/weights/best.pt")

# Define function to print RGB values when mouse moves
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)
        
classNames = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"] 
person_cls = classNames.index("Person")
target_classes = ["NO-Hardhat", "NO-Safety Vest"]
target_cls_indices = [classNames.index(cls) for cls in target_classes]
a = [0,2,4,5,7]       

# Streamlit app layout
st.title('Custom Object Detection using Streamlit')
st.sidebar.title('Custom Object Detection')
use_webcam = st.sidebar.button('Use Webcam')
confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.3)
stop_button = st.sidebar.button('Stop Processing')

# Define the polygonal area of interest
area = [(407,212), (674,208), (967,434), (107,410)]

# Function to perform object detection and display the annotated frame
def perform_object_detection(frame):
    # Draw the polylines
    cv2.polylines(frame, [np.array(area, np.int32)], isClosed=True, color=(255, 0, 255), thickness=3)
    
    results = model.predict(frame)
    person_detected = False
    target_classes_detected = {"NO-Hardhat": False, "NO-Safety Vest": False}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            result = cv2.pointPolygonTest(np.array(area, np.int32), ((x1, y2)), False)

            if result >= 0:
                # Draw bounding box
                if cls in a:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    class_name = classNames[cls] if cls < len(classNames) else "Unknown"
                    label = f'{class_name} {conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                
                # Check if a person is detected
                if cls == person_cls:
                    person_detected = True
                # Check if target classes are detected
                elif cls in [2, 4]:
                    target_classes_detected["NO-Hardhat" if cls == 2 else "NO-Safety Vest"] = True

    # Produce beep sound if a person is detected without a hardhat or safety vest
    if person_detected and (target_classes_detected["NO-Hardhat"] or target_classes_detected["NO-Safety Vest"]):
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms

    return frame

# Streamlit callback to run object detection and display the video feed
def run_object_detection():
    cap = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture('new.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))
        frame = perform_object_detection(frame)
        st.image(frame, channels='BGR', use_column_width=True)
        if stop_button:
            break

# Run Streamlit app
if __name__ == '__main__':
    run_object_detection()
