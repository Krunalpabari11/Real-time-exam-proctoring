import cv2
import numpy as np
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Could not find yunet.onnx at {model_path}")

print(f"OpenCV version: {cv2.__version__}")

try:
    # Adjusted parameters: lowered score threshold to 0.7 and increased NMS threshold to 0.5
    face_detector = cv2.FaceDetectorYN.create(
        model_path, "", (320, 320), 0.7, 0.5, 5000
    )
    print("YuNet model loaded successfully")
except Exception as e:
    print(f"Error loading YuNet model: {e}")
    raise

video_cap = cv2.VideoCapture(0)

initial_delay = 5
warning_delay = 3
max_warnings = 2
face_check_interval = 0

start_time = time.time()
last_check_time = start_time
warning_count = 0
last_warning_time = start_time - warning_delay

print("Initializing camera...")

def detect_faces(frame):
    try:
        face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = face_detector.detect(frame)

        if faces is not None:
            return [(int(face[0]), int(face[1]), int(face[2]), int(face[3])) for face in faces]
        else:
            return []
    except Exception as e:
        print(f"Error in detect_faces: {e}")    
        return []

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to capture frame. Retrying...")
        continue

    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time < initial_delay:
        cv2.putText(frame, f"Starting in: {int(initial_delay - elapsed_time)}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        if current_time - last_check_time >= face_check_interval:
            last_check_time = current_time
            
            faces = detect_faces(frame)

            if len(faces) == 0:
                if current_time - last_warning_time >= warning_delay:
                    warning_count += 1
                    last_warning_time = current_time
                    print(f"Warning {warning_count}: No face detected!")
                    cv2.putText(frame, f"Warning: No face detected!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(faces) > 1:
                if current_time - last_warning_time >= warning_delay:
                    warning_count += 1
                    last_warning_time = current_time
                    print(f"Warning {warning_count}: Multiple faces detected!")
                    cv2.putText(frame, f"Warning: Multiple faces detected!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                warning_count = 0
                cv2.putText(frame, "Face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if warning_count > max_warnings:
            print("Exam terminated: Too many warnings.")
            break

    cv2.imshow('Exam Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()