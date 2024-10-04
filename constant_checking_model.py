import cv2
import numpy as np
import time
import os
import asyncio
import websockets
import json
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Could not find yunet.onnx at {model_path}")

print(f"OpenCV version: {cv2.__version__}")

try:
    face_detector = cv2.FaceDetectorYN.create(
        model_path, "", (320, 320), 0.7, 0.5, 5000
    )
    print("YuNet model loaded successfully")
except Exception as e:
    print(f"Error loading YuNet model: {e}")
    raise

initial_delay = 0
warning_delay = 3
max_warnings = 2
face_check_interval = 0

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

async def send_face_data(websocket, path):
    video_cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    last_check_time = start_time
    warning_count = 0
    last_warning_time = start_time - warning_delay

    print("Initializing camera...")

    try:
        while True:
            try:
                ret, frame = video_cap.read()
                if not ret:
                    print("Failed to capture frame. Retrying...")
                    await asyncio.sleep(1)
                    continue

                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time < initial_delay:
                    status = f"Starting in: {int(initial_delay - elapsed_time)}s"
                else:
                    if current_time - last_check_time >= face_check_interval:
                        last_check_time = current_time
                        
                        faces = detect_faces(frame)

                        if len(faces) == 0:
                            if current_time - last_warning_time >= warning_delay:
                                warning_count += 1
                                last_warning_time = current_time
                                status = f"Warning {warning_count}: No face detected!"
                        elif len(faces) > 1:
                            if current_time - last_warning_time >= warning_delay:
                                warning_count += 1
                                last_warning_time = current_time
                                status = f"Warning {warning_count}: Multiple faces detected!"
                        else:

                            status = "Face detected"

                        # Draw rectangles around detected faces
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # Convert frame to JPEG
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        # Convert to base64 encoding
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                        # Prepare data to send via WebSocket
                        data = {
                            "image": jpg_as_text,
                            "faces": faces,
                            "status": status,
                            "warningCount": warning_count
                        }

                        # Send data via WebSocket
                        await websocket.send(json.dumps(data))

                    if warning_count > max_warnings:
                        print("Exam terminated: Too many warnings.")
                        break

                await asyncio.sleep(0)  

            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed. Waiting for new connection...")
                break

    finally:
        video_cap.release()

async def main():
    while True:
        try:
            server = await websockets.serve(send_face_data, "localhost", 8765)
            print("WebSocket server started on ws://localhost:8765")
            await server.wait_closed()
        except Exception as e:
            print(f"Error in WebSocket server: {e}")
            print("Restarting server in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
