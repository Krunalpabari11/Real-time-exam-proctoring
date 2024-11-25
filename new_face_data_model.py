import asyncio
import websockets
import json
import cv2
import numpy as np
import os
import base64
from concurrent.futures import ThreadPoolExecutor

class FaceRecognition:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.known_faces_coordinates = []
        self.data_file = 'face_data.txt'
        self.tolerance = 0.25 # Adjusted tolerance for better accuracy
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_skip = 0
        self.frame_count = 0
        
        # Load face detection model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
        if os.path.isfile(model_path):
            self.face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.9, 0.3, 5000)
            print("YuNet model loaded successfully")
        else:
            print(f"YuNet model not found at {model_path}. Using Haar Cascade.")
            self.face_detector = None

        # Load Haar Cascade as fallback
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def load_known_faces(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as file:
                    self.known_faces_encodings = []
                    self.known_faces_names = []
                    self.known_faces_coordinates = []
                    
                    for line in file:
                        try:
                            parts = line.strip().split('|')
                            if len(parts) == 3:
                                name = parts[0]
                                encoding = np.fromstring(parts[1], sep=',')
                                coords = np.fromstring(parts[2], sep=',')
                                if len(encoding) > 0 and len(coords) > 0:
                                    self.known_faces_names.append(name)
                                    self.known_faces_encodings.append(encoding)
                                    self.known_faces_coordinates.append(coords)
                            else:
                                print(f"Skipping invalid line: {line}")
                        except Exception as e:
                            print(f"Error processing line: {e}")
                            continue
                print(f"Successfully loaded {len(self.known_faces_names)} faces")
            except Exception as e:
                print(f"Error loading faces: {e}")
                open(self.data_file, 'w').close()
        else:
            print("No face data file found. Creating new one.")
            open(self.data_file, 'w').close()

    def save_to_file(self):
        try:
            with open(self.data_file, 'w') as file:
                for name, encoding, coords in zip(self.known_faces_names, 
                                                   self.known_faces_encodings, 
                                                   self.known_faces_coordinates):
                    encoding_str = ','.join(map(str, encoding))
                    coords_str = ','.join(map(str, coords))
                    file.write(f"{name}|{encoding_str}|{coords_str}\n")
            print(f"Successfully saved {len(self.known_faces_names)} faces to {self.data_file}")
        except Exception as e:
            print(f"Error saving faces: {e}")

    def detect_faces(self, frame):
        if self.face_detector:
            height, width, _ = frame.shape
            self.face_detector.setInputSize((width, height))
            _, faces = self.face_detector.detect(frame)
            if faces is not None:
                return faces, "YuNet"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces, "Haar Cascade"

    def encode_face(self, frame, face):
        if len(face) == 4:
            x, y, w, h = face
        else:
            x, y, w, h = map(int, face[:4])

        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (64, 64))
        face_encoding = resized_face.flatten()
        face_encoding = face_encoding / np.linalg.norm(face_encoding)
        return face_encoding

    def draw_face_boxes(self, frame, faces, names=None):
        for i, face in enumerate(faces):
            if len(face) == 4:
                x, y, w, h = face
            else:
                x, y, w, h = map(int, face[:4])
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if names and i < len(names):
                name = names[i]
                cv2.putText(frame, name, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    async def process_frame(self, frame):
        faces, method = await asyncio.get_event_loop().run_in_executor(self.executor, self.detect_faces, frame)
        
        face_encodings = []
        for face in faces:
            encoding = await asyncio.get_event_loop().run_in_executor(self.executor, self.encode_face, frame, face)
            face_encodings.append(encoding)
            
        return faces, face_encodings, method

    async def recognize_face_websocket(self, websocket, name):
        print(f"Starting face recognition for {name}...")
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            consecutive_matches = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    await websocket.send(json.dumps({"error": "Failed to capture image"}))
                    continue

                faces, face_encodings, method = await self.process_frame(frame)
                
                names = []
                for face_encoding in face_encodings:
                    best_match_name = "Unknown"
                    min_distance = float('inf')
                    
                    for idx, known_encoding in enumerate(self.known_faces_encodings):
                        distance = np.linalg.norm(face_encoding - known_encoding)
                        if distance < min_distance:
                            min_distance = distance
                            if distance < self.tolerance:
                                best_match_name = self.known_faces_names[idx]
                    
                    names.append(best_match_name)
                    
                    if best_match_name == name:
                        consecutive_matches += 1
                        if consecutive_matches >= 10:  # Ensure multiple frames match
                            await websocket.send(json.dumps({
                                "match": True,
                                "name": name,
                                "message": "Face recognized successfully"
                            }))
                            return True
                    else:
                        consecutive_matches = 0

                frame_with_boxes = self.draw_face_boxes(frame.copy(), faces, names)
                _, jpeg_frame = cv2.imencode('.jpg', frame_with_boxes, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_base64 = base64.b64encode(jpeg_frame).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "image": jpeg_base64,
                    "debug_info": f"Detected {len(faces)} faces using {method}",
                    "detected_names": names
                }))

                await asyncio.sleep(0.01)

        finally:
            video_capture.release()

async def websocket_handler(websocket, path):
    face_recognition_system = FaceRecognition()
    face_recognition_system.load_known_faces()

    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get('command')
            print(f"Received command: {command}")
            
            if command == 'recognize':
                username = data.get('username')
                await face_recognition_system.recognize_face_websocket(websocket, username)
            elif command == 'new_face':
                username = data.get('username')
                await face_recognition_system.save_new_face(websocket, username)
            else:
                await websocket.send(json.dumps({"error": "Invalid command"}))
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")

async def main():
    server = await websockets.serve(websocket_handler, "localhost", 8767)
    print("WebSocket server started on ws://localhost:8767")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
