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
        self.tolerance = 0.7
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_skip = 2
        self.frame_count = 0
        self.scale_factor = 1.0  # Changed to 1.0 to avoid scaling issues


        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
        if os.path.isfile(model_path):
            self.face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.9, 0.3, 5000)
            print("YuNet model loaded successfully")
        else:
            print(f"YuNet model not found at {model_path}. Using Haar Cascade as fallback.")
            self.face_detector = None

        # Load Haar Cascade as fallback
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def load_known_faces(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as file:
                for line in file:
                    name, encoding_str, coordinates_str = line.strip().split(': ')
                    encoding = np.fromstring(encoding_str[1:-1], sep=',')
                    coordinates = tuple(map(int, coordinates_str[1:-1].split(',')))
                    self.known_faces_names.append(name)
                    self.known_faces_encodings.append(encoding)
                    self.known_faces_coordinates.append(coordinates)
            print(f"Loaded {len(self.known_faces_names)} known faces.")
        else:
            print("No known faces found. Starting fresh.")

    def draw_face_boxes(self, frame, faces, names=None):
        for i, face in enumerate(faces):
            if len(face) == 4:  # Haar Cascade output
                x, y, w, h = face
            else:  # YuNet output
                x, y, w, h = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if names:
                cv2.putText(frame, names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def save_to_file(self):
        with open(self.data_file, 'w') as file:
            for name, encoding, coordinates in zip(self.known_faces_names, self.known_faces_encodings, self.known_faces_coordinates):
                file.write(f"{name}: {encoding.tolist()}: {coordinates}\n")
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.face_detector:
            height, width, _ = frame.shape
            self.face_detector.setInputSize((width, height))
            _, faces = self.face_detector.detect(frame)
            if faces is not None:
                return faces, "YuNet"
        
        # Fallback to Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces, "Haar Cascade"

    def encode_face(self, frame, face):
        if len(face) == 4:  # Haar Cascade output
            x, y, w, h = face
        else:  # YuNet output
            x, y, w, h = map(int, face[:4])
        face_image = frame[y:y+h, x:x+w]
        face_encoding = cv2.resize(face_image, (128, 128)).flatten()
        return face_encoding / np.linalg.norm(face_encoding)

    async def process_frame(self, frame):
        faces, method = await asyncio.get_event_loop().run_in_executor(self.executor, self.detect_faces, frame)
        face_encodings = [await asyncio.get_event_loop().run_in_executor(self.executor, self.encode_face, frame, face) for face in faces]
        return faces, face_encodings, method

    async def save_new_face(self, websocket, name):
        print("Opening camera...")
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            capture_requested = False
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    await websocket.send(json.dumps({"error": "Failed to capture image."}))
                    continue

                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                faces, face_encodings, method = await self.process_frame(frame)

                frame_with_boxes = self.draw_face_boxes(frame.copy(), faces)
                _, jpeg_frame = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_frame_base64 = base64.b64encode(jpeg_frame).decode('utf-8')

                await websocket.send(json.dumps({
                    "image": jpeg_frame_base64,
                    "debug_info": f"Detected {len(faces)} faces using {method}. Frame shape: {frame.shape}"
                }))

                if capture_requested:
                    if len(faces) > 0:
                        if len(face_encodings) > 0:
                            self.known_faces_encodings.append(face_encodings[0])
                            self.known_faces_names.append(name)
                            self.known_faces_coordinates.append(tuple(map(int, faces[0][:4])))  # Save the coordinates
                            self.save_to_file()
                            await websocket.send(json.dumps({"match": True, "name": name}))
                            print(f"Face captured and saved as {name}.")
                            break
                        else:
                            await websocket.send(json.dumps({"error": "Face detected, but encoding failed. Please try again."}))
                    else:
                        await websocket.send(json.dumps({"error": "No face detected. Please try again."}))
                    capture_requested = False

                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(message)
                    if data.get('command') == 'capture':
                        capture_requested = True
                except asyncio.TimeoutError:
                    await asyncio.sleep(0)

        finally:
            video_capture.release()

    async def recognize_face_websocket(self, websocket, name):
        print("Opening camera for recognition...")
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    await websocket.send(json.dumps({"error": "Failed to capture image."}))
                    continue

                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                faces, face_encodings, method = await self.process_frame(frame)
                
                names = []
                for face_encoding in face_encodings:
                    distances = [np.linalg.norm(face_encoding - known_encoding) for known_encoding in self.known_faces_encodings]
                    if distances and min(distances) < self.tolerance:
                        best_match_index = np.argmin(distances)
                        name_to_display = self.known_faces_names[best_match_index]
                    else:
                        name_to_display = "Unknown"
                    
                    names.append(name_to_display)
                    
                    if name_to_display == name:
                        await websocket.send(json.dumps({"match": True, "name": name}))
                        return True
                    else:
                        await websocket.send(json.dumps({"match": False, "detected_name": name_to_display}))
                
                frame_with_boxes = self.draw_face_boxes(frame.copy(), faces, names)
                
                _, jpeg_frame = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_as_text = base64.b64encode(jpeg_frame).decode('utf-8')
                await websocket.send(json.dumps({
                    "image": jpg_as_text,
                    "debug_info": f"Detected {len(faces)} faces using {method}. Frame shape: {frame.shape}"
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
            print(command)
            if command == 'recognize':
                username = data.get('username')
                await face_recognition_system.recognize_face_websocket(websocket, username)
            elif command == 'new_face':
                username = data.get('username')
                await face_recognition_system.save_new_face(websocket, username)
            elif command == 'capture':
                # This case is now handled within the save_new_face method
                pass
            else:
                await websocket.send(json.dumps({"error": "Invalid command"}))
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")

async def main():
    server = await websockets.serve(websocket_handler, "localhost", 8766)
    print("WebSocket server started on ws://localhost:8766")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())