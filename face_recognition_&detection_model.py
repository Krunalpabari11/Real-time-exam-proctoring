import cv2
import face_recognition
import numpy as np
import os

class FaceRecognition:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.data_file = 'face_data.txt'
        self.tolerance = 0.6  

    def load_known_faces(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as file:
                for line in file:
                    name, encoding_str = line.strip().split(': ')
                    encoding = np.fromstring(encoding_str[1:-1], sep=',')
                    self.known_faces_names.append(name)
                    self.known_faces_encodings.append(encoding)
            print(f"Loaded {len(self.known_faces_names)} known faces.")
        else:
            print("No known faces found. Starting fresh.")

    def draw_face_boxes(self, frame, face_locations, names=None):
        for i, (top, right, bottom, left) in enumerate(face_locations):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if names:
                cv2.putText(frame, names[i], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def save_new_face(self, name):
        print("Opening camera...")
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture image.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            frame_with_boxes = self.draw_face_boxes(frame.copy(), face_locations)
            
            cv2.putText(frame_with_boxes, "Press 's' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame_with_boxes)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if face_locations:
                    print("Face detected. Processing...")
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        self.known_faces_encodings.append(face_encodings[0])
                        self.known_faces_names.append(name)
                        self.save_to_file()
                        print(f"Face captured and saved as {name}.")
                        break
                    else:
                        print("Face detected, but encoding failed. Please try again.")
                else:
                    print("No face detected. Please try again.")
            elif key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def recognize_face(self, name):
        print("Opening camera for recognition...")
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture image.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding, tolerance=self.tolerance)
                name_to_display = "Unknown"
                
                if True in matches:
                    matched_indexes = [i for i, match in enumerate(matches) if match]
                    counts = {}
                    for index in matched_indexes:
                        counts[self.known_faces_names[index]] = counts.get(self.known_faces_names[index], 0) + 1
                    name_to_display = max(counts, key=counts.get)
                
                names.append(name_to_display)
                
                # Debug information
                if name_to_display == name:
                    print(f"Match found for {name}!")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print(f"Face detected, but matched with {name_to_display} instead of {name}")
            
            frame_with_boxes = self.draw_face_boxes(frame.copy(), face_locations, names)
            
            cv2.putText(frame_with_boxes, "Press 's' to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Video', frame_with_boxes)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if not face_encodings:
                    print("No face detected. Please try again.")
            elif key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        print(f"No match found for {name}.")
        return False

    def save_to_file(self):
        with open(self.data_file, 'w') as file:
            for name, encoding in zip(self.known_faces_names, self.known_faces_encodings):
                file.write(f"{name}: {encoding.tolist()}\n")

def main():
    face_recognition_system = FaceRecognition()
    face_recognition_system.load_known_faces()

    while True:
        choice = input("Enter 'new' to capture a new face, 'old' to recognize an existing face, or 'exit' to quit: ")
        if choice == 'new':
            username = input("Enter a username for the new face: ")
            face_recognition_system.save_new_face(username)
        elif choice == 'old':
            username = input("Enter the username to recognize: ")
            face_recognition_system.recognize_face(username)
        elif choice == 'exit':
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()