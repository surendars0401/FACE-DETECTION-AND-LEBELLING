import cv2
import face_recognition
from tkinter import *
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk
import os
from threading import Thread, Lock
import numpy as np

# Ensure known_faces directory exists
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')

# Initialize variables
known_face_encodings = []
known_face_names = []
frame_to_process = None
frame_lock = Lock()
process_this_frame = True

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir('known_faces'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            face_image = face_recognition.load_image_file(f'known_faces/{filename}')
            face_encodings = face_recognition.face_encodings(face_image)
            if face_encodings:  # Check if list is not empty
                face_encoding = face_encodings[0]  # Use the first face encoding found
                known_face_encodings.append(face_encoding)
                known_face_names.append(filename[:-4])  # Remove file extension from name
            else:
                print(f"No faces found in {filename}.")

# Capture frames in a separate thread
def capture_frames():
    global frame_to_process, frame_lock
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        with frame_lock:
            frame_to_process = frame.copy()

# Initialize camera
camera = cv2.VideoCapture(0)

# Start capture thread
capture_thread = Thread(target=capture_frames, daemon=True)
capture_thread.start()

def capture_and_save_face():
    global frame_to_process
    if frame_to_process is not None:
        name = askstring("Input", "Enter the person's name:")
        if name:
            with frame_lock:
                cv2.imwrite(f'known_faces/{name}.jpg', frame_to_process)
            load_known_faces()
            print(f"Saved {name}'s face and updated the dataset.")

def detect_and_display():
    global frame_to_process, process_this_frame
    if frame_to_process is not None:
        # Process every other frame to save time
        if process_this_frame:
            with frame_lock:
                small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame_to_process, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame_to_process, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            pil_image = Image.fromarray(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB))
            tk_image = ImageTk.PhotoImage(pil_image)
            image_label.config(image=tk_image)
            image_label.image = tk_image

        process_this_frame = not process_this_frame
    window.after(10, detect_and_display)

window = Tk()
window.title("Face Recognition with Capture Feature")

image_label = Label(window)
image_label.pack()

capture_button = Button(window, text="Capture Known Face", command=capture_and_save_face)
capture_button.pack()

detect_and_display()

window.mainloop()

camera.release()
