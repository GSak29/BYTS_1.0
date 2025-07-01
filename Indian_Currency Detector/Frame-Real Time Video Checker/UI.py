import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
import pyttsx3

# Path to the trained YOLOv8 model
model_path = "weights/best.pt"
if not os.path.exists(model_path):
    messagebox.showerror("Error", f"Model file not found at '{model_path}'. Please check the path.")
    exit()

model = YOLO(model_path)

# 32 ISL gesture classes from the YAML
hand_sign_classes = ['Bad', 'Brother', 'Father', 'Food', 'Friend', 'Good', 'Hello', 'Help', 'House', 'I', 'Indian', 'Loud', 'Mummy', 'Namaste', 'Name', 'No', 'Place', 'Please', 'Quiet', 'Sleeping', 'Sorry', 'Strong', 'Thank-you', 'Time', 'Today', 'Water', 'What', 'Yes', 'Your', 'language', 'sign', 'you']

cap = None
is_running = False
frame_skip = 2
frame_count = 0
last_detected = []

tts_engine = pyttsx3.init()

def start_detection():
    global cap, is_running
    if is_running:
        messagebox.showinfo("Info", "Detection is already running.")
        return

    source = camera_source.get()
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif source == "Phone Camera":
        ip_address = "http://192.168.174.37:4747/video"  # Change if needed
        cap = cv2.VideoCapture(ip_address)

    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open {source}.")
        return

    is_running = True
    threading.Thread(target=run_detection, daemon=True).start()

def stop_detection():
    global cap, is_running
    if not is_running:
        messagebox.showinfo("Info", "Detection is not running.")
        return

    is_running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Detection stopped.")

def run_detection():
    global cap, is_running, frame_count, last_detected

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            stop_detection()
            return

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        resized_frame = cv2.resize(frame, (480, 360))
        try:
            results = model.predict(resized_frame, conf=0.5, verbose=False)[0]
        except Exception as e:
            print(f"❌ Error during model prediction: {e}")
            stop_detection()
            return

        detected_labels = []

        try:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                class_name = hand_sign_classes[cls_id] if cls_id < len(hand_sign_classes) else "Unknown"
                print(f"Detected: {class_name} with confidence {conf:.2f}")
                detected_labels.append(class_name)

                # Scale bounding boxes to original frame
                scale_x = frame.shape[1] / resized_frame.shape[1]
                scale_y = frame.shape[0] / resized_frame.shape[0]
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                # Draw box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except AttributeError as e:
            print(f"❌ Error processing detection results: {e}")
            detected_labels = []

        update_labels(detected_labels)

        # Update video in GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

def update_labels(labels):
    if labels:
        detected_text.set("\n".join(labels))
        detected_label.config(fg="green")
        announce_labels(labels)
    else:
        detected_text.set("No hand sign detected")
        detected_label.config(fg="red")

def announce_labels(labels):
    global last_detected
    if audio_assistant_enabled.get() and labels != last_detected:
        for label in labels:
            tts_engine.say(label)
        tts_engine.runAndWait()
        last_detected = labels

# --- GUI Setup ---
root = tk.Tk()
root.title("Indian Sign Language Detector")
root.configure(bg="#f0f0f0")

camera_source = tk.StringVar(value="Webcam")
audio_assistant_enabled = tk.BooleanVar(value=False)

# Header
header = tk.Frame(root, bg="#1e3d59", height=60)
header.pack(fill=tk.X)
header_label = tk.Label(header, text="Indian Sign Language Detector", font=("Arial", 24, "bold"), fg="white", bg="#1e3d59")
header_label.pack(pady=10)

# Main Content
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Video Display
video_label = tk.Label(main_frame, bg="black", relief="sunken", bd=2)
video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Control Panel
control_panel = tk.Frame(main_frame, width=300, bg="#ffffff", relief="raised", bd=2)
control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

camera_label = tk.Label(control_panel, text="Select Camera Source:", font=("Arial", 14), bg="#ffffff")
camera_label.pack(pady=10)

camera_dropdown = tk.OptionMenu(control_panel, camera_source, "Webcam", "Phone Camera")
camera_dropdown.config(width=20, font=("Arial", 12), bg="#e0e0e0", fg="black")
camera_dropdown.pack(pady=10)

detected_text = tk.StringVar()
detected_text.set("No hand sign detected")
detected_label = tk.Label(control_panel, textvariable=detected_text, font=("Arial", 16), fg="red", bg="#ffffff", wraplength=260, justify="center")
detected_label.pack(pady=20, padx=10)

start_button = tk.Button(control_panel, text="Start Detection", command=start_detection, width=20, height=2, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
start_button.pack(pady=10)

stop_button = tk.Button(control_panel, text="Stop Detection", command=stop_detection, width=20, height=2, bg="#f44336", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(pady=10)

exit_button = tk.Button(control_panel, text="Exit", command=root.quit, width=20, height=2, bg="#9e9e9e", fg="white", font=("Arial", 12, "bold"))
exit_button.pack(pady=10)

audio_assistant_checkbox = tk.Checkbutton(control_panel, text="Enable Audio Assistant", variable=audio_assistant_enabled, font=("Arial", 12), bg="#ffffff", anchor="w")
audio_assistant_checkbox.pack(pady=10, padx=10, fill=tk.X)

root.mainloop()
