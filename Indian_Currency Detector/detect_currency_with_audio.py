import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
import pyttsx3

model_path = "weights/best.pt"
if not os.path.exists(model_path):
    messagebox.showerror("Error", f"Model file not found at '{model_path}'. Please check the path.")
    exit()

model = YOLO(model_path)

currency_classes = [
    '10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '2000 Rupees',
    '50 Rupees', '500 Rupees', 'new10 Rupees', 'new100 Rupees', 'new20 Rupees', 'new50 Rupees'
]

cap = None
is_running = False
frame_skip = 2
frame_count = 0

tts_engine = pyttsx3.init()

def start_detection():
    global cap, is_running
    if is_running:
        messagebox.showinfo("Info", "Detection is already running.")
        return

    if camera_source.get() == "Webcam":
        cap = cv2.VideoCapture(0)
    elif camera_source.get() == "Phone Camera":
        ip_address = "http://192.168.187.193:4747/video"
        cap = cv2.VideoCapture(ip_address)

    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open {camera_source.get()}.")
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
    global cap, is_running, frame_count
    while is_running:
        ret, frame = cap.read()
        frame_count += 0
        if not ret:
            print("❌ Error: Failed to capture frame")
            stop_detection()
            return

        if frame_count % frame_skip != 0:
            continue

        resized_frame = cv2.resize(frame, (480, 360))

        results = model.predict(resized_frame, conf=0.6)
        detected_labels = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if cls_id >= len(currency_classes):
                    class_name = "Unknown Currency"
                else:
                    class_name = currency_classes[cls_id]

                print(f"Detected: {class_name} with confidence {conf:.2f}")
                detected_labels.append(class_name)

                scale_x = frame.shape[1] / resized_frame.shape[1]
                scale_y = frame.shape[0] / resized_frame.shape[0]
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        update_labels(detected_labels)

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
        detected_text.set("No currency detected")
        detected_label.config(fg="red")

def announce_labels(labels):
    if audio_assistant_enabled.get() and labels:
        for label in labels:
            tts_engine.say(label)
        tts_engine.runAndWait()

root = tk.Tk()
root.title("Indian Currency Detector")
root.configure(bg="#f0f0f0")

camera_source = tk.StringVar(value="Webcam")
audio_assistant_enabled = tk.BooleanVar(value=False)

header = tk.Frame(root, bg="#1e3d59", height=60)
header.pack(fill=tk.X)
header_label = tk.Label(header, text="Indian Currency Detector", font=("Arial", 24, "bold"), fg="white", bg="#1e3d59")
header_label.pack(pady=10)

main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

video_label = tk.Label(main_frame, bg="black", relief="sunken", bd=2)
video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

control_panel = tk.Frame(main_frame, width=300, bg="#ffffff", relief="raised", bd=2)
control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

camera_label = tk.Label(control_panel, text="Select Camera Source:", font=("Arial", 14), bg="#ffffff")
camera_label.pack(pady=10)
camera_dropdown = tk.OptionMenu(control_panel, camera_source, "Webcam", "Phone Camera")
camera_dropdown.config(width=20, font=("Arial", 12), bg="#e0e0e0", fg="black")
camera_dropdown.pack(pady=10)

detected_text = tk.StringVar()
detected_text.set("No currency detected")
detected_label = tk.Label(control_panel, textvariable=detected_text, font=("Arial", 16), fg="red", bg="#ffffff", wraplength=260, justify="center")
detected_label.pack(pady=20, padx=10)

start_button = tk.Button(control_panel, text="Start Detection", command=start_detection, width=20, height=2, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
start_button.pack(pady=10)

stop_button = tk.Button(control_panel, text="Stop Detection", command=stop_detection, width=20, height=2, bg="#f44336", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(pady=10)

exit_button = tk.Button(control_panel, text="Exit", command=root.quit, width=20, height=2, bg="#9e9e9e", fg="white", font=("Arial", 12, "bold"))
exit_button.pack(pady=10)

audio_assistant_checkbox = tk.Checkbutton(
    control_panel, text="Enable Audio Assistant", variable=audio_assistant_enabled,
    font=("Arial", 12), bg="#ffffff", anchor="w"
)
audio_assistant_checkbox.pack(pady=10, padx=10, fill=tk.X)

root.mainloop()
