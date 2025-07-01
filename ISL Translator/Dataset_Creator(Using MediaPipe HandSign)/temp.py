import os
import cv2
import time
import uuid
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk

# === Folder Setup ===
BASE_PATH = 'datasets/train'
IMG_PATH = os.path.join(BASE_PATH, 'images')
LABEL_PATH = os.path.join(BASE_PATH, 'labels')

os.makedirs(IMG_PATH, exist_ok=True)
os.makedirs(LABEL_PATH, exist_ok=True)

# === Labels and Class Mapping ===
labels = ['Hello', 'Yes', 'No', 'Thanks', 'ILoveYou', 'Please']
label_map = {label: idx for idx, label in enumerate(labels)}

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# === Global Variables ===
paused = True
current_label_idx = 0
img_num = 0
number_of_images = 20

# === Camera Capture ===
cap = cv2.VideoCapture(0)

# === Tkinter Functions ===
def start_capturing():
    global paused
    paused = False
    status_var.set("Status: Collecting Images...")

def stop_capturing():
    global paused
    paused = True
    status_var.set("Status: Paused")

def next_label():
    global current_label_idx, img_num, paused
    if current_label_idx < len(labels) - 1:
        current_label_idx += 1
        img_num = 0
        paused = True
        update_label()
        status_var.set("Status: Ready")
    else:
        messagebox.showinfo("Info", "All labels completed!")
        quit_app()

def quit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

def update_label():
    label_var.set(f"Current Label: {labels[current_label_idx]}")

def update_progress():
    progress_bar["value"] = img_num
    total_progress = (current_label_idx * number_of_images) + img_num
    total_progress_bar["value"] = total_progress
    image_count_var.set(f"Images Captured: {img_num}/{number_of_images}")

# === Tkinter UI Setup ===
root = tk.Tk()
root.title("ISL Dataset Collector")
root.geometry("1000x600")  # Adjusted size for a cleaner layout
root.configure(bg="#f5f5f5")

# Main layout frames
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Video Frame
video_label = tk.Label(main_frame, bg="#000000", width=800, height=500)
video_label.pack(side="left", padx=20, pady=20)

# Controls Frame
controls_frame = tk.Frame(main_frame, bg="#f5f5f5")
controls_frame.pack(side="right", padx=20, pady=20)

title = tk.Label(controls_frame, text="ISL Dataset Collector", font=('Helvetica', 24, 'bold'), bg="#f5f5f5", fg="#333333")
title.pack(pady=10)

label_var = tk.StringVar()
label_var.set(f"Current Label: {labels[current_label_idx]}")
label_display = tk.Label(controls_frame, textvariable=label_var, font=('Helvetica', 18), bg="#f5f5f5", fg="#555555")
label_display.pack(pady=5)

status_var = tk.StringVar()
status_var.set("Status: Ready")
status_display = tk.Label(controls_frame, textvariable=status_var, font=('Helvetica', 16), bg="#f5f5f5", fg="#28a745")
status_display.pack(pady=5)

image_count_var = tk.StringVar()
image_count_var.set(f"Images Captured: {img_num}/{number_of_images}")
image_count_display = tk.Label(controls_frame, textvariable=image_count_var, font=('Helvetica', 16), bg="#f5f5f5", fg="#007bff")
image_count_display.pack(pady=5)

button_frame = tk.Frame(controls_frame, bg="#f5f5f5")
button_frame.pack(pady=20)

start_btn = tk.Button(button_frame, text="Start Capturing", width=15, height=2, bg="#28a745", fg="white",
                      font=('Helvetica', 12, 'bold'), command=start_capturing)
start_btn.grid(row=0, column=0, padx=10, pady=10)

stop_btn = tk.Button(button_frame, text="Stop Capturing", width=15, height=2, bg="#ffc107", fg="white",
                     font=('Helvetica', 12, 'bold'), command=stop_capturing)
stop_btn.grid(row=0, column=1, padx=10, pady=10)

next_btn = tk.Button(button_frame, text="Next Label", width=15, height=2, bg="#007bff", fg="white",
                     font=('Helvetica', 12, 'bold'), command=next_label)
next_btn.grid(row=1, column=0, padx=10, pady=10)

quit_btn = tk.Button(button_frame, text="Quit", width=15, height=2, bg="#dc3545", fg="white",
                     font=('Helvetica', 12, 'bold'), command=quit_app)
quit_btn.grid(row=1, column=1, padx=10, pady=10)

# === Capture Loop ===
def capture_loop():
    global img_num
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords) * w)
                y_min = int(min(y_coords) * h)
                x_max = int(max(x_coords) * w)
                y_max = int(max(y_coords) * h)

                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                # Draw bounding box directly on the video feed
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, labels[current_label_idx], (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f'Images: {img_num}/{number_of_images}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert frame to ImageTk format and update video_label
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if not paused and results.multi_hand_landmarks:
            img_id = str(uuid.uuid1())
            img_filename = os.path.join(IMG_PATH, f"{labels[current_label_idx]}_{img_id}.jpg")
            cv2.imwrite(img_filename, frame)

            # Save label in YOLO format
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h

            label_filename = os.path.join(LABEL_PATH, f"{labels[current_label_idx]}_{img_id}.txt")
            with open(label_filename, 'w') as f:
                f.write(f"{label_map[labels[current_label_idx]]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

            img_num += 1
            image_count_var.set(f"Images Captured: {img_num}/{number_of_images}")
            time.sleep(1)

            if img_num >= number_of_images:
                stop_capturing()
                next_label()

        root.update_idletasks()
        root.update()

capture_thread = threading.Thread(target=capture_loop)
capture_thread.start()

root.mainloop()
