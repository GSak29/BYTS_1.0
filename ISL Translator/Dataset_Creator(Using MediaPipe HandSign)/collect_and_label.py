import os
import cv2
import time
import uuid
import mediapipe as mp

# === Folder Setup ===
BASE_PATH = 'datasets/train'
IMG_PATH = os.path.join(BASE_PATH, 'images')
LABEL_PATH = os.path.join(BASE_PATH, 'labels')

os.makedirs(IMG_PATH, exist_ok=True)
os.makedirs(LABEL_PATH, exist_ok=True)

# === Labels and Class Mapping ===
labels = ['Hello', 'Yes', 'No', 'Thanks', 'ILoveYou', 'Please']
label_map = {label: idx for idx, label in enumerate(labels)}

# === Mediapipe Hand Detection Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# === Number of images to collect per class ===
number_of_images = 20

for label in labels:
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(5)

    for img_num in range(number_of_images):
        ret, frame = cap.read()
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

                # Add margin
                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                # Save image
                img_id = str(uuid.uuid1())
                img_filename = os.path.join(IMG_PATH, f"{label}_{img_id}.jpg")
                cv2.imwrite(img_filename, frame)

                # YOLO normalized format
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                bbox_width = (x_max - x_min) / w
                bbox_height = (y_max - y_min) / h

                # Save label
                label_filename = os.path.join(LABEL_PATH, f"{label}_{img_id}.txt")
                with open(label_filename, 'w') as f:
                    f.write(f"{label_map[label]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        time.sleep(1.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
