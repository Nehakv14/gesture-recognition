import cv2
import mediapipe as mp
import csv
import numpy as np
import os

# -----------------------------
# Setup Mediapipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# -----------------------------
# File setup
# -----------------------------
file_path = 'hand_landmarks.csv'
file_exists = os.path.isfile(file_path)

gesture_name = input("Enter gesture name (e.g., help, fine, water, etc.): ").strip().lower()

# Open CSV file
file = open(file_path, 'a', newline='')
csv_writer = csv.writer(file)

# If file is new, add header
if not file_exists:
    header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
    csv_writer.writerow(header)

# -----------------------------
# Start Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)
print("Press 's' to save frame data, 'q' to quit")
sample_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Samples: {sample_count}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                row = landmarks + [gesture_name]
                csv_writer.writerow(row)
                sample_count += 1
                print(f"✅ Saved sample {sample_count} for gesture: {gesture_name}")

            elif key == ord('q'):
                print(f"\nTotal samples saved for '{gesture_name}': {sample_count}")
                cap.release()
                cv2.destroyAllWindows()
                file.close()
                exit()
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Collect Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nTotal samples saved for '{gesture_name}': {sample_count}")
            break

cap.release()
cv2.destroyAllWindows()
file.close()
