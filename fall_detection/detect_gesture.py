# final_detection.py
import cv2
import mediapipe as mp
import numpy as np
import time
import imutils
from collections import deque
import pickle
import sys
import os

# -----------------------------
# Configuration Parameters
# -----------------------------
Y_VEL_WINDOW = 5             # frames used to compute velocity
VELOCITY_THRESHOLD = 60      # px/sec threshold for sudden fall (tweakable)
ANGLE_HORIZONTAL_TH = 35     # degrees -> below this is considered horizontal
ANGLE_VERTICAL_TH = 60       # degrees -> above this considered upright
SUSTAIN_FRAMES = 10          # consecutive frames of "flat" to confirm fall
HIP_HISTORY_LEN = 80         # how many hip positions to store
RECOVERY_MIN_SECONDS = 3     # minimal seconds before allowing another fall detection
DISPLAY_MSG_SECONDS = 4      # how long to show messages
SHOULDER_DIFF_THRESHOLD = 0.08  # normalized coordinate difference (tune 0.06-0.16)
SHOULDER_DIFF_PIX_THRESHOLD = 40  # alternate pixel threshold if using pixels

# -----------------------------
# Load gesture recognition model (optional)
# -----------------------------
MODEL_PATH = 'gesture_model.pkl'
gesture_model = None
if os.path.exists(MODEL_PATH):
    try:
        gesture_model = pickle.load(open(MODEL_PATH, 'rb'))
        print("[INFO] Loaded gesture model:", MODEL_PATH)
    except Exception as e:
        print("[WARN] Failed to load gesture_model.pkl:", e)
        gesture_model = None
else:
    print("[WARN] gesture_model.pkl not found. Gesture recognition will be disabled.")

# -----------------------------
# Initialize Mediapipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# -----------------------------
# Helper functions
# -----------------------------
def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

def torso_angle_deg(shoulder_mid, hip_mid):
    """
    Returns angle in degrees where:
      - ~90 deg => vertical (standing)
      - ~0 deg  => horizontal (lying)
    """
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle

def get_landmark_coords(landmark, image_w, image_h):
    """
    Return integer pixel coords and visibility (0..1).
    Also returns normalized coords for shoulder-diff detection.
    """
    # normalized x,y (0..1)
    nx = landmark.x
    ny = landmark.y
    x = int(np.clip(nx * image_w, 0, image_w - 1))
    y = int(np.clip(ny * image_h, 0, image_h - 1))
    vis = landmark.visibility if hasattr(landmark, "visibility") else 1.0
    return (x, y, nx, ny, vis)

def smooth_velocity(y_deque, t_deque, use_frames=Y_VEL_WINDOW):
    """
    Compute smoothed vertical velocity using the difference between
    the most recent sample and the sample 'use_frames' back.
    Returns px/sec (positive when moving downward on image).
    """
    if len(y_deque) < 2:
        return 0.0
    N = min(use_frames, len(y_deque) - 1)
    dy = y_deque[-1] - y_deque[-1 - N]  # positive if hip moved downwards (fall)
    dt = t_deque[-1] - t_deque[-1 - N]
    if dt <= 0:
        dt = 1e-3
    return dy / dt

# -----------------------------
# Messages for gestures
# -----------------------------
messages = {
    'help': "Patient needs HELP!",
    'fine': "Patient is fine.",
    'water': "Patient needs water.",
    'call': "Patient is calling the nurse!"
}

# -----------------------------
# Setup Webcam + State
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    sys.exit(1)

hip_y_hist = deque(maxlen=HIP_HISTORY_LEN)
time_hist = deque(maxlen=HIP_HISTORY_LEN)
sustain_counter = 0
fall_detected = False
last_fall_time = 0.0
prev_gesture = None
display_message = ""
display_until = 0.0

fps_time_prev = time.time()

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, exiting.")
            break

        frame = imutils.resize(frame, width=800)
        frame = cv2.flip(frame, 1)
        image_h, image_w = frame.shape[:2]

        # Convert for Mediapipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose and hands
        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        # defaults
        current_angle = 90.0
        current_vel = 0.0
        gesture = None

        # -----------------------------
        # --- FALL DETECTION PART ---
        # -----------------------------
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            # Get key landmarks (shoulders and hips) with both pixel and normalized coords
            L_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            R_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            L_hp = lm[mp_pose.PoseLandmark.LEFT_HIP]
            R_hp = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            lsh_x, lsh_y, lsh_nx, lsh_ny, lsh_vis = get_landmark_coords(L_sh, image_w, image_h)
            rsh_x, rsh_y, rsh_nx, rsh_ny, rsh_vis = get_landmark_coords(R_sh, image_w, image_h)
            lhp_x, lhp_y, lhp_nx, lhp_ny, lhp_vis = get_landmark_coords(L_hp, image_w, image_h)
            rhp_x, rhp_y, rhp_nx, rhp_ny, rhp_vis = get_landmark_coords(R_hp, image_w, image_h)

            sh_mid = midpoint((lsh_x, lsh_y), (rsh_x, rsh_y))
            hp_mid = midpoint((lhp_x, lhp_y), (rhp_x, rhp_y))

            # Hip vertical position history (in pixels)
            hip_y = hp_mid[1]
            hip_y_hist.append(hip_y)
            time_hist.append(time.time())

            # Smoothed vertical velocity (px/sec). Positive when moving down.
            current_vel = smooth_velocity(hip_y_hist, time_hist, use_frames=Y_VEL_WINDOW)

            # Torso angle (0 = horizontal, 90 = vertical)
            current_angle = torso_angle_deg(sh_mid, hp_mid)

            # Shoulder differences (both pixel and normalized X/Y)
            shoulder_y_diff = abs(lsh_y - rsh_y)           # pixel vertical diff
            shoulder_x_diff = abs(lsh_x - rsh_x)           # pixel horizontal diff
            shoulder_norm_diff = abs(lsh_ny - rsh_ny)      # normalized Y diff (tilt forward/back)
            shoulder_norm_xdiff = abs(lsh_nx - rsh_nx)     # normalized X diff (tilt left/right)

            # Draw pose skeleton only (kept as requested)
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Fall detection conditions
            # 1) Large shoulder vertical or horizontal difference (covers left/right tilt)
            cond_shoulder = (
                (shoulder_norm_diff > SHOULDER_DIFF_THRESHOLD) or
                (shoulder_norm_xdiff > SHOULDER_DIFF_THRESHOLD) or
                (shoulder_y_diff > SHOULDER_DIFF_PIX_THRESHOLD) or
                (shoulder_x_diff > SHOULDER_DIFF_PIX_THRESHOLD)
            )

            # 2) Body horizontal (torso angle small)
            cond_angle = (current_angle < ANGLE_HORIZONTAL_TH)

            # 3) Sudden downward hip velocity
            cond_vel = (current_vel > VELOCITY_THRESHOLD)

            # Combine: accept either (sustained shoulder tilt) OR (sudden drop + horizontal)
            if cond_shoulder:
                sustain_counter += 1
            else:
                sustain_counter = 0

            time_since_last_fall = time.time() - last_fall_time
            rule_trigger = (sustain_counter >= SUSTAIN_FRAMES) or (cond_vel and cond_angle)

            if rule_trigger and (not fall_detected) and (time_since_last_fall > RECOVERY_MIN_SECONDS):
                # verify hips became stable (optional)
                recent = list(hip_y_hist)[-SUSTAIN_FRAMES:] if len(hip_y_hist) >= SUSTAIN_FRAMES else list(hip_y_hist)
                stable_after = False
                if len(recent) >= 3:
                    stable_after = (max(recent) - min(recent)) < 40
                else:
                    stable_after = True  # accept when not enough samples

                if stable_after:
                    fall_detected = True
                    last_fall_time = time.time()
                    display_message = "⚠️ FALL DETECTED!"
                    display_until = time.time() + DISPLAY_MSG_SECONDS
                    print("[ALERT] FALL DETECTED at", time.strftime("%Y-%m-%d %H:%M:%S"))
                # else: do not set until stable

            # Recovery: if person becomes upright again AND some time has passed, reset
            if fall_detected and current_angle > ANGLE_VERTICAL_TH:
                fall_detected = False
                sustain_counter = 0
                last_fall_time = time.time()
                display_message = "Person upright. Fall flag reset."
                display_until = time.time() + DISPLAY_MSG_SECONDS
                print("[INFO] Person upright again. Resetting fall flag.")

        # -----------------------------
        # --- GESTURE RECOGNITION PART ---
        # -----------------------------
        if hand_results and hand_results.multi_hand_landmarks and gesture_model is not None:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Build flattened normalized landmark vector [x1,y1,x2,y2,...]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks = np.array(landmarks).reshape(1, -1)
                try:
                    prediction = gesture_model.predict(landmarks)
                    gesture = prediction[0]
                except Exception as e:
                    # try cast
                    try:
                        prediction = gesture_model.predict(landmarks.astype(np.float32))
                        gesture = prediction[0]
                    except Exception as e2:
                        print("[WARN] Gesture model prediction failed:", e2)
                        gesture = None

                if gesture:
                    # Show gesture text (short-lived)
                    cv2.putText(frame, f"{str(gesture).upper()}", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                break  # only first detected hand

        # Handle gesture message display (mapped verbosely)
        if gesture and gesture != prev_gesture:
            if gesture in messages:
                display_message = messages[gesture]
                display_until = time.time() + DISPLAY_MSG_SECONDS
                print("[INFO] Gesture message:", display_message)
            prev_gesture = gesture

        # Expire display message after timeout
        if display_message and time.time() > display_until:
            display_message = ""

        # Draw the display message (bottom area) - only UI element besides pose/hand skeletons
        if display_message:
            cv2.rectangle(frame, (30, image_h - 120), (image_w - 30, image_h - 30), (0, 0, 0), -1)
            cv2.putText(frame, display_message, (40, image_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show final frame
        cv2.imshow("Combined Fall & Gesture Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    print("[INFO] Exited cleanly.")
