import cv2
import csv
import mediapipe as mp
import math
import sys
import os

# === Inputs from command line ===
input_video = sys.argv[1]
output_video = "outputs/processed_throw.avi"

# === Helper: calculate angle between three points ===
def calculate_angle(a, b, c):
    import numpy as np
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# === Video Setup ===
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Make sure the output folder exists
os.makedirs(os.path.dirname(output_video), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPEG works in AVI
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Mediapipe Pose Setup ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

angles_data = []
dominant_side = None
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Convenience function to get coordinates
        def lm_coords(name):
            point = lm[mp_pose.PoseLandmark[name].value]
            return [point.x, point.y]

        # Determine dominant side on first frame
        if frame_idx == 0:
            left_wrist, right_wrist = lm_coords("LEFT_WRIST"), lm_coords("RIGHT_WRIST")
            dominant_side = "LEFT" if right_wrist[0] > left_wrist[0] else "RIGHT"

        # Assign side-specific landmarks
        if dominant_side == "RIGHT":
            shoulder, elbow, wrist = lm_coords("RIGHT_SHOULDER"), lm_coords("RIGHT_ELBOW"), lm_coords("RIGHT_WRIST")
            hip, knee, ankle = lm_coords("RIGHT_HIP"), lm_coords("RIGHT_KNEE"), lm_coords("RIGHT_ANKLE")
        else:
            shoulder, elbow, wrist = lm_coords("LEFT_SHOULDER"), lm_coords("LEFT_ELBOW"), lm_coords("LEFT_WRIST")
            hip, knee, ankle = lm_coords("LEFT_HIP"), lm_coords("LEFT_KNEE"), lm_coords("LEFT_ANKLE")

        # Calculate angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        hip_angle = calculate_angle(shoulder, hip, ankle)
        knee_angle = calculate_angle(hip, knee, ankle)

        # Print for inspection
        print(f"Frame {frame_idx}: Side={dominant_side}, "
              f"Elbow={elbow_angle:.1f}, Shoulder={shoulder_angle:.1f}, "
              f"Hip={hip_angle:.1f}, Knee={knee_angle:.1f}")

        # Save angles
        angles_data.append({
            "frame": frame_idx,
            "throwing_side": dominant_side,
            "elbow_angle": elbow_angle,
            "shoulder_angle": shoulder_angle,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle
        })

        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay angle text
        cv2.putText(frame, f"Elbow: {elbow_angle:.1f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Hip: {hip_angle:.1f}", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Knee: {knee_angle:.1f}", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Write the frame to output video
    out.write(frame)
    frame_idx += 1

# Release everything
cap.release()
out.release()
pose.close()

# === Save angles CSV ===
csv_file = os.path.splitext(output_video)[0] + "_angles.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "throwing_side", "elbow_angle", "shoulder_angle", "hip_angle", "knee_angle"])
    writer.writeheader()
    for row in angles_data:
        writer.writerow(row)

print(f"\nSaved angles to {csv_file}")
print(f"Processed video saved to {output_video}")
