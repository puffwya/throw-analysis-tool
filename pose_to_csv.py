import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe Pose (legacy API)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video file path
video_path = "throw.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video file: {video_path}")

all_landmarks = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(frame_rgb)

    # Extract landmarks
    if results.pose_landmarks:
        frame_data = []
        for lm in results.pose_landmarks.landmark:
            frame_data.extend([lm.x, lm.y, lm.z])
        all_landmarks.append(frame_data)
    else:
        # Store NaNs if no landmarks detected
        all_landmarks.append([float("nan")] * 33 * 3)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
print(f"Total frames processed: {frame_count}")

# Save landmarks to CSV
df = pd.DataFrame(all_landmarks)
df.to_csv("throw_landmarks.csv", index=False)
print("Landmarks saved to throw_landmarks.csv")
