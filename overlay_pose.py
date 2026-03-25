import cv2
import mediapipe as mp
import pandas as pd

# ---------- Load CSV ----------
angles_df = pd.read_csv("throw_angles.csv")

# ---------- Mediapipe setup ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ---------- Video setup ----------
cap = cv2.VideoCapture("throw.mp4")
out = cv2.VideoWriter(
    "throw_with_angles.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay angles from CSV if available
        if frame_idx < len(angles_df):
            row = angles_df.iloc[frame_idx]
            text_y = 30
            for joint in ["elbow_angle", "shoulder_angle", "hip_angle", "knee_angle"]:
                cv2.putText(
                    frame,
                    f"{joint}: {row[joint]:.1f}",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                text_y += 30

            # Optional: show throwing side
            cv2.putText(
                frame,
                f"Throwing side: {row['throwing_side']}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    # Write frame to output video
    out.write(frame)

    frame_idx += 1
    # Optional: display live
    # cv2.imshow("Throw with Angles", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
pose.close()
# cv2.destroyAllWindows()
print("Saved throw_with_angles.mp4")
