import cv2
import mediapipe as mp
import numpy as np
import os

# === Mediapipe Pose setup ===
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
STICK_FIGURE_COLOR = (0, 255, 0)  # Green

# === Ask for video source ===
source = input("Enter path to video (or press Enter for webcam): ").strip()
if source == "":
    cap = cv2.VideoCapture(0)  # Use webcam
else:
    if not os.path.exists(source):
        print("ERROR: File not found.")
        exit()
    cap = cv2.VideoCapture(source)  # Use provided video file

if not cap.isOpened():
    print("ERROR: Could not open video source.")
    exit()

# === Ask if stick figure should mirror or copy exactly ===
mirror_choice = input("Do you want the stick figure to MIRROR your movement? (y/n): ").strip().lower()
mirror_mode = (mirror_choice == "y")

# === Ask if output should be saved ===
save_choice = input("Do you want to save the output video in 'outputs' folder? (y/n): ").strip().lower()
save_video = (save_choice == "y")

# === Video info ===
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Prepare output video writer if saving is enabled ===
if save_video:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "stick_figure_buddy.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
else:
    out = None

# === Draw stick figure ===
def draw_stick_figure(image, landmarks, image_width, image_height, mirror_mode=False):
    """Draw stick figure as buddy (non-mirror with gap) or mirrored copy (mirror mode)."""
    if not landmarks:
        return

    GAP_RATIO = 0.20  # 20% of frame width
    gap = int(image_width * GAP_RATIO)

    for connection in POSE_CONNECTIONS:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]

        if start_point.visibility > 0.5 and end_point.visibility > 0.5:
            if mirror_mode:
                # Mirrored: flip horizontally, no gap shift
                x1 = int((1 - start_point.x) * image_width)-gap
                x2 = int((1 - end_point.x) * image_width)-gap
            else:
                # Buddy: normal coords but shifted left for side-by-side
                x1 = int(start_point.x * image_width) - gap
                x2 = int(end_point.x * image_width) - gap

            # Keep within frame bounds
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width - 1))

            y1 = int(start_point.y * image_height)
            y2 = int(end_point.y * image_height)

            cv2.line(image, (x1, y1), (x2, y2), STICK_FIGURE_COLOR, 2)

# === Overlay with transparency ===
def overlay_transparent(background, overlay, alpha=0.5):
    mask = overlay > 0
    background[mask] = cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)[mask]
    return background

# === Main loop ===
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Flip to feel natural

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        stick_layer = np.zeros_like(frame)

        if results.pose_landmarks:
            draw_stick_figure(stick_layer, results.pose_landmarks.landmark, width, height, mirror_mode)

        final_frame = overlay_transparent(frame.copy(), stick_layer, alpha=1)

        cv2.namedWindow("Virtual Buddy Stick Figure", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Virtual Buddy Stick Figure", width, height)
        cv2.imshow("Virtual Buddy Stick Figure", final_frame)

        if save_video and out:
            out.write(final_frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
if save_video and out:
    out.release()
cv2.destroyAllWindows()

if save_video:
    print(f"✅ Video saved to: {output_filename}")
else:
    print("ℹ Video not saved.")
