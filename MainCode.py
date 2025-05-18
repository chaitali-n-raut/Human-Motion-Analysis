import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

# === Utility Functions ===
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arccos(
        np.clip(np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0)
    )
    return int(np.degrees(radians))

# === MediaPipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# === Function to select video file from the user ===
def select_video():
    # Create a Tkinter window to allow the user to select a file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a Video", 
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    return file_path

# === Get Video Path from User ===
video_path = select_video()

# Check if the user selected a file
if not video_path:
    print("No video file selected. Exiting.")
    exit()

# === Open Video Capture ===
cap = cv2.VideoCapture(video_path)

# === Tracking Variables ===
prev_time = 0
prev_position = None
total_distance = 0.0

# === Main Loop ===
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended or cannot be loaded.")
            break

        # Get video frame dimensions
        h, w, _ = frame.shape

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect Pose
        results = pose.process(image)

        # Convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If pose landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract shoulder, elbow, and wrist positions
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Convert to pixel coordinates
            shoulder_pixel = (int(shoulder[0] * w), int(shoulder[1] * h))
            elbow_pixel = (int(elbow[0] * w), int(elbow[1] * h))
            wrist_pixel = (int(wrist[0] * w), int(wrist[1] * h))
            hip_pixel = (int(hip[0] * w), int(hip[1] * h))

            # Calculate movement distance
            if prev_position is not None:
                distance = calculate_distance(shoulder_pixel, prev_position)
                total_distance += distance / 100  # scale to meters approx

                # Calculate time and speed
                curr_time = cv2.getTickCount() / cv2.getTickFrequency()
                time_diff = curr_time - prev_time if prev_time > 0 else 1e-6
                speed = (distance / time_diff) * 0.036  # pixels/sec to km/h
                prev_time = curr_time
            else:
                speed = 0
                distance = 0
                prev_time = cv2.getTickCount() / cv2.getTickFrequency()

            prev_position = shoulder_pixel

            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Adjust font size relative to video dimensions
            font_scale = min(w, h) / 600  # Scale the font based on video size
            thickness = int(min(w, h) / 500)  # Adjust thickness based on video size

            # Calculate text size
            speed_text = f"Speed: {speed:.2f} km/h"
            distance_text = f"Distance: {total_distance:.2f} meters"
            elbow_angle_text = f"Elbow Angle: {elbow_angle} degrees"

            # Get text size for dynamic adjustment
            text_size_speed = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_size_distance = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_size_angle = cv2.getTextSize(elbow_angle_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            # Dynamic positioning based on video size
            text_x = int(w * 0.05)
            text_y_speed = int(h * 0.05)
            text_y_distance = text_y_speed + text_size_speed[1] + int(h * 0.02)
            text_y_angle = text_y_distance + text_size_distance[1] + int(h * 0.02)

            # Adding black background boxes for better text readability
            cv2.rectangle(image, (text_x - 5, text_y_speed - 30), (text_x + text_size_speed[0] + 5, text_y_speed + text_size_speed[1] + 5), (0, 0, 0), -1)  # Speed text background
            cv2.rectangle(image, (text_x - 5, text_y_distance - 30), (text_x + text_size_distance[0] + 5, text_y_distance + text_size_distance[1] + 5), (0, 0, 0), -1)  # Distance text background
            cv2.rectangle(image, (text_x - 5, text_y_angle - 30), (text_x + text_size_angle[0] + 5, text_y_angle + text_size_angle[1] + 5), (0, 0, 0), -1)  # Elbow Angle background

            # White text to make sure it stands out on any background
            cv2.putText(image, speed_text, (text_x, text_y_speed), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(image, distance_text, (text_x, text_y_distance), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(image, elbow_angle_text, (text_x, text_y_angle), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show frame
        cv2.imshow("Motion Analysis", image)

        # Handle window close event
        if cv2.getWindowProperty("Motion Analysis", cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
