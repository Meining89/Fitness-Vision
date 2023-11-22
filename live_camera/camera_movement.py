import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import math
from utils.angles import *
from utils.draw_display import *
from utils.constant import *
from collections import deque


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize shoulder Y positions
    shoulder_positions = deque(maxlen=NUM_FRAMES_FOR_AVERAGE)
    left_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)
    right_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)

    # Initalize counter
    count = 0
    going_up = False

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (change it if you have multiple cameras)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_height, frame_width, _ = frame.shape

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame,
                                                      results.pose_landmarks,
                                                      mp_pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
                                                                                             thickness=15,
                                                                                             circle_radius=5),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255),
                                                                                             thickness=15,
                                                                                             circle_radius=5)
                                                      )

            # Get Y positions of the left and right shoulders
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Update deque with shoulder positions
            shoulder_positions.append((left_shoulder_y, right_shoulder_y))

            average_left_shoulder_y = sum(pos[0] for pos in shoulder_positions) / len(shoulder_positions)
            average_right_shoulder_y = sum(pos[1] for pos in shoulder_positions) / len(shoulder_positions)

            ###################### Calculate knee angles ######################

            left_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)
            left_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)
            left_ankle = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)

            right_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
            right_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)
            right_ankle = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

            # Calculate knee angles
            left_knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle_3d(right_hip, right_knee, right_ankle)

            left_knee_angles.append(left_knee_angle)
            right_knee_angles.append(right_knee_angle)

            # Calculate moving average of knee angles
            average_left_knee_angle = sum(left_knee_angles) / len(left_knee_angles)
            average_right_knee_angle = sum(right_knee_angles) / len(right_knee_angles)

            left_knee_pixel_x = int(left_knee[0] * frame_width)
            left_knee_pixel_y = int(left_knee[1] * frame_height)
            knee_loc = (left_knee_pixel_x, left_knee_pixel_y)
            knee_angle = min(average_left_knee_angle, average_right_knee_angle)

            # Draw the left leg in red if the knee angle is greater than the threshold
            if knee_angle > KNEE_ANGLE_DEPTH:
                draw_leg_landmarks(mp, frame, results)
            else:
                draw_leg_landmarks(mp, frame, results, color=(0, 255, 0))

            # Compare with previous Y positions to determine movement direction
            if left_shoulder_y < average_left_shoulder_y and right_shoulder_y < average_right_shoulder_y:
                direction_text = "UP"
                # Change in direction: going up now
                if not going_up:
                    count += 1
                    going_up = True

            elif left_shoulder_y > average_left_shoulder_y and right_shoulder_y > average_right_shoulder_y:
                direction_text = "DOWN"
                going_up = False
            else:
                direction_text = "STABLE"

            # Display the direction text on the frame
            cycle_x = 50
            cycle_y = 100
            text_to_display = f"{direction_text} | Cycles: {count}"
            draw_text(frame, (cycle_x, cycle_y), text_to_display)

            knee_x = 50
            knee_y = 200
            knee_text = f"Left Knee: {average_left_knee_angle:.2f} degrees | Right Knee: {average_right_knee_angle:.2f} degrees"
            draw_text(frame, (knee_x, knee_y), knee_text)

            knee_angle_text = f"{knee_angle:.2f} degrees"
            draw_text(frame, knee_loc, knee_angle_text)

            # Update previous Y positions
            prev_left_shoulder_y = left_shoulder_y
            prev_right_shoulder_y = right_shoulder_y

        # Display the resulting frame
        cv2.imshow('Pose Estimation', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
