import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import math
from time import sleep
from collections import deque


# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)# Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
  #extract keypoints and convert to np array
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
    
def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def calculate_angle_3D(a,b,c):
    """
    Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another

    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def get_coordinates(landmarks, mp_pose, side, joint):
    """
    Retrieves x and y coordinates of a particular keypoint from the pose estimation model

     Args:
         landmarks: processed keypoints from the pose estimation model
         mp_pose: Mediapipe pose estimation model
         side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
         joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.

    """
    coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
    x_coord_val = landmarks[coord.value].x
    y_coord_val = landmarks[coord.value].y
    return [x_coord_val, y_coord_val]


def calculate_angle_3d(a,b,c):
    """
    Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another

    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle


def draw_text(frame, position, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, color=(0, 255, 0), thickness=2):
    """
    Draws text on a frame

    """
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize shoulder Y positions
    prev_left_shoulder_y = 0
    prev_right_shoulder_y = 0
    num_frames_for_average = 30
    shoulder_positions = deque(maxlen=num_frames_for_average)
    left_knee_angles = deque(maxlen=num_frames_for_average)
    right_knee_angles = deque(maxlen=num_frames_for_average)

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

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, 
                                                      results.pose_landmarks, 
                                                      mp_pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=15, circle_radius=5),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=15, circle_radius=5)
                                                    )
            
            # Get Y positions of the left and right shoulders
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Update deque with shoulder positions
            shoulder_positions.append((left_shoulder_y, right_shoulder_y))

            average_left_shoulder_y = sum(pos[0] for pos in shoulder_positions) / num_frames_for_average
            average_right_shoulder_y = sum(pos[1] for pos in shoulder_positions) / num_frames_for_average

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
            left_knee_angle = calculate_angle_3D(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle_3D(right_hip, right_knee, right_ankle)
            
            left_knee_angles.append(left_knee_angle)
            right_knee_angles.append(right_knee_angle)

            # Calculate moving average of knee angles
            average_left_knee_angle = sum(left_knee_angles) / num_frames_for_average
            average_right_knee_angle = sum(right_knee_angles) / num_frames_for_average


            # Calculate hip angles
            # left_hip_angle = calculate_angle_3D(left_shoulder, left_hip, left_knee)
            # right_hip_angle = calculate_angle_3D(right_shoulder, right_hip, right_knee)

            # Display the direction text on the frame
            cycle_x = 50
            cycle_y = 100
            text_to_display = f"{direction_text} | Cycles: {count}"
            draw_text(frame, (cycle_x, cycle_y), text_to_display)

            knee_x = 50
            knee_y = 200
            knee_text = f"Left Knee: {average_left_knee_angle:.2f} degrees | Right Knee: {average_right_knee_angle:.2f} degrees"
            draw_text(frame, (knee_x, knee_y), knee_text)

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
