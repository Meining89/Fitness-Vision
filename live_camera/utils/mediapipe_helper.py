import cv2
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
import numpy as np
import math


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)  # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    # extract keypoints and convert to np array
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose


def extract_keypoints_no_arm(results, run_check):
    # Extract keypoints and convert to np array
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        # Remove landmarks 13-22 (indices 12 to 22, inclusive)
        # Since landmarks are 0-indexed, landmark 13 starts at index 12*4=48 and landmark 22 ends at index 22*4=88
        pose = np.concatenate((pose[:13], pose[23:]), axis=0)
    else:
        pose = np.zeros((33 - 10) * 4)  # Adjusted for the removal of 10 landmarks
    
    return pose.flatten()

def feature_extraction_data(mp_pose, frame_list, width=1920, height=1080):
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #initialize lists to store frame features and images.
    keypoint_list = []

    #get estimation of number of frames
    video_frames_count = len(frame_list)

    SEQUENCE_LENGTH = 30

    frame_indices = [video_frames_count*i // SEQUENCE_LENGTH for i in range(SEQUENCE_LENGTH)]

    for current_frame in frame_indices:
        keypoints = frame_list[current_frame]
        #Add the frame and image to the list
        keypoint_list.append(keypoints)

    return keypoint_list

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist




def get_coordinates(landmarks, mp_pose, side, joint):
    """
    Retrieves x and y coordinates of a particular keypoint from the pose estimation model

     Args:
         landmarks: processed keypoints from the pose estimation model
         mp_pose: Mediapipe pose estimation model
         side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
         joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.

    """
    coord = getattr(mp_pose.PoseLandmark, side.upper() + "_" + joint.upper())
    x_coord_val = landmarks[coord.value].x
    y_coord_val = landmarks[coord.value].y
    return [x_coord_val, y_coord_val]

