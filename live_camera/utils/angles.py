import numpy as np


def calculate_angle_3d_1(a, b, c):
    """
    Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another

    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    return rad_angle * 180 / np.pi


def calculate_angle_3d_2(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    ba = a - b
    bc = c - b

    # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # angle = np.arccos(cosine_angle)

    angle = angle_between(ba, bc)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_3d(a, b, c):
    vector_ab = np.array(a) - np.array(b)
    vector_bc = np.array(c) - np.array(b)

    dot_product = np.dot(vector_ab, vector_bc)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_bc = np.linalg.norm(vector_bc)

    cos_theta = dot_product / (magnitude_ab * magnitude_bc)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def calculate_knee_angles(results, mp_pose):
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

    return left_knee_angle, right_knee_angle
