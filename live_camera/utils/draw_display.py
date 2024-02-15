import cv2


def draw_text(frame, position, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, color=(0, 255, 0), thickness=2):
    """
    Draws text on a frame

    """
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_leg_landmarks(mp, frame, results, color=(0, 0, 255)):
    """
    Draws both leg landmarks on a frame
    """
    left_hip = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
    left_knee = mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
    left_ankle = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value

    # Left leg
    mp.solutions.drawing_utils.draw_landmarks(frame,
                                              results.pose_landmarks,
                                              [(left_hip, left_knee),
                                               (left_knee, left_ankle)],
                                              mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
                                                                                     thickness=15,
                                                                                     circle_radius=5),
                                              mp.solutions.drawing_utils.DrawingSpec(color=color,
                                                                                     thickness=15,
                                                                                     circle_radius=5)
                                              )

    right_hip = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
    right_knee = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
    right_ankle = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value

    # Right leg
    mp.solutions.drawing_utils.draw_landmarks(frame,
                                              results.pose_landmarks,
                                              [(right_hip, right_knee),
                                               (right_knee, right_ankle)],
                                              mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
                                                                                     thickness=15,
                                                                                     circle_radius=5),
                                              mp.solutions.drawing_utils.DrawingSpec(color=color,
                                                                                     thickness=15,
                                                                                     circle_radius=5)
                                              )
