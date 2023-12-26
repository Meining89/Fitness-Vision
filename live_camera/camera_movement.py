import mediapipe as mp
from utils.angles import *
from utils.draw_display import *
from exercise.squat import *
from collections import deque


def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize shoulder Y positions
    shoulder_positions = deque(maxlen=NUM_FRAMES_SHOULDER)
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
            left_knee_angle, right_knee_angle = calculate_knee_angles(results, mp_pose)

            left_knee_angles.append(left_knee_angle)
            right_knee_angles.append(right_knee_angle)

            # Calculate moving average of knee angles
            average_left_knee_angle = sum(left_knee_angles) / len(left_knee_angles)
            average_right_knee_angle = sum(right_knee_angles) / len(right_knee_angles)

            left_knee_pixel_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * frame_width)
            left_knee_pixel_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * frame_height)
            knee_loc = (left_knee_pixel_x + 10, left_knee_pixel_y)
            knee_angle = min(left_knee_angle, right_knee_angle)

            # Draw the left leg in red if the knee angle is greater than the threshold
            draw_leg_landmarks(mp, frame, results, color=(0, 255, 0) if knee_angle < KNEE_ANGLE_DEPTH else (0, 0, 255))

            # Compare with previous Y positions to determine movement direction
            if is_standing_up(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                              left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                direction_text = "UP"
                # Change in direction: going up now
                if not going_up:
                    count += 1
                    going_up = True

            elif is_squatting_down(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                                   left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                direction_text = "DOWN"
                going_up = False

                if knee_angle > KNEE_ANGLE_DEPTH:
                    text_to_display = "Go lower!"
                    draw_text(frame, (knee_loc[0], knee_loc[1] + knee_text_height + 20), text_to_display, font_scale=2,
                              color=(0, 0, 255))
            else:
                direction_text = "STABLE"

            # Display the direction text on the frame
            cycle_x = 50
            cycle_y = 100
            text_to_display = f"{direction_text} | Cycles: {count}"
            draw_text(frame, (cycle_x, cycle_y), text_to_display)

            knee_info_x = 50
            knee_info_y = 200
            knee_text = f"Left Knee: {average_left_knee_angle:.2f} degrees | Right Knee: {average_right_knee_angle:.2f} degrees"
            # show per frame values
            # knee_text = f"Left Knee: {left_knee_angle:.2f} degrees | Right Knee: {right_knee_angle:.2f} degrees"
            draw_text(frame, (knee_info_x, knee_info_y), knee_text)

            knee_angle_text = f"{knee_angle:.2f} degrees"
            draw_text(frame, knee_loc, knee_angle_text)
            _, knee_text_height = cv2.getTextSize(knee_angle_text, cv2.FONT_HERSHEY_SIMPLEX, 2, thickness=2)[0]

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
