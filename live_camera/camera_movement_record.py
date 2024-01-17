import mediapipe as mp
from utils.angles import *
from utils.draw_display import *
from exercise.squat import *
from collections import deque
from utils.mediapipe_helper import * 

from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
from collections import deque

## Create and Load the Model
def attention_block(inputs, time_steps):
    """
    Attention layer for deep neural network

    """
    # Swap dimension to prepare for input for dense layer
    a = Permute((2, 1))(inputs)
    # Compute attention weights, use softmax to make sure it sums up to 1
    a = Dense(time_steps, activation='softmax')(a)

    # Attention vector
    a_probs = Permute((2, 1), name='attention_vec')(a)

    # Luong's multiplicative score
    # Performs an element-wise multiplication between the input sequence (inputs) and the attention vector (a_probs).
    # This multiplication emphasizes the elements of the input sequence that have higher attention weights.
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')

    return output_attention_mul

def create_model():
    """
    create and load LSTM Model with attention mechinism

    """

    HIDDEN_UNITS = 256
    sequence_length = 30
    num_landmarks = 33
    num_values = 4
    num_input_values = num_landmarks*num_values
    num_classes = 7

    # Input
    inputs = Input(shape=(sequence_length, num_input_values))

    # Bi-LSTM
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)

    # Attention Block
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)

    # Fully Connected Layer
    # Common Practice to double number of hidden units in the fully connected layer compared to the LSTM layer.
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    #Dropput Layer to avoid overfitting. 50% of the units in the fully connected layer are dropped out during training.
    x = Dropout(0.5)(x)

    # Output Layer
    x = Dense(num_classes, activation='softmax')(x)

    # Bring it all together
    # AttnLSTM = Model(inputs=[inputs], outputs=x)

    folder = 'models/LSTM_model_0.0005'

    AttnLSTM = load_model(folder)
    print(AttnLSTM.summary())
    
    return AttnLSTM


class VideoProcessor:
    def __init__(self):
        #Initilize parameters and variables
        self.sequence_length = 30
        self.actions = ['Bad_head', 'Bad_back_round', 'Bad_back_warp', 'Bad_lifted_heels', 'Bad_inward_knee', 'Bad_shallow','Good']
        self.sequence = deque(maxlen=self.sequence_length)

        self.frame_history = []
        self.counter = 0
        self.colors = [
            (245, 117, 16),  # Orange
            (117, 245, 16),  # Lime Green
            (16, 117, 245),  # Royal Blue
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0)    # Yellow
        ]
      #  self.threshold = 

    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def inference_process(self, model, image, results):
        """
        Function to process and run inference on AttnLSTM with real time video frame input

        Args:
            model: the AttnLSTM classification model
            image (numpy array): input image from the webcam
            results: Processed frame from mediapipe Pose

        Returns:
            numpy array: processed image with keypoint detection and classification
        """

        # Prediction logic
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))

        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(list(self.sequence), axis=0), verbose=0)[0]
            self.current_action = self.actions[np.argmax(res)]

            # Viz probabilities
            image = self.prob_viz(res, image)

        return image



def main():
    # Create LSTM model
    AttnLSTM = create_model()
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize Video Processor
    video_processor = VideoProcessor()
    # Initialize shoulder Y positions
    shoulder_positions = deque(maxlen=NUM_FRAMES_SHOULDER)
    left_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)
    right_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)

    # Initalize counter
    count = 0
    going_up = False
    start_recording = False

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (change it if you have multiple cameras)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print ("FPS value",fps)

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
            # Process the frame with AttnLSTM model
            # img = video_processor.inference_process(AttnLSTM, rgb_frame, results)
            frame = rgb_frame
            prediction = [0] * 7

            video_processor.frame_history.append(extract_keypoints(results))

            if len(video_processor.frame_history) > 100:
                print("Clearing frame history")
                video_processor.frame_history = video_processor.frame_history[-video_processor.sequence_length:]

            # Viz probabilities
            frame = video_processor.prob_viz(prediction, frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

                if len(video_processor.frame_history) >= 30:
                    # sample 30 frames
                    model_input = feature_extraction_data(mp_pose, video_processor.frame_history)
                    print(len(model_input))
                    # make prediction
                    prediction = AttnLSTM.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
                    video_processor.current_action = video_processor.actions[np.argmax(prediction)]

                    print("Prediction: ", prediction)
                    print("Max prediction", video_processor.current_action)

                    # # Viz probabilities
                    # img = video_processor.prob_viz(prediction, img)
                    frame = video_processor.prob_viz(prediction, frame)

                # clear frame history
                video_processor.frame_history = []

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
            cv2.imshow('Classification', frame)

            # # Display the resulting frame
            # cv2.imshow('Pose Estimation', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
