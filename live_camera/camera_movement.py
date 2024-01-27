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
    num_input_values = num_landmarks * num_values
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
    x = Dense(2 * HIDDEN_UNITS, activation='relu')(attention_mul)
    # Dropput Layer to avoid overfitting. 50% of the units in the fully connected layer are dropped out during training.
    x = Dropout(0.5)(x)

    # Output Layer
    x = Dense(num_classes, activation='softmax')(x)

    # Bring it all together
    # AttnLSTM = Model(inputs=[inputs], outputs=x)

    folder = 'models/meg_owndata'

    AttnLSTM = load_model(folder)
    print(AttnLSTM.summary())

    return AttnLSTM


class VideoProcessor:
    def __init__(self):
        # Initilize parameters and variables
        self.sequence_length = 30
        self.actions = ['Bad Head', 'Bad Back', 'Bad Lifted Heels', 'Bad Inward Knee', 'Good']
        self.sequence = deque(maxlen=self.sequence_length)

        self.prediction_history = deque(maxlen=5)
        self.counter = 0
        self.colors = [
            (245, 117, 16),  # Orange
            (117, 245, 16),  # Lime Green
            (16, 117, 245),  # Royal Blue
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 0)  # Green
        ]

    #  self.threshold =

    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        font_size = 2
        for num, prob in enumerate(res):
            # change prob * ___ for longer length
            cv2.rectangle(output_frame, (0, 100 + num * 60), (int(1 * 550), 150 + num * 60), (0, 0, 0),
                          -1)  # black background

            cv2.rectangle(output_frame, (0, 100 + num * 60), (int(prob * 550), 150 + num * 60), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 145 + num * 60), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (255, 255, 255), 2, cv2.LINE_AA)

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
        moving_average = np.zeros(len(self.actions))
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))

        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(list(self.sequence), axis=0), verbose=0)[0]
            # self.current_action = self.actions[np.argmax(res)]
            self.prediction_history.append(res)

            if len(self.prediction_history) == self.prediction_history.maxlen:
                moving_average = np.mean(self.prediction_history, axis=0)
                self.current_action = self.actions[np.argmax(moving_average)]

            # Viz probabilities
            image = self.prob_viz(moving_average, image)

        return image


class LandmarkData:
    def __init__(self, landmark_left, landmark_right, max_frames):
        self.landmark_left = landmark_left
        self.landmark_right = landmark_right

        # tracks x value (position[0] = left, position[1] = right)
        self.x_queue = deque(maxlen=max_frames)
        # tracks y value (position[0] = left, position[1] = right)
        self.y_queue = deque(maxlen=max_frames)

        # left and right contain both (x value, y value)
        self.left = None
        self.right = None
        # average of y values
        self.avg_val_left = None
        self.avg_val_right = None

        self.angles = deque(maxlen=max_frames)
        self.left_angle = None
        self.right_angle = None
        self.avg_angle_left = None
        self.avg_angle_right = None

    def update_values(self, results, update_y=True):
        self.left = results.pose_landmarks.landmark[self.landmark_left]
        self.right = results.pose_landmarks.landmark[self.landmark_right]
        if update_y:
            self.y_queue.append((self.left.y, self.right.y))
        else:
            self.x_queue.append((self.left.x, self.right.x))

        self.update_average(update_y=update_y)

    def update_average(self, update_y=True):
        queue = self.y_queue if update_y else self.x_queue

        self.avg_val_left, self.avg_val_right = map(lambda x: sum(x) / len(x), zip(*queue))


    def update_angles(self, left_angle, right_angle):
        self.left_angle = left_angle
        self.right_angle = right_angle
        self.angles.append((self.left_angle, self.right_angle))

        self.avg_angle_left, self.avg_angle_right = map(lambda x: sum(x) / len(x), zip(*self.angles))


class Counter:
    def __init__(self):
        self.count = 0
        self.direction_text = "STABLE"
        self.going_up = False

    def update_counter(self, shoulder_obj, knee_obj):
        # Compare with previous Y positions to determine movement direction
        if is_standing_up(shoulder_obj, knee_obj):
            self.direction_text = "UP"
            # Change in direction: going up now
            if not self.going_up:
                self.count += 1
                self.going_up = True

        elif is_squatting_down(shoulder_obj, knee_obj):
            self.direction_text = "DOWN"
            self.going_up = False

        else:
            self.direction_text = "STABLE"


def main():
    # Create LSTM model
    AttnLSTM = create_model()
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (change it if you have multiple cameras)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS value", fps)

    # Initialize Video Processor
    video_processor = VideoProcessor()

    # Initialize landmark data for shoulder
    shoulder_obj = LandmarkData(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, NUM_FRAMES_SHOULDER)
    knee_obj = LandmarkData(mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, NUM_FRAMES_KNEE)

    # Initialize counter
    counter_obj = Counter()

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

            # Update landmark data
            shoulder_obj.update_values(results)
            knee_obj.update_values(results)

            # update knee angles
            left_knee_angle, right_knee_angle = calculate_knee_angles(results, mp_pose)
            knee_obj.update_angles(left_knee_angle, right_knee_angle)

            # update counter - MUST update landark data first
            counter_obj.update_counter(shoulder_obj, knee_obj)

            knee_angle = min(knee_obj.left_angle, knee_obj.right_angle)
            # Draw the left leg in red if the knee angle is greater than the threshold
            draw_leg_landmarks(mp, frame, results, color=(0, 255, 0) if knee_angle < KNEE_ANGLE_DEPTH else (0, 0, 255))

            ################### ERROR CHECKING ###################

            process_shallow(frame, counter_obj, knee_obj)

            ######################################################

            cycle_x = 0
            cycle_y = 50
            text_to_display = f"{counter_obj.direction_text} | Cycles: {counter_obj.count}"
            draw_text(frame, (cycle_x, cycle_y), text_to_display, color=(255, 255, 255))

            # Process the frame with AttnLSTM model
            frame = video_processor.inference_process(AttnLSTM, frame, results)
            cv2.imshow('Classification', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
