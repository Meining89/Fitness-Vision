import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from utils.angles import *
from utils.draw_display import *
from exercise.squat import *
from collections import deque
from utils.mediapipe_helper import * 

from keras.models import Model, load_model
from keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
from collections import deque
import av

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

## Stream Webcam Video and Run Model
# Create LSTM model
AttnLSTM = create_model()
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

## App
st.write("# AI Personal Fitness Trainer Web App")

st.markdown("❗❗ **Development Note** ❗❗")
st.markdown("Currently, the exercise recognition model uses the the x, y, and z coordinates of each anatomical landmark from the MediaPipe Pose model. These coordinates are normalized with respect to the image frame (e.g., the top left corner represents (x=0,y=0) and the bottom right corner represents(x=1,y=1)).")
st.markdown("I'm currently developing and testing two new feature engineering strategies:")
st.markdown("- Normalizing coordinates by the detected bounding box of the user")
st.markdown("- Using joint angles rather than keypoint coordaintes as features")            
st.write("Stay Tuned!")

st.write("## Settings")
threshold1 = st.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)

st.write("## Activate the AI 🤖🏋️‍♂️")

class VideoProcessor :
    def __init__(self):
        #Initilize parameters and variables
        self.sequence_length = 30
        self.actions = ['Bad Head', 'Bad Back Round', 'Bad Back Warp', 'Bad Lifted Heels', 'Bad Inward Knee', 'Bad_Shallow','Good']
        self.sequence = deque(maxlen=self.sequence_length)

        self.prediction_history = deque(maxlen=5)
        self.colors = [
            (245, 117, 16),  # Orange
            (117, 245, 16),  # Lime Green
            (16, 117, 245),  # Royal Blue
            (255, 0, 0),     # Red
            (0, 0, 255),     # Blue
            (255, 255, 0),    # Yellow
            (0, 255, 0)  # Green
        ]
        # Initialize shoulder Y positions
        self.shoulder_positions = deque(maxlen=NUM_FRAMES_SHOULDER)
        self.left_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)
        self.right_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)

        # Initalize counter
        self.count = 0
        self.going_up = False

        self.direction_text = "STABLE"

    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        font_size = 2
        for num, prob in enumerate(res):
            # change prob * ___ for longer length
            cv2.rectangle(output_frame, (0, 100+num*60), (int(1*550), 150+num*60), (0,0,0), -1) # black background

            cv2.rectangle(output_frame, (0, 100+num*60), (int(prob*550), 150+num*60), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 145+num*60), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2, cv2.LINE_AA)
            
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
            print('start predicting...')
            # self.current_action = self.actions[np.argmax(res)]
            self.prediction_history.append(res)

            if len(self.prediction_history) == self.prediction_history.maxlen:
                moving_average = np.mean(self.prediction_history, axis=0)
                self.current_action = self.actions[np.argmax(moving_average)]

            # Viz probabilities
            image = self.prob_viz(moving_average, image)

        return image
    
    def process(self, frame):
        knee_text_height = 10

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
            self.shoulder_positions.append((left_shoulder_y, right_shoulder_y))

            average_left_shoulder_y = sum(pos[0] for pos in self.shoulder_positions) / len(self.shoulder_positions)
            average_right_shoulder_y = sum(pos[1] for pos in self.shoulder_positions) / len(self.shoulder_positions)

            ###################### Calculate knee angles ######################
            left_knee_angle, right_knee_angle = calculate_knee_angles(results, mp_pose)

            self.left_knee_angles.append(left_knee_angle)
            self.right_knee_angles.append(right_knee_angle)

            # Calculate moving average of knee angles
            average_left_knee_angle = sum(self.left_knee_angles) / len(self.left_knee_angles)
            average_right_knee_angle = sum(self.right_knee_angles) / len(self.right_knee_angles)

            left_knee_pixel_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * frame_width)
            left_knee_pixel_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * frame_height)
            knee_loc = (left_knee_pixel_x + 10, left_knee_pixel_y)
            knee_angle = min(left_knee_angle, right_knee_angle)

            # Draw the left leg in red if the knee angle is greater than the threshold
            draw_leg_landmarks(mp, frame, results, color=(0, 255, 0) if knee_angle < KNEE_ANGLE_DEPTH else (0, 0, 255))

            # Compare with previous Y positions to determine movement direction
            if is_standing_up_old(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                            left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                self.direction_text = "UP"
                # Change in direction: going up now
                if not self.going_up:
                    self.count += 1
                    self.going_up = True

            elif is_squatting_down_old(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                                left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                self.direction_text = "DOWN"
                self.going_up = False

                if knee_angle > KNEE_ANGLE_DEPTH:
                    text_to_display = "Go lower!"
                    draw_text(frame, (knee_loc[0], knee_loc[1] + knee_text_height + 20), text_to_display, font_scale=2,
                            color=(0, 0, 255))
            else:
                self.direction_text = "STABLE"

            # Display the direction text on the frame
            cycle_x = 0
            cycle_y = 50
            text_to_display = f"{self.direction_text} | Cycles: {self.count}"
            draw_text(frame, (cycle_x, cycle_y), text_to_display, color=(255, 255, 255))

            # knee_info_x = 50
            # knee_info_y = 200
            # knee_text = f"Left Knee: {average_left_knee_angle:.2f} degrees | Right Knee: {average_right_knee_angle:.2f} degrees"
            # # show per frame values
            # # knee_text = f"Left Knee: {left_knee_angle:.2f} degrees | Right Knee: {right_knee_angle:.2f} degrees"
            # draw_text(frame, (knee_info_x, knee_info_y), knee_text)

            knee_angle_text = f"{knee_angle:.2f} degrees"
            draw_text(frame, knee_loc, knee_angle_text)
            _, knee_text_height = cv2.getTextSize(knee_angle_text, cv2.FONT_HERSHEY_SIMPLEX, 2, thickness=2)[0]

            # Update previous Y positions
            prev_left_shoulder_y = left_shoulder_y
            prev_right_shoulder_y = right_shoulder_y

            # Process the frame with AttnLSTM model
            frame = self.inference_process(AttnLSTM, frame, results)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
            

    def recv(self, frame):
        """
        Receive and process video stream from webcam

        Args:
            frame: current video frame

        Returns:
            av.VideoFrame: processed video frame
        """
        img = frame.to_ndarray(format="bgr24")
        img = self.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        

## Stream Webcam Video and Run Model
# Options
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# Streamer
webrtc_ctx = webrtc_streamer(
    key="AI trainer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

