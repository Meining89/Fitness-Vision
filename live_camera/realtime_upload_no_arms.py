import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from utils.angles import *
from utils.draw_display import *
from exercise.squat import *
from collections import deque
from utils.mediapipe_helper import * 
import tempfile
from keras.models import Model, load_model
from keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
from collections import deque
import av
import pandas as pd
class_labels = ['Bad Head', 'Bad Back', 'Bad Frontal Knees', 'Bad Inward Knee', 'Bad Shallow','Good']

st.set_page_config(layout="wide")
# Add custom CSS for styling
st.markdown("""
<style>
.styled-box {
    background-color: #fafafa;
    border-left: 5px solid #4A90E2;
    padding: 2rem;
    margin: 1rem 0rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.styled-header {
    color: #4A90E2;
    font-weight: 700;
}

.styled-text {
    color: #333333;
    padding-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't already exist
# This is to prevent processing uploaded videos multiple times
if 'last_processed' not in st.session_state:
    st.session_state['last_processed'] = None
if 'prediction_dict' not in st.session_state:
    st.session_state['prediction_dict'] = None

st.title("Welcome to Fitness Vision!")

st.write("\n")


st.markdown("""
    <div class="styled-box">
        <h3>Instructions</h3>
        <ul>
            <li>To get started, please make sure your entire body is visible in the frame, from shoulders to feet.</li>
            <li>Please stand 1m to 4m away from the camera and 45 degrees to the camera.</li>
            <li>Sufficient lighting is recommended.</li>
            <li>Adjust the baseline settings to tailor the experience to your needs. Use the sliders below to fine-tune the detection to your liking.</li>
            <li>Enjoy exploring the cool features of our app!</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


st.write("\n")
st.markdown("---")

st.markdown("### ✨ Personalize Your Settings", unsafe_allow_html=True)
threshold1 = st.slider("Keypoint Detection Confidence", 0.00, 1.00, 0.50, help="Adjust the sensitivity for mediapipe keypoint detection to ensure accurate pose detection.")
threshold2 = st.slider("Tracking Confidence", 0.00, 1.00, 0.50, help="Set the stability level for consistent tracking throughout your workout.")
KNEE_ANGLE_DEPTH = st.slider("Knee Angle for Sufficient Depth", 75, 115, 95, help="Select the perfect knee angle to hit the right depth for your squats.")
professional_mode = st.toggle("Enable Depth - Professional Mode", value=False, help="Toggle this switch to Depth - Professional Mode, where a squat is counted only if the knee angle <= threshold.")
if professional_mode:
    st.markdown("Depth - Professional Mode is enabled: a squat is counted only if the knee angle is less than or equal to the threshold. Live camera analysis only.")

@st.cache_resource
def create_model():
   
    folder = 'models/no_arm_our_0.001'

    AttnLSTM = load_model(folder)
    print(AttnLSTM.summary())
    
    return AttnLSTM

# Create LSTM model
AttnLSTM = create_model()       

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2) # mediapipe pose model

def feature_extraction_data(uploaded_file, width=1920, height=1080):
 
    # Read the uploaded video file into a byte stream and then into OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.getvalue())
    
    # Use the temporary file's name to read it into OpenCV
    video_stream = cv2.VideoCapture(tfile.name)
    #initialize lists to store frame features and images.
    keypoint_list = []

    #get estimation of number of frames
    video_frames_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    SEQUENCE_LENGTH=30

    frame_indices = [video_frames_count*i// SEQUENCE_LENGTH for i in range(SEQUENCE_LENGTH)]

    for current_frame in frame_indices:

        # Set current frame to be the specific frame
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        #Read the specific frame
        #check if the reading is successful
        success, frame = video_stream.read()

        if not success:
            return keypoint_list

        # resize video frame if too large
        vid_height, vid_width, channels = frame.shape
        if vid_height + vid_width > 3000:
          cv2.resize(frame, (width, height))

        # Make detection
        image, results = mediapipe_detection(frame, pose)
        #extract keypoints
        keypoints = extract_keypoints_no_arm(results)
        #Add the frame and image to the list
        keypoint_list.append(keypoints)

    video_stream.release()
    tfile.close()
    return keypoint_list
  
def make_prediction(uploaded_file, model):
    keypoints = feature_extraction_data(uploaded_file)

    X = np.array([keypoints])
    # Make predictions
    predictions = model.predict(X)[0]  # Extract the first element of the predictions array

    # Correspond prediction probabilities with classes
    prediction_dict = {class_label: prob for class_label, prob in zip(class_labels, predictions)}

    return prediction_dict

def display_predictions(prediction_dict):
    # Convert the prediction dictionary to a pandas DataFrame for nicer display
    predictions_df = pd.DataFrame(list(prediction_dict.items()), columns=['Class', 'Probability'])
    
    # Sort the DataFrame by probability in descending order
    predictions_df.sort_values(by='Probability', ascending=False, inplace=True)
    
    # Format the probabilities to be more readable (e.g., as percentages)
    predictions_df['Probability'] = predictions_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
    
    first_row_class = predictions_df.iloc[0]['Class']
    if first_row_class == 'Good':
        highlight_color = "#90ee90"  # Light green
    else:
        highlight_color = "#ffcccb"  # Light red
    
    # Convert DataFrame to HTML, hide the index, and highlight the first row based on the class
    predictions_html = predictions_df.to_html(index=False, escape=False)
    if predictions_html:
        # Add style for highlighting the first row based on the class
        predictions_html = predictions_html.replace('<tr>', f'<tr style="background-color: {highlight_color}">', 1)
    
    
    # Display the HTML using Streamlit, without the index
    st.write("#### Prediction Results:")
    st.markdown(predictions_html, unsafe_allow_html=True)

st.write("\n")  # Add some space
st.markdown("---")  # Visual separator, like a horizontal line

# Uploaded video processing section
st.write("### 🎥 Analyze Your Uploaded Video ")
uploaded_file = st.file_uploader("Upload one repetition of squat to analyze", type=["mp4"])
if uploaded_file is not None and uploaded_file != st.session_state.get('last_processed', None):
    with st.spinner('Processing...'):
        prediction_dict = make_prediction(uploaded_file, AttnLSTM)
        st.session_state['last_processed'] = uploaded_file  # Update session state
        st.session_state['prediction_dict'] = prediction_dict
        # Display the prediction results
        display_predictions(prediction_dict)
elif uploaded_file is None:
    st.write("Please upload a video file for analysis.")
else:
    # If the same file is still there, just show the last predictions without reprocessing
    #st.write("Showing results from the last processed video:")
    if 'prediction_dict' in st.session_state:
        display_predictions(st.session_state['prediction_dict'])
        

class VideoProcessor :
    def __init__(self):
        #Initilize parameters and variables
        self.sequence_length = 30
        self.actions = ['Bad Head', 'Bad Back', 'Bad Frontal Knee', 'Bad Inward Knee', 'Bad Shallow','Good']
        self.sequence = deque(maxlen=self.sequence_length)
        self.sequence_saved = False
        self.frame_count = 0
        self.frame_sequence = []

        self.prediction_history = deque(maxlen=5)
        self.colors = [
            (245, 117, 16),  # Orange
            (117, 245, 16),  # Lime Green
            (16, 117, 245),  # Royal Blue
            (255, 0, 0),     # Red
            (0, 0, 255),     # Blue
            (0, 255, 0),  # Green
            (255, 255, 0)    # Yellow

        ]
        # Initialize shoulder Y positions
        self.shoulder_positions = deque(maxlen=NUM_FRAMES_SHOULDER)
        self.left_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)
        self.right_knee_angles = deque(maxlen=NUM_FRAMES_KNEE)

        # Initalize counter
        self.count = 0
        self.going_up = False
        self.increment = False

        self.direction_text = "STABLE"
        self.check=True

    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of classes
        as a horizontal bar graph

        """
        output_frame = input_frame.copy()
        font_size = 1.5
        for num, prob in enumerate(res):
            # change prob * ___ for longer length
            cv2.rectangle(output_frame, (0, 70 + num * 50), (int(1 * 450), 130 + num * 50), (0, 0, 0),
                          -1)  # black background

            cv2.rectangle(output_frame, (0, 70 + num * 50), (int(prob * 450), 130 + num * 50), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0,115 + num * 50), cv2.FONT_HERSHEY_SIMPLEX, font_size,
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
        keypoints = extract_keypoints_no_arm(results, self.check)
        if self.check:
            self.check = False
        moving_average = np.zeros(len(self.actions))
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        

        if len(self.sequence) == self.sequence_length:
            
            if not self.sequence_saved:
                # Save sequence when its length reaches 30
                np.save('meidapipe_sequence.npy', np.array(self.sequence))
                self.sequence_saved = True

            res = model.predict(np.expand_dims(list(self.sequence), axis=0), verbose=0)[0]
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

            if self.check:
                for landmark in mp.solutions.pose.PoseLandmark:
                    print(landmark.name, landmark.value)

                print("---------------------------------")
                self.check = False


            # Save only the first 30 frames to numpy files
            if self.frame_count <= 30:
                # keypoints = extract_keypoints_no_arm(results)
                self.frame_sequence.append(rgb_frame)

                if self.frame_count == 30:
                    np.save('frame_sequence.npy', np.array(self.frame_sequence))

            self.frame_count += 1

            test_landmark = []

            # for landmark in mp.solutions.pose.PoseLandmark:
            #     if 13 <= landmark.value <= 22:
            #         continue
            #     test_landmark.append(landmark)

            for idx in range(33):
                if 13 <= idx <= 22:
                    continue
                landmark = results.pose_landmarks.landmark[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # mp.solutions.drawing_utils.draw_landmarks(frame,
            #                                         # results.pose_landmarks,
            #                                           test_landmark,
            #                                         mp_pose.POSE_CONNECTIONS,
            #                                         mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66),
            #                                                                                 thickness=10,
            #                                                                                 circle_radius=5),
            #                                         mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255),
            #                                                                                 thickness=10,
            #                                                                                 circle_radius=5)
            #                                         )

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
            # draw_leg_landmarks(mp, frame, results, color=(0, 255, 0) if knee_angle <= KNEE_ANGLE_DEPTH else (0, 0, 255))

            # Compare with previous Y positions to determine movement direction
            if is_standing_up_old(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                            left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                self.direction_text = "UP"
                # Change in direction: going up now
                if not self.going_up and self.increment:
                    self.count += 1
                    self.going_up = True
                    self.increment = False
            elif is_squatting_down_old(left_shoulder_y, right_shoulder_y, average_left_shoulder_y, average_right_shoulder_y,
                                left_knee_angle, right_knee_angle, average_left_knee_angle, average_right_knee_angle):
                self.direction_text = "DOWN"
                self.going_up = False

                # Professional mode: only count if knee angle is less than or equal to threshold
                if professional_mode:
                    if knee_angle <= KNEE_ANGLE_DEPTH:
                        self.increment = True
                else:
                    # Non-professional mode: count every time the user stands up
                    self.increment = True

                if knee_angle > KNEE_ANGLE_DEPTH:
                    text_to_display = "Go lower!"
                    draw_text(frame, (knee_loc[0], knee_loc[1] + knee_text_height + 40), text_to_display, font_scale=2,
                            color=(0, 0, 255))
            else:
                self.direction_text = "STABLE"

            # Display the direction text on the frame
            cycle_x = 0
            cycle_y = 50
            text_to_display = f"{self.direction_text} | Count: {self.count}"
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
        else:
            frame = self.prob_viz(np.zeros(len(self.actions)), frame)

        # Display the direction text on the frame
        cycle_x = 0
        cycle_y = 50
        text_to_display = f"{self.direction_text} | Cycles: {self.count}"
        draw_text(frame, (cycle_x, cycle_y), text_to_display, color=(255, 255, 255))

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
st.write("\n")
st.markdown("---")

with st.container():
    st.write("### 🏋️‍♂️ Activate the AI Real-time Detection ")
# Options
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# Streamer
webrtc_ctx = webrtc_streamer(
    key="AI trainer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

