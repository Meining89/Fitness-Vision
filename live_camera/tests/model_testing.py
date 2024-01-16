import mediapipe as mp
from keras.models import load_model
import cv2
import numpy as np
# Define the class labels
class_labels = ['Bad_head', 'Bad_back_round', 'Bad_back_warp', 'Bad_lifted_heels', 'Bad_inward_knee', 'Bad_shallow', 'Good']


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

def create_model():
    folder = '/Users/jaqueline/Capstone/Fitness-Vision/live_camera/models/LSTM_model_0.0005'

    AttnLSTM = load_model(folder)
    print(AttnLSTM.summary())
    
    return AttnLSTM

def feature_extraction_data(video_path, width=1920, height=1080):
  mp_pose = mp.solutions.pose
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #initialize lists to store frame features and images.
    keypoint_list = []
    video_reader = cv2.VideoCapture(video_path)

    #get estimation of number of frames
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    SEQUENCE_LENGTH=30

    frame_indices = [video_frames_count*i// SEQUENCE_LENGTH for i in range(SEQUENCE_LENGTH)]

    for current_frame in frame_indices:

        # Set current frame to be the specific frame
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        #Read the specific frame
        #check if the reading is successful
        success, frame = video_reader.read()

        if not success:
            return keypoint_list

        # resize video frame if too large
        vid_height, vid_width, channels = frame.shape
        if vid_height + vid_width > 3000:
          cv2.resize(frame, (width, height))

        # Make detection
        image, results = mediapipe_detection(frame, pose)
        #extract keypoints
        keypoints = extract_keypoints(results)
        #Add the frame and image to the list
        keypoint_list.append(keypoints)

    video_reader.release()
    return keypoint_list
  
def make_prediction(video_path,model):
    keypoints = feature_extraction_data(video_path)

    X = np.array([keypoints])
    # Make predictions
    predictions = model.predict(X)[0]  # Extract the first element of the predictions array

    # Correspond prediction probabilities with classes
    prediction_dict = {class_label: prob for class_label, prob in zip(class_labels, predictions)}

    # Print and display the results
    print("Predictions:")
    for class_label, prob in prediction_dict.items():
        print(f"{class_label}: {prob:.4f}")
   

def main():
    # Create LSTM model
    AttnLSTM = create_model()
    
    external_bad_video_path = 'external_bad_video_test.mp4'
    external_good_video_path = 'external_good_video_test.mp4'
    
    make_prediction(external_good_video_path,AttnLSTM)
    make_prediction(external_bad_video_path, AttnLSTM )




if __name__ == "__main__":
    main()

