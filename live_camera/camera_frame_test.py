import cv2
import time

# Open the video capture for the desired camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get the resolution of the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution: {frame_width}x{frame_height}")

# Initialize variables for calculating the actual frame rate
num_frames = 60
start_time = time.time()

# Start capturing and displaying frames
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate and display the actual FPS every 'num_frames' frames
    if i == num_frames - 1:
        end_time = time.time()
        time_elapsed = end_time - start_time
        fps = num_frames / time_elapsed
        print(f"Actual FPS: {fps:.2f}")
        # Reset the frame counter and the start time
        num_frames = 0
        start_time = time.time()

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
