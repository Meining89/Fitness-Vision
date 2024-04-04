import cv2
import time

# Open the video capture for the second camera
cap = cv2.VideoCapture(1)  # 1 should be the index for the second camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Define the resolutions to test
resolutions = [(640, 480), (1280, 720), (1920, 1080)]

for resolution in resolutions:
    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Print current resolution
    print(f"Testing resolution: {resolution[0]}x{resolution[1]}")
    
    num_frames_to_test = 60
    print(f"Capturing {num_frames_to_test} frames")

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(num_frames_to_test):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the resulting frame (optional)
        cv2.imshow('Frame', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    fps = num_frames_to_test / seconds
    print(f"Estimated frames per second at resolution {resolution[0]}x{resolution[1]}: {fps}")

# Release everything
cap.release()
cv2.destroyAllWindows()
