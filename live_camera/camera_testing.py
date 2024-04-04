import cv2

# Open the video capture for the desired camera
# Change the index to 0, 1, 2, etc., based on which camera you want to use
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# cv2.imshow('Frame', frame)
# Get the frame rate of the camera
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the current resolution of the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Frame Rate: {fps} FPS")
print(f"Camera Resolution: {frame_width}x{frame_height}")

# Cleanup
cap.release()
