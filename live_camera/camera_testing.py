import cv2

# Open the video capture for the desired camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get the frame rate of the camera
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the current resolution of the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Frame Rate: {fps} FPS")
print(f"Camera Resolution: {frame_width}x{frame_height}")

# Start capturing and displaying frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
