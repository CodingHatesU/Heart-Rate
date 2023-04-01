import cv2
import dlib
import numpy as np
import time

# Define a function to detect the person's face and extract the forehead or cheek ROI
def get_roi(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        roi = frame[y:y+h, x:x+w]
        return roi
    return None

# Define a function to estimate the person's heart rate from the ROI
def estimate_heart_rate(roi, fps):
    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to segment skin regions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Compute the mean intensity in each frame
    intensity = np.mean(thresh)

    # Compute the heart rate from the intensity signal
    heart_rate = intensity / fps * 10

    return heart_rate

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Start the video stream
cap = cv2.VideoCapture(0)

# Get the frame rate of the video stream
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Start a timer
start_time = time.time()

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Get the ROI around the person's forehead or cheek
    roi = get_roi(frame, detector)

    if roi is not None:
        # Estimate the person's heart rate from the ROI
        heart_rate = estimate_heart_rate(roi, fps)

        # Display the estimated heart rate on the video frame
        cv2.putText(frame, f"Heart Rate: {heart_rate:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the video frame
    cv2.imshow("Heart Rate Estimation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print the heart rate every 10 seconds
    if time.time() - start_time > 10:
        print(f"Heart Rate: {heart_rate:.0f} bpm")
        start_time = time.time()

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()