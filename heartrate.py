import cv2
import numpy as np
import time

def get_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    rects = []
    for (x, y, w, h) in faces:
        rects.append((x, y, w, h))
    for rect in rects:
        x, y, w, h = rect
        roi = frame[y:y+h, x:x+w]
        return roi
    return None

def estimate_heart_rate(roi, fps):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    intensity = np.mean(thresh)
    heart_rate = intensity / fps * 10
    return heart_rate

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
start_time = time.time()

while True:
    ret, frame = cap.read()
    roi = get_roi(frame)
    if roi is not None:
        heart_rate = estimate_heart_rate(roi, fps)
        cv2.putText(frame, f"Heart Rate: {heart_rate:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Heart Rate Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time > 10:
        print(f"Heart Rate: {heart_rate:.0f} bpm")
        start_time = time.time()
cap.release()
cv2.destroyAllWindows()