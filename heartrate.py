import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class HeartRateAndSpO2Estimation:
    def __init__(self, camera_id=0, show_video=True):
        self.show_video = show_video
        self.cap = cv2.VideoCapture(camera_id)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.heart_rates = []
        self.times = []
        self.heart_rate = None
        self.spo2 = None
        self.running = False
        
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.video_loop()
        
    def stop(self):
        self.running = False
        self.on_exit()
        
    def video_loop(self):
        if not self.running:
            return
        
        while self.running:
            ret, frame = self.cap.read()
            roi = self.get_roi(frame)
            if roi is not None:
                heart_rate, spo2 = self.estimate_heart_rate_spo2(roi, self.fps)
                self.heart_rates.append(heart_rate)
                self.times.append(time.time() - self.start_time)
                cv2.putText(frame, f"Heart Rate: {heart_rate:.0f} SpO2: {spo2:.2f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.show_video:
                cv2.imshow("Heart Rate Estimation", frame)
            else:
                print(f"Heart Rate: {heart_rate:.0f} bpm\nSpO2: {spo2:.2f}%")  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
            if time.time() - self.start_time > 10:
                print(f"Heart Rate: {heart_rate:.0f} bpm")
                self.start_time = time.time()
            
    def on_exit(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_roi(self, frame):
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

    def estimate_heart_rate_spo2(self, roi, fps):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        intensity = np.mean(thresh)
        heart_rate = intensity / fps * 10
        
        blue_channel, green_channel, red_channel = cv2.split(roi)
        mean_red = np.mean(red_channel)
        mean_infrared = np.mean(green_channel)
        ratio = mean_red / mean_infrared
        spo2 = -45.060 * ratio * ratio + 30.354 * ratio + 94.845
        return heart_rate, spo2
    
    def plot_heart_rate(self, times, heart_rates):
        plt.plot(times, heart_rates)
        plt.xlabel("Time (s)")
        plt.ylabel("Heart Rate (bpm)")
        plt.show()
        
heart_rate = HeartRateAndSpO2Estimation()
heart_rate.start()