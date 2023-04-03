import cv2
import numpy as np
import time
import tkinter
from tkinter import ttk, Tk
import sv_ttk
from PIL import Image, ImageTk

class HeartRateEstimationGUI:
    def __init__(self):
        self.root = Tk()
        sv_ttk.set_theme("dark")
        self.root.title("Heart Rate Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.label_heart_rate = ttk.Label(self.root, text="Heart Rate: --- bpm", font=("Arial", 14))
        self.label_heart_rate.pack(pady=10)

        self.canvas_video = tkinter.Canvas(self.root, width=800, height=480)
        self.canvas_video.pack()

        self.button_start = ttk.Button(self.root, text="Start", command=self.start)
        self.button_start.pack(pady=10)

        self.button_stop = ttk.Button(self.root, text="Stop", command=self.stop, state=tkinter.DISABLED)
        self.button_stop.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.heart_rate = None
        self.running = False

        self.root.mainloop()

    def start(self):
        self.button_start.config(state=tkinter.DISABLED)
        self.button_stop.config(state=tkinter.NORMAL)
        self.running = True
        self.start_time = time.time()
        self.video_loop()

    def stop(self):
        self.button_start.config(state=tkinter.NORMAL)
        self.button_stop.config(state=tkinter.DISABLED)
        self.running = False
        self.on_exit()

    def video_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        roi = self.get_roi(frame)
        if roi is not None:
            self.heart_rate = self.estimate_heart_rate(roi, self.fps)
            self.label_heart_rate.config(text=f"Heart Rate: {self.heart_rate:.0f} bpm")

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()
        image_width = self.photo.width()
        image_height = self.photo.height()
        x = (canvas_width - image_width) / 2
        y = (canvas_height - image_height) / 2
        self.canvas_video.create_image(x, y, image=self.photo, anchor=tkinter.NW)
        self.root.after(15, self.video_loop)

    def get_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        rects = []
        for (x, y, w, h) in faces:
            rects.append((x, y, w, h))
        for rect in rects:
            x, y, w, h = rect
            roi = frame[y:y + h, x:x + w]
            return roi
        return None

    def estimate_heart_rate(self, roi, fps):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        intensity = np.mean(thresh)
        heart_rate = intensity / fps * 10
        return heart_rate
        
    def on_exit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

HeartRateEstimationGUI()