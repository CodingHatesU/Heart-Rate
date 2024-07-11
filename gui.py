import cv2
import numpy as np
import time
import tkinter
from tkinter import ttk, Tk
import sv_ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HeartRateEstimationGUI:
    def __init__(self):
        self.root = Tk()
        sv_ttk.set_theme("dark")
        self.root.title("Heart Rate Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.label_heart_rate = ttk.Label(self.root, text="Heart Rate: --- bpm\nSpO2: ---", font=("Arial", 14))
        self.label_heart_rate.grid(row=0,column=0, columnspan=2, pady=10)

        self.frame_video = ttk.Frame(self.root)
        self.frame_video.grid(row=1, column=0, columnspan=2, pady=10, padx=20)

        self.canvas_video = tkinter.Canvas(self.frame_video, width=800, height=480)
        self.canvas_video.grid(row=1,column=0)
        
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.button_start = ttk.Button(self.frame_buttons, text="Start", command=self.start)
        self.button_start.grid(row=2,column=0, padx=(0,10))

        self.button_stop = ttk.Button(self.frame_buttons, text="Stop", command=self.stop, state=tkinter.DISABLED)
        self.button_stop.grid(row=2,column=1, padx=(10,0))
        
        self.frame_buttons.grid_columnconfigure(0, weight=1)
        self.frame_buttons.grid_columnconfigure(1, weight=1)

        self.cap = cv2.VideoCapture(0)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.heart_rate = None
        self.spo2 = None
        self.heart_rates = []
        self.spo2_levels = []
        self.times = []
        self.running = False

        self.fig = plt.figure(figsize=(6, 4), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Intensity')
        self.line, = self.ax.plot([], [])
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.frame_video)
        self.canvas_graph.get_tk_widget().grid(row=1,column=1)

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
            self.heart_rate, self.spo2 = self.estimate_heart_rate_spo2(roi, self.fps)
            self.heart_rates.append(self.heart_rate)
            self.spo2_levels.append(self.spo2)
            self.times.append(time.time() - self.start_time)
            self.label_heart_rate.config(text=f"Heart Rate: {self.heart_rate:.0f} bpm\nSpO2: {self.spo2:.0f}%")
            
            t = time.time() - self.start_time
            intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            self.ax.scatter(self.times, self.heart_rates, color='red', label='Heart Rate')
            self.ax.scatter(self.times, self.spo2_levels, color='blue', label='SpO2')
            self.line.set_data(self.ax.get_lines()[0].get_xdata(), self.ax.get_lines()[0].get_ydata())
            self.ax.set_xlim(0, t + 10)
            self.ax.set_ylim(0, 255)
            self.canvas_graph.draw()

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
        
    def on_exit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

HeartRateEstimationGUI()