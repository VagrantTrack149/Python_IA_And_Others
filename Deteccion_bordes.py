import cv2
import tkinter as tk
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Tomar Foto", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.btn_start_camera = tk.Button(window, text="Mostrar Cámara", width=50, command=self.show_camera)
        self.btn_start_camera.pack(anchor=tk.CENTER, expand=True)

        self.btn_close = tk.Button(window, text="Cerrar", width=50, command=self.close)
        self.btn_close.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.show_camera_flag = False
        self.snapshot_flag = False

        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.window.mainloop()

    def show_camera(self):
        if not self.show_camera_flag:
            self.show_camera_flag = True
            self.btn_start_camera.config(text="Cerrar Cámara")
        else:
            self.show_camera_flag = False
            self.btn_start_camera.config(text="Mostrar Cámara")

    def snapshot(self):
        self.snapshot_flag = True

    def update(self):
        if self.show_camera_flag:
            ret, frame = self.vid.read()
            if ret:
                if self.snapshot_flag:
                    edges = cv2.Canny(frame, 100, 200)
                    cv2.imshow('Edges', edges)
                    self.snapshot_flag = False

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def close_window(self):
        self.close()
        self.window.destroy()

    def close(self):
        if self.show_camera_flag:
            self.show_camera()
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    AppWindow = tk.Tk()
    App = CameraApp(AppWindow, "Camera Application")
