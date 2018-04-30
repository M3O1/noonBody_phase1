from tkinter import *
from PIL import Image, ImageTk
import cv2
import time
import threading
from datetime import datetime

# Background Color 색상
R, G, B = 17, 21, 28
IPHONE_COLOR = "#{:2X}{:2X}{:2X}".format(R,G,B)

# WINDOW 크기
RATIO = 0.4
WIDTH = int(1125 * RATIO)
HEIGHT = int(2436 * RATIO)

# Button 크기
BUTTON_RATIO = 75
# BUTTON 위치
BTN_UP_PATH = "./source/button_up.png"
BTN_DOWN_PATH = "./source/button_down.png"


# output 저장 위치
PICTURE_DIR = "./body/" # 이미지 저장 폴더


# 프로그램 상단에 뜰 제목
APPLICATION_TITLE = 'noonBody'

class Application(Frame):
    def __init__(self, master):
        self.master = master
        self.set_window()

    def set_window(self):
        self.master.title(APPLICATION_TITLE)

        self.canvas = Canvas(self.master,
            bg="white",
            height=HEIGHT,
            width=WIDTH)
        self.canvas.pack()

        self.set_video()
        self.show_frame()
        # button

        self.set_iphone_round_border(100)
        self.set_iphone_button()
        self.master.minsize(width=WIDTH,height=HEIGHT)

    def set_video(self):
        self.widget = Label(self.canvas)
        self.widget.pack()
        self.canvas.create_window(225,500,window=self.widget)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2*HEIGHT)

    def show_frame(self):
        _, frame = self.cap.read()

        frame = frame[:,200:440,:]
        frame = cv2.resize(frame,(430,860))
        frame = self.preprocess_image(frame)
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.widget.image = imgtk
        self.widget.configure(image=imgtk)
        self.canvas.after(3, self.show_frame)

    def preprocess_image(self,frame):
        return cv2.GaussianBlur(frame,(5,5),0)

    def set_iphone_round_border(self, radius=100):
        global WIDTH, HEIGHT
        points = [0+radius, 0, 0+radius, 0,
          WIDTH-radius, 0, WIDTH-radius, 0,
          WIDTH, 0,
          WIDTH, 0+radius, WIDTH, 0+radius,
          WIDTH, HEIGHT-radius, WIDTH, HEIGHT-radius,
          WIDTH, HEIGHT, WIDTH-radius, HEIGHT,
          WIDTH-radius, HEIGHT,
          0+radius, HEIGHT, 0+radius, HEIGHT,
          0, HEIGHT,
          0, HEIGHT-radius, 0, HEIGHT-radius,
          0, 0+radius, 0, 0+radius,
          0, 0]
        self.canvas.create_polygon(points,
            fill=IPHONE_COLOR,
            smooth=True)

    def set_iphone_button(self):
        self.image_up = self.read_image(BTN_UP_PATH)
        self.image_down = self.read_image(BTN_DOWN_PATH)

        self.shutter_btn = Button(image=self.image_up)
        self.shutter_btn.bind("<ButtonPress>",
            self.press_shutter_btn)
        self.shutter_btn.bind("<ButtonRelease>",
            self.release_shutter_btn)

        self.shutter_btn.configure(activebackground = IPHONE_COLOR,
                                  bg=IPHONE_COLOR,
                                  highlightthickness=0,
                                  borderwidth=0,
                                  relief=FLAT,
                                  overrelief=FLAT)

        self.canvas.create_window(WIDTH//2,HEIGHT-100, window=self.shutter_btn)

    def read_image(self,path):
        image = Image.open(path)
        return ImageTk.PhotoImage(image.resize((BUTTON_RATIO,BUTTON_RATIO), Image.ANTIALIAS))

    def press_shutter_btn(self, event=None):
        self.shutter_btn.config(image=self.image_down)

    def release_shutter_btn(self, event=None):
        self.shutter_btn.config(image=self.image_up)

if __name__ == "__main__":
    root = Tk()

    app = Application(root)
    root.bind("<Escape>", lambda e : root.quit())
    root.mainloop()
