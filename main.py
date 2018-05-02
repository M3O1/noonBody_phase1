from tkinter import *
from PIL import Image, ImageTk
import cv2
import time
import threading
from threading import Thread
from datetime import datetime

# Background Color 색상
R, G, B = 17, 21, 28
IPHONE_COLOR = "#{:2X}{:2X}{:2X}".format(R,G,B)

# WINDOW 크기
RATIO = 0.4
WIDTH = int(1125 * RATIO)
HEIGHT = int(2436 * RATIO)

# Video 프레임 재생 주기(1ms)
FPS = 3

# counter 크기
COUNTER = 5

# Button 크기
BUTTON_RATIO = 75
BTN_UP_PATH = "./source/button_up.png"
BTN_DOWN_PATH = "./source/button_down.png"

# output 저장 위치
PICTURE_DIR = "./body/" # 이미지 저장 폴더

# 프로그램 상단에 뜰 제목
APPLICATION_TITLE = 'noonBody'

def threaded(fn):
    '''
    Thread로 background에서 동작하도록 만들어주는 함수
    '''
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class Application(Frame):
    def __init__(self, master):
        self.master = master

        self.counter = COUNTER
        self.counter_thread = None
        self.set_window()

    def set_window(self):
        self.master.title(APPLICATION_TITLE)

        self.canvas = Canvas(self.master,
            bg="white",
            height=HEIGHT,
            width=WIDTH)
        self.canvas.pack()

        # set the video
        self.set_video()
        self.show_frame()

        # set Shutter button
        self.set_iphone_button()
        # set the border of iphone
        self.set_iphone_round_border(100)

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

        cv2image = self.postprocess_image(cv2image)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.widget.image = imgtk
        self.widget.configure(image=imgtk)
        self.canvas.after(FPS, self.show_frame)

    def preprocess_image(self,frame):
        return cv2.GaussianBlur(frame,(5,5),0)

    def postprocess_image(self, frame):
        return frame

    def set_iphone_round_border(self, radius=100):
        global WIDTH, HEIGHT
        points = [0+radius, 0, 0+radius, 0, WIDTH-radius, 0, WIDTH-radius, 0,
          WIDTH, 0, WIDTH, 0+radius, WIDTH, 0+radius, WIDTH, HEIGHT-radius,
          WIDTH, HEIGHT-radius, WIDTH, HEIGHT, WIDTH-radius, HEIGHT,
          WIDTH-radius, HEIGHT, 0+radius, HEIGHT, 0+radius, HEIGHT,
          0, HEIGHT, 0, HEIGHT-radius, 0, HEIGHT-radius, 0, 0+radius, 0, 0+radius, 0, 0]

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
        if self.counter_thread is None:
            self.counter_thread = CounterThread(COUNTER, self.canvas)
        elif not self.counter_thread.is_alive():
            self.counter_thread = CounterThread(COUNTER, self.canvas)

    def release_shutter_btn(self, event=None):
        self.shutter_btn.config(image=self.image_up)
        if self.counter_thread.is_alive():
            self.counter_thread.set_counter(COUNTER)
        else:
            self.counter_thread.start()

class CounterThread(Thread):
    # 화면에 카운트다운을 출력하는 함수
    def __init__(self, counter, canvas):
        Thread.__init__(self)
        self.counter = counter
        self.canvas = canvas

    def run(self):
        text = self.canvas.create_text(50,30,fill="white",font="Times 40 bold", text=str(self.counter),tags=('counter',))
        while self.counter > 0:
            time.sleep(1)
            self.counter -= 1
            self.canvas.itemconfigure(text, text=str(self.counter))
        self.canvas.delete('counter')

    def set_counter(self, counter):
        self.counter = counter

if __name__ == "__main__":
    root = Tk()

    app = Application(root)
    root.bind("<Escape>", lambda e : root.quit())
    root.mainloop()
