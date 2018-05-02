from tkinter import *
from PIL import Image, ImageTk
import os
import cv2
import time
import threading
import numpy as np
from threading import Thread
from datetime import datetime

# Background Color 색상
R, G, B = 17, 21, 28
IPHONE_COLOR = "#{:2X}{:2X}{:2X}".format(R,G,B)

# WINDOW 크기
RATIO = 0.4
WIDTH = int(1125 * RATIO)
HEIGHT = int(2436 * RATIO)
RADIUS = 100 # 곡선 형태

# Video 프레임 재생 주기(1ms)
FPS = 3

# 사진 찍기 까지의 delay 시간
SHUTTER_LAG = 3

# Button 크기
BUTTON_RATIO = 75
BTN_UP_PATH = "./source/button_up.png"
BTN_DOWN_PATH = "./source/button_down.png"

# output 저장 위치
PICTURE_DIR = "./body/"

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
        self.canvas = None # 카메라 화면이 들어가는 부분

        self.counter = SHUTTER_LAG  # 사진 찍기 까지의 delay 시간
        self.counter_thread = None # 사진 찍기 전 counter를 노출하는 thread

        self.toggle_save = False # 이미지를 저장할 것인가 유무
        self.shutter_effect = 0
        self.set_window()

    def set_window(self):
        self.master.title(APPLICATION_TITLE)

        self.set_menu()
        self.set_video()
        self.set_iphone_button()
        self.set_iphone_round_border()

        self.show_frame() # 무한 루프로 계속 canvas에 frame을 노출시킴
        self.master.minsize(width=WIDTH,height=HEIGHT)

    def set_menu(self):
        self.menubar = Menu(self.master)

        filtermenu = Menu(self.menubar, tearoff=0)
        filtermenu.add_command(label="Gray")
        filtermenu.add_command(label="clahe")
        filtermenu.add_separator()
        self.menubar.add_cascade(label="filter",menu=filtermenu)

        outlinemenu = Menu(self.menubar, tearoff=0)
        outlinemenu.add_command(label="transparent")
        outlinemenu.add_command(label="bold outline")
        outlinemenu.add_separator()
        self.menubar.add_cascade(label="outline",menu=outlinemenu)

        self.master.config(menu=self.menubar)

    def set_video(self):
        # 카메라의 출력 부분을 설정해주는 함수
        self.canvas = Canvas(self.master,
            bg="white",
            height=HEIGHT,
            width=WIDTH)
        self.canvas.pack()

        self.widget = Label(self.canvas)
        self.widget.pack()
        self.canvas.create_window(225,500,window=self.widget)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2*HEIGHT)

    def show_frame(self):
        # 카메라의 frame 출력 처리에 대한 pipeline
        # 카메라의 filter 설정 등은 여기서 이루어져야 함
        _, frame = self.cap.read()

        frame = frame[:,200:440,:]
        frame = cv2.resize(frame,(430,860))
        frame = self.preprocess_image(frame)

        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2image = self.postprocess_image(cv2image)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.widget.image = imgtk
        self.widget.configure(image=imgtk)

        self.canvas.after(FPS, self.show_frame) # application이 FPS만큼 후 다시 self.show_frame을 실행

    def preprocess_image(self,frame):
        # 영상 품질 보정을 해주는 부분
        return cv2.GaussianBlur(frame,(5,5),0)

    def postprocess_image(self, frame):
        # shutter가 동작하였으면 그 이미지를 body 폴더에 담는 코드
        if self.toggle_save:
            self.save_image(frame)
            self.toggle_save = False
            self.shutter_effect = 255

        # shutter 효과 (찍었을 때 반짝하는 것)
        if self.shutter_effect > 0:
            frame = self.apply_shutter_effect(frame,self.shutter_effect)
            self.shutter_effect -= 40

        return frame

    def set_iphone_round_border(self):
        # 아이폰처럼 화면 내 border을 둥글게 나오도록 세팅
        points = [0+RADIUS, 0, 0+RADIUS, 0, WIDTH-RADIUS, 0, WIDTH-RADIUS, 0,
          WIDTH, 0, WIDTH, 0+RADIUS, WIDTH, 0+RADIUS, WIDTH, HEIGHT-RADIUS,
          WIDTH, HEIGHT-RADIUS, WIDTH, HEIGHT, WIDTH-RADIUS, HEIGHT,
          WIDTH-RADIUS, HEIGHT, 0+RADIUS, HEIGHT, 0+RADIUS, HEIGHT,
          0, HEIGHT, 0, HEIGHT-RADIUS, 0, HEIGHT-RADIUS, 0, 0+RADIUS, 0, 0+RADIUS, 0, 0]

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

    def save_image(self, frame):
        filename = datetime.now().strftime("%y%m%d_%H%M%S") + ".png"
        filepath = os.path.join(PICTURE_DIR, filename)
        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def apply_shutter_effect(self, frame, shutter_effect):
        frame[frame<shutter_effect] = shutter_effect
        return frame

    def press_shutter_btn(self, event=None):
        self.shutter_btn.config(image=self.image_down)
        if self.counter_thread is None:
            self.counter_thread = CounterThread(SHUTTER_LAG, self)
        elif not self.counter_thread.is_alive():
            self.counter_thread = CounterThread(SHUTTER_LAG, self)

    def release_shutter_btn(self, event=None):
        self.shutter_btn.config(image=self.image_up)
        if self.counter_thread.is_alive():
            self.counter_thread.set_counter(SHUTTER_LAG)
        else:
            self.counter_thread.start()

class CounterThread(Thread):
    # 화면에 카운트다운을 출력하는 함수
    def __init__(self, counter, app):
        Thread.__init__(self)
        self.counter = counter
        self.app = app

    def run(self):
        text = app.canvas.create_text(50,30,fill="white",font="Times 40 bold", text=str(self.counter),tags=('counter',))
        while self.counter > 0:
            time.sleep(1)
            self.counter -= 1
            app.canvas.itemconfigure(text, text=str(self.counter))
        app.canvas.delete('counter')
        self.app.toggle_save = True

    def set_counter(self, counter):
        self.counter = counter

if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    root.bind("<Escape>", lambda e : root.quit())
    root.mainloop()
