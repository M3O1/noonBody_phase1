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

CAM_WIDTH = 430
CAM_HEIGHT = 860

# Video 프레임 재생 주기(1ms)
FPS = 1

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

class Application(Frame):
    def __init__(self, master):
        self.master = master
        self.canvas = None # 카메라 화면이 들어가는 부분

        self.counter = SHUTTER_LAG  # 사진 찍기 까지의 delay 시간
        self.counter_thread = None # 사진 찍기 전 counter를 노출하는 thread

        self.toggle_save = False # 이미지를 저장할 것인가 유무
        self.shutter_effect = 0

        self.check_gray = IntVar()

        self.check_vignette = IntVar()
        self.vignette_xs, self.vignette_xc = 0.8, 1.2 # xs -> x축 side 강도 / xc -> x축 center 강도
        self.vignette_ys, self.vignette_yc = 0.8, 1.2 # ys -> y축 side 강도 / yc -> y축 center 강도
        self.update_vignette_filter()

        self.check_clahe = IntVar()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self.check_gamma = IntVar()
        self.gamma_value = 0.8
        self.update_gamma_lut()

        self.check_color = IntVar()
        self.red_weight = 1.2   # red channel에 대한 gamma 가중치
        self.green_weight = 1.0 # green channel에 대한 gamma 가중치
        self.blue_weight = 0.8  # blue channel에 대한 gamma 가중치
        self.update_color_lut()

        self.set_window()
        self.bind_key_to_frame()

    def set_window(self):
        # application 화면 구성을 정하는 메소드
        self.master.title(APPLICATION_TITLE)

        self.set_video()
        self.set_board()
        self.set_iphone_button()
        self.set_iphone_round_border()

        self.show_frame() # 무한 루프로 계속 canvas에 frame을 노출시킴
        self.master.minsize(width=WIDTH+300,height=HEIGHT)

    def set_board(self):
        self.frame = Frame(self.master,width=300,height=500)
        self.frame.grid(row=0,column=1,sticky='ne')

        row_idx = 0

        # Gray 이미지로 변환
        Label(self.frame, text="GRAY < - > RGB").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.gray_button = Checkbutton(self.frame, text='GRAY', variable=self.check_gray)
        self.gray_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        # vignette 필터 효과
        row_idx += 1
        Label(self.frame, text="Vignette filter").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_button = Checkbutton(self.frame, text='Vignette', variable=self.check_vignette)
        self.vignette_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="  수직 side weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.vignette_ys_scale = Scale(self.frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_ys_scale.set(self.vignette_ys)
        self.vignette_ys_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.frame, text="  수직 center weight").grid(row=row_idx,column=0,pady=3,sticky='w')
        self.vignette_yc_scale = Scale(self.frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_yc_scale.set(self.vignette_yc)
        self.vignette_yc_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.frame, text="  수평 side weight").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_xs_scale = Scale(self.frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_xs_scale.set(self.vignette_xs)
        self.vignette_xs_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.frame, text="  수평 center weight").grid(row=row_idx,column=0,pady=3,sticky='w')
        self.vignette_xc_scale = Scale(self.frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_xc_scale.set(self.vignette_xc)
        self.vignette_xc_scale.grid(row=row_idx,column=1,sticky="ew")

        # contrast 강화 효과
        row_idx += 1
        Label(self.frame, text="contrast(clahe)").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.clahe_button = Checkbutton(self.frame, text='apply', variable=self.check_clahe)
        self.clahe_button.grid(row=row_idx, column=1, sticky='E',pady=10)

        # 명도 보정 효과
        row_idx += 1
        Label(self.frame, text="명도 보정").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.gamma_button = Checkbutton(self.frame, text='gamma correction', variable=self.check_gamma)
        self.gamma_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="gamma value").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.gamma_scale = Scale(self.frame, from_=0.3,to=2.0, orient=HORIZONTAL,resolution=0.1)
        self.gamma_scale.set(self.gamma_value)
        self.gamma_scale.grid(row=row_idx,column=1,sticky="ew")

        # 색조 보정 효과
        row_idx += 1
        Label(self.frame, text="색조 보정").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_button = Checkbutton(self.frame, text='apply', variable=self.check_color)
        self.vignette_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="  red weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.red_weight_scale = Scale(self.frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.red_weight_scale.set(self.red_weight)
        self.red_weight_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.frame, text="  green weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.green_weight_scale = Scale(self.frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.green_weight_scale.set(self.green_weight)
        self.green_weight_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.frame, text="  blue weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.blue_weight_scale = Scale(self.frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.blue_weight_scale.set(self.blue_weight)
        self.blue_weight_scale.grid(row=row_idx,column=1,sticky="ew")

    def bind_key_to_frame(self):
        # component와 event handler를 bind하는 메소드
        self.vignette_ys_scale.configure(command=self.convert_vignette_ys)
        self.vignette_yc_scale.configure(command=self.convert_vignette_yc)
        self.vignette_xs_scale.configure(command=self.convert_vignette_xs)
        self.vignette_xc_scale.configure(command=self.convert_vignette_xc)
        self.gamma_scale.configure(command=self.convert_gamma)
        self.red_weight_scale.configure(command=self.convert_red_weight)
        self.green_weight_scale.configure(command=self.convert_green_weight)
        self.blue_weight_scale.configure(command=self.convert_blue_weight)

    def set_video(self):
        # 카메라의 출력 부분을 설정해주는 함수
        self.canvas = Canvas(self.master,
            bg="white",
            height=HEIGHT,
            width=WIDTH)
        self.canvas.grid(row=0,column=0)

        self.widget = Label(self.canvas)
        self.widget.pack()
        self.canvas.create_window(225,500,window=self.widget)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2*HEIGHT)

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

    def show_frame(self):
        # 카메라의 frame 출력 처리에 대한 pipeline
        # 카메라의 filter 설정 등은 여기서 이루어져야 함
        _, frame = self.cap.read()

        frame = frame[:,200:440,:]
        frame = cv2.resize(frame,(CAM_WIDTH,CAM_HEIGHT))
        frame = self.preprocess_image(frame)

        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2image = self.postprocess_image(cv2image)
        cv2image = self.shutter_image(cv2image)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.widget.image = imgtk
        self.widget.configure(image=imgtk)

        self.canvas.after(FPS, self.show_frame) # application이 FPS만큼 후 다시 self.show_frame을 실행

    def preprocess_image(self,frame):
        # 영상 품질 보정을 해주는 부분
        return cv2.GaussianBlur(frame,(5,5),0)

    def shutter_image(self,frame):
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

    def postprocess_image(self,frame):
        # 필터 처리를 담당하는 메소드
        if self.check_vignette.get() == 1:
            # vignette 필터 적용
            frame =np.uint8(np.clip(frame*self.vignette_mask, 0, 250))

        if self.check_clahe.get() == 1:
            # clahe 필터 적용
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            frame[:,:,0] = self.clahe.apply(frame[:,:,0])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)

        if self.check_gamma.get() == 1:
            # 명도 보정 필터 적용
            frame = cv2.LUT(frame, self.gamma_lut)

        if self.check_color.get() == 1:
            # 색감 보정 필터 적용
            frame = cv2.LUT(frame, self.color_lut)

        if self.check_gray.get() == 1:
            # 흑백으로 변경
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return frame

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

    def convert_vignette_ys(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.vignette_ys = float(event)
            self.update_vignette_filter()

    def convert_vignette_yc(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.vignette_yc = float(event)
            self.update_vignette_filter()

    def convert_vignette_xs(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.vignette_xs = float(event)
            self.update_vignette_filter()

    def convert_vignette_xc(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.vignette_xc = float(event)
            self.update_vignette_filter()

    def convert_gamma(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.gamma_value = float(event)
            self.update_gamma_lut()

    def convert_red_weight(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.red_weight = float(event)
            self.update_color_lut()

    def convert_green_weight(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.green_weight = float(event)
            self.update_color_lut()

    def convert_blue_weight(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.blue_weight = float(event)
            self.update_color_lut()

    def update_vignette_filter(self):
        self.vignette_mask = create_vignette_mask(CAM_WIDTH,CAM_HEIGHT,self.vignette_xs, self.vignette_xc, self.vignette_ys, self.vignette_yc)

    def update_gamma_lut(self):
        self.gamma_lut = create_lut(self.gamma_value)

    def update_color_lut(self):
        self.color_lut = create_lut([self.red_weight,self.green_weight,self.blue_weight])

'''
Helper
    1. decorator
        - threaded : 메소드를 background thread로 동작하도록 만드는 함수
        - memoize  : cache를 이용하여 반복 연산하지 않도록 도와주는 함수

    2. class
        - CounterThread : 화면에 카운트다운을 출력하도록 도와주는 클래스. Thread를 이용하여 background 연산함

    3. method
        - create_vignette_mask & vignette_func : vignette filter(hightlight 강조)하는 함수
        - create_lut : Look-Up Table을 생성하는 함수
'''
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def memoize(func):
    cache = {}

    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer

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

def create_vignette_mask(width, height,
                        side_width, center_width,
                        side_height, center_height):
    f_x = vignette_func(width, side_width, center_width)
    f_y = vignette_func(height, side_height, center_height)

    vignette_mask = (np.vstack([f_x(np.arange(0,width))]*height)*np.vstack([f_y(np.arange(0,height))]*width).T)
    return np.expand_dims(vignette_mask,axis=-1)

def vignette_func(n, min_d, max_d):
    return np.vectorize(lambda x : -(max_d-min_d)/((n//2)**2)*x*(x-n+1)+min_d)

@memoize
def create_lut(weights):
    if isinstance(weights, (int,float)):
        inv_w = 1.0 / weights
        lut = np.array([((i / 255.0) ** inv_w) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return lut
    else:
        luts = []
        for weight in weights:
            inv_w = 1.0 / weight
            luts.append(np.array([((i / 255.0) ** inv_w) * 255
                for i in np.arange(0, 256)]).astype("uint8"))
        lut = np.dstack(luts)
        return lut

if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    root.bind("<Escape>", lambda e : root.quit())
    root.mainloop()
