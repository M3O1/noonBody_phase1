from tkinter import *
from tkinter import filedialog
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

        self.check_mask = IntVar()
        self.check_mask.set(1)
        self.mask_image = None
        self.check_blend_type = StringVar()
        self.check_mask_type = StringVar()
        self.filename_text = StringVar()
        self.blend_ratio = 0.1

        self.check_outfocus_bg = IntVar()
        self.outfocus_blur = (31,31) # outfocus를 위한 blur의 ksize
        self.check_dark_bg = IntVar()
        self.background_lut = create_lut(0.5)
        self.check_grid = IntVar()
        self.grid_mask = self.get_grid_mask()

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
        self.master.minsize(width=WIDTH+700,height=HEIGHT)

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

    def set_board(self):
        self.frame = Frame(self.master,width=300,height=2000,borderwidth=1,relief=GROOVE,padx=20,pady=10)
        self.frame.grid(row=0,column=1,sticky='ne')

        row_idx = 0

        row_idx += 1
        Label(self.frame, text="MASK", font=('Helvetica',15)).grid(row=row_idx,column=0,pady=10,sticky='w')
        # filename show
        row_idx += 1
        Label(self.frame, text="mask 적용 유무 :").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.mask_button = Checkbutton(self.frame, text='apply', variable=self.check_mask)
        self.mask_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="mask 적용 방식 :").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.mask_option = OptionMenu(self.frame, self.check_blend_type, "합치기", "블렌드하기")
        self.check_blend_type.set("합치기")
        self.mask_option.grid(row=row_idx, column=1, sticky="E",pady=5)

        row_idx += 1
        Label(self.frame, text="mask 적용 형태 :").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.mask_feature_option = OptionMenu(self.frame, self.check_mask_type, "외각선만", "전체")
        self.check_mask_type.set("외각선만")
        self.mask_feature_option.grid(row=row_idx, column=1, sticky="E",pady=5)

        row_idx += 1
        Label(self.frame, text="blend ratio").grid(row=row_idx,column=0,pady=3,sticky='w')
        self.blend_scale = Scale(self.frame, from_=0.0,to=1.0, orient=HORIZONTAL,resolution=0.1)
        self.blend_scale.set(self.blend_ratio)
        self.blend_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        self.file_mask_btn = Button(self.frame, text="mask image",width=30,height=2)
        self.file_mask_btn.grid(row=row_idx,column=0,columnspan=2,sticky="nwe",pady=5)
        row_idx+=1
        self.preview_imagebox = Label(self.frame,height=20,width=20,background='black')
        self.preview_imagebox.grid(row=row_idx,column=0,columnspan=2,sticky="ew",pady=5)

        row_idx += 1
        Label(self.frame, text="촬영 효과", font=('Helvetica',15)).grid(row=row_idx,column=0,pady=10,sticky='w')

        row_idx += 1
        Label(self.frame, text="out-focus").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.outfocus_bg_button = Checkbutton(self.frame, text='적용하기', variable=self.check_outfocus_bg)
        self.outfocus_bg_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="background-dark").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.dark_bg_button = Checkbutton(self.frame, text='적용하기', variable=self.check_dark_bg)
        self.dark_bg_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.frame, text="grid-line").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.grid_button = Checkbutton(self.frame, text='적용하기', variable=self.check_grid)
        self.grid_button.grid(row=row_idx, column=1, sticky='E',pady=5)


        self.mask_frame = Frame(self.master,width=300,height=500,borderwidth=1,relief=GROOVE,padx=20,pady=10)
        self.mask_frame.grid(row=0,column=2,sticky='ne')
        row_idx = 0

        Label(self.mask_frame, text="FILTER",font=('Helvetica',15)).grid(row=row_idx,column=0,pady=10,sticky='w')
        # Gray 이미지로 변환
        row_idx += 1
        Label(self.mask_frame, text="GRAY < - > RGB").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.gray_button = Checkbutton(self.mask_frame, text='GRAY', variable=self.check_gray)
        self.gray_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        # vignette 필터 효과
        row_idx += 1
        Label(self.mask_frame, text="Vignette filter").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_button = Checkbutton(self.mask_frame, text='Vignette', variable=self.check_vignette)
        self.vignette_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.mask_frame, text="  수직 side weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.vignette_ys_scale = Scale(self.mask_frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_ys_scale.set(self.vignette_ys)
        self.vignette_ys_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.mask_frame, text="  수직 center weight").grid(row=row_idx,column=0,pady=3,sticky='w')
        self.vignette_yc_scale = Scale(self.mask_frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_yc_scale.set(self.vignette_yc)
        self.vignette_yc_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.mask_frame, text="  수평 side weight").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_xs_scale = Scale(self.mask_frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_xs_scale.set(self.vignette_xs)
        self.vignette_xs_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.mask_frame, text="  수평 center weight").grid(row=row_idx,column=0,pady=3,sticky='w')
        self.vignette_xc_scale = Scale(self.mask_frame, from_=0.5,to=1.5, orient=HORIZONTAL,resolution=0.1)
        self.vignette_xc_scale.set(self.vignette_xc)
        self.vignette_xc_scale.grid(row=row_idx,column=1,sticky="ew")

        # contrast 강화 효과
        row_idx += 1
        Label(self.mask_frame, text="contrast(clahe)").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.clahe_button = Checkbutton(self.mask_frame, text='apply', variable=self.check_clahe)
        self.clahe_button.grid(row=row_idx, column=1, sticky='E',pady=10)

        # 명도 보정 효과
        row_idx += 1
        Label(self.mask_frame, text="명도 보정").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.gamma_button = Checkbutton(self.mask_frame, text='gamma correction', variable=self.check_gamma)
        self.gamma_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.mask_frame, text="gamma value").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.gamma_scale = Scale(self.mask_frame, from_=0.3,to=2.0, orient=HORIZONTAL,resolution=0.1)
        self.gamma_scale.set(self.gamma_value)
        self.gamma_scale.grid(row=row_idx,column=1,sticky="ew")

        # 색조 보정 효과
        row_idx += 1
        Label(self.mask_frame, text="색조 보정").grid(row=row_idx,column=0,pady=10,sticky='w')
        self.vignette_button = Checkbutton(self.mask_frame, text='apply', variable=self.check_color)
        self.vignette_button.grid(row=row_idx, column=1, sticky='E',pady=5)

        row_idx += 1
        Label(self.mask_frame, text="  red weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.red_weight_scale = Scale(self.mask_frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.red_weight_scale.set(self.red_weight)
        self.red_weight_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.mask_frame, text="  green weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.green_weight_scale = Scale(self.mask_frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.green_weight_scale.set(self.green_weight)
        self.green_weight_scale.grid(row=row_idx,column=1,sticky="ew")

        row_idx += 1
        Label(self.mask_frame, text="  blue weight").grid(row=row_idx,column=0,pady=3,sticky=S)
        self.blue_weight_scale = Scale(self.mask_frame, from_=0.3,to=1.8, orient=HORIZONTAL,resolution=0.1)
        self.blue_weight_scale.set(self.blue_weight)
        self.blue_weight_scale.grid(row=row_idx,column=1,sticky="ew")

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

    def bind_key_to_frame(self):
        # component와 event handler를 bind하는 메소드
        self.file_mask_btn.configure(command=self.select_mask_file)
        self.blend_scale.configure(command=self.convert_blend_ratio)
        self.vignette_ys_scale.configure(command=self.convert_vignette_ys)
        self.vignette_yc_scale.configure(command=self.convert_vignette_yc)
        self.vignette_xs_scale.configure(command=self.convert_vignette_xs)
        self.vignette_xc_scale.configure(command=self.convert_vignette_xc)
        self.gamma_scale.configure(command=self.convert_gamma)
        self.red_weight_scale.configure(command=self.convert_red_weight)
        self.green_weight_scale.configure(command=self.convert_green_weight)
        self.blue_weight_scale.configure(command=self.convert_blue_weight)

    def show_frame(self):
        # 카메라의 frame 출력 처리에 대한 pipeline
        # 카메라의 filter 설정 등은 여기서 이루어져야 함
        _, frame = self.cap.read()

        frame = self.adjust_frame(frame) # 영상을 올바른 형태로 변환
        frame = self.preprocess_image(frame) # 영상 품질 보정
        frame = self.shutter_image(frame) # 영상 저장할 경우, 찰칵 효과 + 저장
        frame = self.postprocess_image(frame) # 영상에 필터 적용

        frame = self.apply_mask(frame) # 영상에 마스크 적용

        self.print_to_canvas(frame) # 영상을 canvas에 올리기
        self.canvas.after(FPS, self.show_frame) # application이 FPS만큼 후 다시 self.show_frame을 실행

    def show_preview_image(self):
        # 현재 mask의 original image를 보여줌
        if self.mask_image is not None:
            preview_image = cv2.resize(self.mask_image,(150,300))
            self.preview_image = ImageTk.PhotoImage(Image.fromarray(preview_image))
            self.preview_imagebox.configure(image=self.preview_image,height=300,width=30)
            self.preview_imagebox.image = self.preview_image

    def adjust_frame(self,frame):
        # frame의 size, direction, color을 보정
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:,200:440,:]
        frame = cv2.resize(frame,(CAM_WIDTH,CAM_HEIGHT))
        return frame

    def preprocess_image(self,frame):
        # 영상 품질 보정을 해주는 부분
        return cv2.GaussianBlur(frame,(5,5),0)

    def postprocess_image(self,frame):
        frame = self.filter_image(frame)
        frame = self.apply_shooting_effect(frame)
        return frame

    def filter_image(self,frame):
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
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    def apply_shooting_effect(self,frame):
        # 마스크 적용하기
        if self.mask_image is None:
            return frame

        if self.check_outfocus_bg.get() == 1 or self.check_outfocus_bg.get() == 1:

            mask = cv2.cvtColor(self.mask_image, cv2.COLOR_RGB2GRAY)
            mask_inv = ~mask
            bg = cv2.bitwise_and(frame,frame,mask=mask_inv)
            frame = cv2.bitwise_and(frame,frame,mask=mask)

            if self.check_outfocus_bg.get() == 1:
                bg = cv2.blur(bg,self.outfocus_blur)
                bg = cv2.bitwise_and(bg,bg,mask=mask_inv)

            if self.check_dark_bg.get() == 1:
                bg = cv2.LUT(bg, self.background_lut)

            frame = cv2.add(bg,frame)

        if self.check_grid.get() == 1:
            frame = cv2.add(self.grid_mask,frame)

        return frame

    def apply_outfocus_effect(self,frame):
        mask = cv2.cvtColor(self.mask_image, cv2.COLOR_RGB2GRAY)
        mask_inv = ~mask

        bg = cv2.bitwise_and(frame,frame,mask=mask_inv)
        frame = cv2.bitwise_and(frame,frame,mask=mask)

        bg = cv2.blur(bg,(31,31))
        bg = cv2.bitwise_and(bg,bg,mask=mask_inv)
        return cv2.add(frame,bg)

    def apply_mask(self, frame):
        # 마스크 적용하기
        if self.mask_image is None:
            return frame
        if self.check_mask.get() == 1:
            if self.check_mask_type.get() == "외각선만":
                mask_image = self.mask_contour_image
            else:
                mask_image = self.mask_image

            if self.check_blend_type.get() == "합치기":
                mask = (mask_image * self.blend_ratio).astype(np.uint8)
                frame = cv2.add(frame, mask)

            else:
                frame = cv2.addWeighted(frame, (1-self.blend_ratio), mask_image, self.blend_ratio, 0)
        return frame

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

    def print_to_canvas(self,frame):
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.widget.image = imgtk
        self.widget.configure(image=imgtk)

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

    def extract_contour_mask(self, mask):
        kernel = np.ones((10,10), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations =1)
        return mask - erosion

    def get_grid_mask(self):
        grid_mask = np.zeros((CAM_HEIGHT,CAM_WIDTH,3),dtype=np.uint8)
        for i in np.linspace(0,CAM_WIDTH,3,endpoint=False, dtype=np.int)[1:]:
            grid_mask = cv2.line(grid_mask,(i,0),(i,CAM_HEIGHT),(255,255,255),1)
        for i in np.linspace(0,CAM_HEIGHT,3,endpoint=False, dtype=np.int)[1:]:
            grid_mask = cv2.line(grid_mask,(0,i),(CAM_WIDTH,i),(255,255,255),1)
        return grid_mask

    def select_mask_file(self, event=None):
        # 적용할 마스크가 있는 이미지를 선택
        self.mask_path = filedialog.askopenfilename()
        if self.mask_path == "" or self.mask_path is None:
            return
        self.mask_image = cv2.imread(self.mask_path)
        self.mask_image = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2RGB)
        self.mask_image = cv2.resize(self.mask_image, (CAM_WIDTH,CAM_HEIGHT))
        self.mask_contour_image = self.extract_contour_mask(self.mask_image)

        self.show_preview_image()

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

    def convert_blend_ratio(self,event):
        re_num = re.compile("^\d*(\.?\d*)$")
        if re_num.match(str(event)):
            self.blend_ratio = float(event)

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
