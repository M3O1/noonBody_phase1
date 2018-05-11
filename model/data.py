import os
import time
import glob
import random
import re

import h5py
import numpy as np
import pandas as pd
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise

import boto3
from botocore.exceptions import ClientError

from multiprocessing import Pool
import multiprocessing
from functools import partial

BUCKET_NAME = "baidu-segmentation-dataset"

'''
    Full Data Generation Method

    핵심 메소드
    dataGenerator
    : 데이터를 배치단위로 생성하는 메소드

    관련 메소드

    1. 데이터셋 생성 관련 메소드
        - load_dataset
        - check_h5
        - get_fnameset

    2. Augmentation 관련 메소드
        - apply_rotation
        - apply_rescaling
        - apply_flip
        - apply_random_crop

    3. S3 Communication 관련 메소드
        - download
        - upload
        - get_s3_keys
        - download_whole_dataset

'''
def dataGenerator(data_dir, input_size, batch_size=64):
    x_data,ydata = load_dataset(data_dir=data_dir,input_size=input_size)
    dataset = list(zip(xdata,ydata))

    while True:
        random.shuffle(dataset)
        gen = dataset.copy()

        counter = 0; batch_x = []; batch_y = []
        for image, label in gen:
            counter += 1

            x,y = apply_rotation(image,label)
            x,y = apply_rescaling(x,y)
            x,y = apply_flip(x,y)
            x,y = apply_random_crop(x,y, input_size)
            x = random_noise(x,mode='gaussian',mean=0,var=0.001)
            # adjust the range of value
            x = np.clip(x,0.,1.)
            y = (y>0.7).astype(int)

            batch_x.append(x); batch_y.append(y)
            if counter == batch_size:
                yield np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)
                counter = 0; batch_x = []; batch_y = []

def load_dataset(data_dir="./data", input_size=(48,48)):

    # 이전에 h5py로 저장해두었다면 그것을 Load
    h5_path = os.path.join(data_dir,"image.h5")
    xdata,ydata = check_h5(h5_path,input_size)
    if xdata is not None:
        return xdata, ydata

    image_dir = os.path.join(data_dir,"images")
    label_dir = os.path.join(data_dir,"profiles")

    # 아예 다운도 받지 않았다면 다운 받자
    if image_dir is None or label_dir is None or\
        not os.path.exists(image_dir) or not os.path.exists(label_dir) or\
        len(os.listdir(image_dir)) <= 100 or len(os.listdir(label_dir)) <= 100:
        s3 = boto3.client('s3')
        image_dir, label_dir = download_whole_dataset(s3)

    # 다운 받았으면 이것을 저장해버리자
    fname_list = list(get_fnameset(image_dir) & get_fnameset(label_dir))
    cut_size = int(input_size[0] * 4/3), int(input_size[1] * 4/3)
    images = []
    labels = []
    while len(fname_list) > 0:
        fname = fname_list.pop()
        image_path = os.path.join(image_dir,fname)
        label_path = os.path.join(label_dir,fname)
        # read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path,0)

        # normalization
        image = image / 255.
        label = label / 255

        image = cv2.resize(image,cut_size)
        label = cv2.resize(label,cut_size)

        label = (label>0.7).astype(int)
        images.append(image)
        labels.append(label)

    xdata = np.stack(images, axis=0)
    ydata = np.stack(labels, axis=0)

    with h5py.File(h5_path) as h5:
        # 저장
        grp = h5.create_group("{},{}".format(*input_size))
        grp.create_dataset('xdata',data=xdata)
        grp.create_dataset('ydata',data=ydata)

    return xdata, ydata

def check_h5(h5_path,input_size):
    if os.path.exists(h5_path):
        with h5py.File(h5_path) as h5:
            key = "{},{}".format(*input_size)
            if key in h5:
                xdata = h5[key]['xdata'][:]
                ydata = h5[key]['ydata'][:]
                return xdata, ydata
    return None, None

def get_fnameset(dirpath):
    # 디렉토리 내 filename set을 추출하는 메소드
    return set([filename for filename in os.listdir(dirpath)])

'''
    Augmentation 관련 메소드
'''
def apply_rotation(image, label):
    rotation_angle = random.randint(-8,8)
    return rotate(image, rotation_angle), rotate(label, rotation_angle)

def apply_rescaling(image, label):
    rescale_ratio = 0.9 + random.random() * 0.2 # from 0.9 to 1.1
    return rescale(image, rescale_ratio,mode='reflect'),\
            rescale(label, rescale_ratio,mode='reflect')

def apply_flip(image, label):
    if random.random()>0.5:
        return image[:,::-1,:], label[:,::-1]
    else:
        return image, label

def apply_random_crop(image, label, input_size):
    iv = random.randint(0,image.shape[0]-input_size[0])
    ih = random.randint(0,image.shape[1]-input_size[1])
    return image[iv:iv+input_size[0],ih:ih+input_size[1],:],\
            label[iv:iv+input_size[0],ih:ih+input_size[1]]

'''
    S3 Communication 관련 메소드
'''
def download(s3, bucket, obj, local_file_path):
    s3.download_file(bucket, obj,local_file_path)

def upload(s3, bucket, obj, local_file_path):
    s3.upload_file(local_file_path,bucket,obj)

def get_s3_keys(s3, bucket=BUCKET_NAME):
    # s3 버킷 내 key 집합을 구함
    keys = []
    res = s3.list_objects_v2(Bucket=bucket)
    while True:
        if not 'Contents' in res:
            break
        for obj in res['Contents']:
            keys.append(obj['Key'])

        last = res['Contents'][-1]['Key']
        res = s3.list_objects_v2(Bucket=BUCKET_NAME,StartAfter=last)
    return keys

def download_whole_dataset(s3):
    image_dir = "./data/images/"
    label_dir = "./data/profiles/"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    keys = get_s3_keys(s3)
    start_time = time.time()
    for idx, key in enumerate(keys):
        download(s3, BUCKET_NAME, key, key)
        if idx % 100 == 0:
            print("{} download is completed -- {:.2f}".format(idx, time.time()-start_time))
            start_time = time.time()
    return image_dir, label_dir
