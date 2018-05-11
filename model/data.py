import os
import time
import glob
import random
import re

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
    get_dataset
    : local 혹은 S3에서 데이터를 가져와 full dataset을 만드는 메소드

    관련 메소드

    1. 디렉토리 처리 메소드
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
def get_dataset(image_dir="./data/images/", label_dir="./data/profiles", input_size=(48,48), is_train=True):
    # image_dir와 label_dir 내 파일 이름은 동일
    # 그러므로 intersection된 파일 이름이 우리가 학습할 데이터
    if not os.path.exists(image_dir) or not os.path.exists(label_dir) or len(os.listdir(image_dir)) <= 100 or len(os.listdir(label_dir)) <= 100:
        print("Start to Download---")
        s3 = boto3.client('s3')
        image_dir, label_dir = download_whole_dataset(s3)

    fname_list = list(get_fnameset(image_dir) & get_fnameset(label_dir))
    random.shuffle(fname_list)

    # 데이터셋 다운로드 받기
    pool = Pool(processes=multiprocessing.cpu_count())
    start_time  = time.time()
    result = pool.map(partial(data_workers,input_size,is_train,image_dir,label_dir),
        np.array_split(fname_list,multiprocessing.cpu_count()))
    x,y = list(zip(*result))
    xdata = np.concatenate(x,axis=0)
    ydata = np.concatenate(y,axis=0)
    print("consumed---{}".format(time.time()-start_time))
    pool.close()

    return xdata, ydata

def data_workers(input_size, is_train, image_dir, label_dir, fname_list):
    cut_size = int(input_size[0] * 4/3), int(input_size[1] * 4/3)
    images = []
    labels = []
    if isinstance(fname_list,np.ndarray):
        fname_list = list(fname_list)
    while len(fname_list) > 0:
        fname = fname_list.pop()
        image_path = os.path.join(image_dir,fname)
        label_path = os.path.join(label_dir,fname)

        # read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path,0)

        # normalization
        image = image/255.
        label = label/255

        if is_train:
            image = cv2.resize(image,cut_size)
            label = cv2.resize(label,cut_size)
        else :
            image = cv2.resize(image,input_size)
            label = cv2.resize(label,input_size)
            label = (label>0.7).astype(int)
            images.append(image)
            labels.append(label)
            continue

        # data augmentation
        x,y = apply_rotation(image,label)
        x,y = apply_rescaling(x,y)
        x,y = apply_flip(x,y)
        x,y = apply_random_crop(x,y, input_size)
        x = random_noise(x,mode='gaussian',mean=0,var=0.001)

        # adjust the range of value
        x = np.clip(x,0.,1.)
        y = (y>0.7).astype(int)

        images.append(x)
        labels.append(y)

    xdata = np.stack(images, axis=0)
    ydata = np.stack(labels, axis=0)
    return xdata, ydata

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
