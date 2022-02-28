import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
import glob

from RandAugment import RandAugment # RandAugment

# offline data augmentation
import cv2 # !apt-get -y install libgl1-mesa-glx
import shutil


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

        

df = pd.read_csv('/opt/ml/input/data_copy/train/test_train.csv')
pat = '/opt/ml/input/data_copy/train/RandAug_images/' # 기존 image 폴더를 복사한 폴더 경로
df_test = pd.read_csv('/opt/ml/input/data_copy/train/test_train.csv') # 기존 train.csv를 복사한 경로

# 이미지 폴더를 복사하여 사용
shutil.copytree('/opt/ml/input/data_copy/train/images', pat)

ra1 = RandAugment(3,0.5)
ra2 = RandAugment(3,1)

update = df_test
for i in range(len(df)):
    # if df['age'][i] >= 60:
        img_list = glob.glob(pat+df['path'][i]+'/*')
        print(f"/n update {df['path'][i]}")
        update = update.append({'id' : str(int(update.iloc[-1]['id'])+1).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : df_test['age'][i],
                        'path' : df_test['path'][i]+'RandAug1'}, ignore_index=True)
        update = update.append({'id' : str(int(update.iloc[-1]['id'])+2).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : df_test['age'][i],
                        'path' : df_test['path'][i]+'RandAug2'}, ignore_index=True)
        update.to_csv('/opt/ml/input/data_copy/train/RandAug_train.csv', header=True, index=True)
        
        try:
            target = "/".join(img_list[0].split('/')[:-1])
        except IndexError:
            print(df['path'][i])
            print(img_list)
        print(f"creating Folder in traget: {target} \n")

        createFolder(target+'_RandAug1')
        createFolder(target+'_RandAug2')
        
        for im in img_list:
            img = Image.open(im)
              
            # RandAug1
            trs_img = ra1(img)
            trs_img.save(target+'_RandAug1/'+im.split('/')[-1].split('.')[0]+'_RandAug1.jpg')
            
            # RandAug2
            trs_img = ra2(img)
            trs_img.save(target+'_RandAug2/'+im.split('/')[-1].split('.')[0]+'_RandAug2.jpg')
