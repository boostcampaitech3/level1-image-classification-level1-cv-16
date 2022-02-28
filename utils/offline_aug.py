import cv2
import shutil
import pandas as pd
import glob
import os
from PIL import Image
import random
import numpy as np

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

        

df = pd.read_csv('./data/train/train.csv')

pat = './data/train/test_images/' # 기존 image 폴더를 복사한 폴더 경로

if not os.path.exists(pat):
    shutil.copytree('./data/train/images', pat)
else:
    print("test_images were already created")
shutil.copyfile("./data/train/train.csv", "./data/train/test_train.csv")
df_test = pd.read_csv('./data/train/test_train.csv') # 기존 train.csv를 복사한 경로

# 이미지 폴더를 복사하여 사용
print("Image dir copy complete!")
aug_df = pd.DataFrame(None, columns=['id', 'gender', 'race', 'age', 'path'])

update = df_test
for i in range(len(df)):
    if df['age'][i] == 60:
        img_list = glob.glob(pat+df['path'][i]+'/*')
        
        update = update.append({'id' : str(int(update.iloc[-1]['id'])+1).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_inverted'}, ignore_index=True)
        update = update.append({'id' : str(int(update.iloc[-1]['id'])+2).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_rotated'}, ignore_index=True)
        update = update.append({'id' : str(int(update.iloc[-1]['id'])+3).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_noise'}, ignore_index=True)          
        update.to_csv('./data/train/test_train.csv', header=True, index=True)

        aug_df = aug_df.append({'id' : str(int(update.iloc[-1]['id'])+1).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_inverted'}, ignore_index=True)
        aug_df = aug_df.append({'id' : str(int(update.iloc[-1]['id'])+2).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_rotated'}, ignore_index=True)
        aug_df = aug_df.append({'id' : str(int(update.iloc[-1]['id'])+3).zfill(6),
                        'gender' : df_test['gender'][i],
                        'race' : df_test['race'][i], 'age' : 60,
                        'path' : df_test['path'][i]+'_noise'}, ignore_index=True)          

        try:
            target = "/".join(img_list[0].split('/')[:-1])
        except IndexError:
            print(df['path'][i])
            print(img_list)
        createFolder(target+'_inverted')
        createFolder(target+'_rotated')
        createFolder(target+'_noise')
        
        for im in img_list:
            img = Image.open(im)
              
            # 좌우반전
            trs_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            trs_img.save(target+'_inverted/'+im.split('/')[-1].split('.')[0]+'_inverted.jpg')
            
            # 회전
            trs_img = img.rotate(random.randrange(-20, 20))
            trs_img.save(target+'_rotated/'+im.split('/')[-1].split('.')[0]+'_rotated.jpg')

            # noise
            img2 = cv2.imread(im)
            row,col,ch= (img.size[1], img.size[0], 3)
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy_array = img2 + gauss
            noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
            noisy_image.save(target+'_noise/'+im.split('/')[-1].split('.')[0]+'_noise.jpg')

aug_df['age'] = aug_df['age'].astype(int)
aug_df.to_csv('./data/train/aug_train.csv', index=False)        