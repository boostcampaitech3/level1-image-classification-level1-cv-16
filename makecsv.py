from glob import glob
import os
import numpy as np
import pandas as pd


columns = ['id', 'gender', 'race', 'age', 'mask', 'path', 'label']
data = []

folder_list = glob('/opt/ml/input/data/train/test_images/*')
folder_list.sort()

for folder in folder_list:
    folder_name = folder.split('/')[-1]
    f = folder_name.split('_')
    id, gd, race, ag = f[0], f[1], f[2], int(f[3])
    if gd == 'male':
        gender = 0
    else:
        gender = 1
    if ag < 30:
        age = 0
    elif ag < 60:
        age = 1
    else:
        age = 2
    data.append([id, gender, race, age, 0, folder, 0])
    # images = [a for a in os.listdir(folder) if '_' not in a]
    # for img in images:
    #     if 'mask' in img:
    #         mask = 0
    #     elif 'incorrect' in img:
    #         mask = 1
    #     else:
    #         mask = 2
        
    #     label = mask*6 + gender*3 + age
    #     path = os.path.join(folder, img)
    #     data.append([id, gender, race, age, mask, path, label])
        
data = np.array(data)
df = pd.DataFrame(data, columns=columns)
df.to_csv('/opt/ml/level1-image-classification-level1-cv-16/train_all.csv')