import pandas as pd
from glob import glob


columns = ['id', 'gender', 'race', 'age', 'path', 'label']

folder_list = glob('/opt/ml/input/data/train/test_images')
print(folder_list[0])
