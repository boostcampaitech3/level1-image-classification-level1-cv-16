from dataset.kfold import KFold
import numpy as np
import torch
import pandas as pd
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from model import criterion_entrypoint
from dataset import MaskDataset
from dataset.transform import *
from utils import *


csv_path = '../data/train/test_train.csv'
img_path = '../data/train/test_images'

folds = KFold(csv_path=csv_path, img_path=img_path, n_splits=5, random_state=1004)
 
train_df, val_df = folds[0]

[test]=train_df.index[train_df["path"]=='../data/train/test_images/000004_male_Asian_54/mask1.jpg'].tolist()
print(train_df['age'].iloc[1])
train_df['age'].iloc[1]=train_df['age'].iloc[1]*2
print(train_df['age'].iloc[1])
# train_df_mask, val_df_mask = pd.DataFrame(), pd.DataFrame()
# train_df_inc, val_df_inc = pd.DataFrame(), pd.DataFrame()
# train_df_not, val_df_not = pd.DataFrame(), pd.DataFrame()

# for i in range(len(train_df)):
#     if train_df.loc[i]['mask'] == 0:
#         train_df_mask=pd.concat([train_df_mask,train_df.loc[[i]]])
#     elif train_df.loc[i]['mask'] == 1:
#         train_df_inc=pd.concat([train_df_inc,train_df.loc[[i]]])
#     else:
#         train_df_not=pd.concat([train_df_not,train_df.loc[[i]]])
        
# for i in range(len(val_df)):
#     if val_df.loc[i]['mask'] == 0:
#         val_df_mask=pd.concat([val_df_mask,val_df.loc[[i]]])
#     elif val_df.loc[i]['mask'] == 1:
#         val_df_inc=pd.concat([val_df_inc,val_df.loc[[i]]])
#     else:
#         val_df_not=pd.concat([val_df_not,val_df.loc[[i]]])


# train_df.to_csv('../data/train/train_df.csv')
# val_df.to_csv('../data/train/val_df.csv')
# train_df_mask.to_csv('../data/train/train_df_mask.csv')
# train_df_inc.to_csv('../data/train/train_df_inc.csv')
# train_df_not.to_csv('../data/train/train_df_not.csv')
# val_df_mask.to_csv('../data/train/val_df_mask.csv')
# val_df_inc.to_csv('../data/train/val_df_inc.csv')
# val_df_not.to_csv('../data/train/val_df_not.csv')