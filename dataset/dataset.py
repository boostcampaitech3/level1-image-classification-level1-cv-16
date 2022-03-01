import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class MaskDataset(Dataset):
    def __init__(self, df, transform = None, target = 'label'):
        self.df = df
        self.transform = transform
        self.target = target

    def __getitem__(self, idx):
        image_path = self.df['path'].iloc[idx]
        if self.target == "gender_age":
            label = self.df['label'].iloc[idx] % 6
        else:
            label = self.df[self.target].iloc[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.df)


# inference에 사용됨
class MaskTestDataset(Dataset):
    def __init__(self, df, img_path, transform = None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.df['ImageID'].iloc[idx]
        image_path = os.path.join(self.img_path, image_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.df)