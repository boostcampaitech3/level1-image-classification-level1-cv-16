import os 
from pathlib import Path
import glob
import re
import torch
from torch.utils.data.sampler import WeightedRandomSampler

class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

def target_to_class_num(target):
    """
    학습 대상에 따라 다른 class num을 반환
    """
    if target == 'label':
        print("전체 클래스를 대상으로 학습이 진행됩니다.")
        return 18
    elif target == 'age':
        print("나이를 대상으로 학습이 진행됩니다.")
        return 3
    elif target == 'gender':
        print("성별을 대상으로 학습이 진행됩니다.")
        return 2
    else: # mask
        print("마스크를 대상으로 학습이 진행됩니다.")
        return 3

def makedirs(path): 
    try:    
        os.makedirs(path) 
    except OSError: 
        if not os.path.isdir(path): 
            raise

def create_directory(directory):
    try: 
        if not os.path.exists(directory):
            os.makedirs(directory) 
    except OSError:
        print("Error: Failed to create the directory.")

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_sampler_weights(target_df):
    class_counts = target_df.value_counts().to_list()
    num_samples = sum(class_counts)
    labels = target_df.to_list()

    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]

    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    return sampler