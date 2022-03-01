import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
import random
import yaml
from utils import DictAsMember
from trainer import Trainer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="train_config.yaml", type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DictAsMember(config)
    seed_everything(config.seed)

    trainer = Trainer(config, **config.trainer)
    trainer.train(config.train)