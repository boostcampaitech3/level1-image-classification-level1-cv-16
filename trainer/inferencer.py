import multiprocessing
import os
from importlib import import_module

import torch
import pandas as pd

from dataset import MaskTestDataset
from dataset.transform import *
from utils import *

def load_model(saved_model, num_classes, device, config):
    model_cls = getattr(import_module("model"), config.model.name)
    model = model_cls(
        num_classes=num_classes
    )
    if config.score == 'f1': # f1
        target_name = 'best_f1'
        model_path = os.path.join(saved_model, 'best_f1.pth')
    elif config.score == 'acc': # acc
        target_name = 'best_acc'
        model_path = os.path.join(saved_model, 'best_acc.pth')
    else: # epoch
        target_name = 'epoch' + str(config.target_epoch)
        statedict_name = 'epoch' + str(config.target_epoch) + '.pth'
        model_path = os.path.join(saved_model, statedict_name)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, target_name

class Inferencer:
    def __init__(self, model_dir, csv_path, img_path, save_dir):
        self.csv_path = csv_path
        self.img_path = img_path
        self.model_dir = model_dir
        self.save_dir = save_dir
        makedirs(self.save_dir)
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def inference(self, config):
        test_df = pd.read_csv(self.csv_path)

        # -- transform
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        test_transform = transform_module(augment=False, **config.augmentation.args)

        model, target_name = load_model(self.model_dir, 18, self.device, config)
        model.to(self.device)
        model.eval()
        dataset = MaskTestDataset(test_df, img_path=self.img_path, transform=test_transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=multiprocessing.cpu_count()//2,
            pin_memory=self.use_cuda,
            **config.data_loader
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(self.device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())
        model_name = self.model_dir.split('/')[-1]
        output_name = model_name + '_' + target_name + '.csv'

        info = pd.read_csv(self.csv_path)

        info['ans'] = preds
        info.to_csv(os.path.join(self.save_dir, output_name), index=False)
        print(f'Inference Done!')
    

    def inference_with_confidence(self, config):
        test_df = pd.read_csv(self.csv_path)

        # -- transform
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        test_transform = transform_module(augment=False, **config.augmentation.args)

        model, target_name = load_model(self.model_dir, 18, self.device, config)
        model.to(self.device)

        dataset = MaskTestDataset(test_df, img_path=self.img_path, transform=test_transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=multiprocessing.cpu_count()//2,
            pin_memory=self.use_cuda,
            **config.data_loader
        )

        print("Calculating inference results..")
        preds = []
        probs = []
        cons = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(self.device)
                pred = model(images)
                prob = torch.nn.Softmax(dim = -1)(pred)
                con, pred = prob.max(1)
                preds.extend(pred.cpu().numpy())
                probs.extend(prob.cpu().numpy())
                cons.extend(con.cpu().numpy())
        model_name = self.model_dir.split('/')[-1]

        output_name = model_name + '_' + target_name + '.csv'
        pseudo_name = model_name + '_' + target_name + '_pseudo.csv'
        ensemble_name = model_name + '_' + target_name + '_ensemble.csv'
        info = pd.read_csv(self.csv_path)

        info['ans'] = preds
        info.to_csv(os.path.join(self.save_dir, output_name), index=False)

        info['confidence'] = cons
        info.rename(columns = {"ImageID" : "path", "ans" : 'label'}, inplace = True)
        info['path'] = './data/eval/images/' + info['path']
        info = info.sort_values(by = ['confidence'], ascending = False)
        df_pseudo = info.iloc[0:len(info) // 2]
        df_pseudo.to_csv(os.path.join(self.save_dir, pseudo_name), index=False)
        
        info = pd.read_csv(self.csv_path)['ImageID']
        df_prob = pd.DataFrame(probs)
        info = pd.concat([info, df_prob], axis = 1)
        info.to_csv(os.path.join(self.save_dir, ensemble_name), index=False)

        print(f'Inference Done!')