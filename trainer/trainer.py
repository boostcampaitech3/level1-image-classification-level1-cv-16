import multiprocessing
import os
from importlib import import_module

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



class Trainer:
    def __init__(self, config, csv_path, img_path, save_dir):
        self.csv_path = csv_path
        self.img_path = img_path
        self.save_dir = increment_path(os.path.join(save_dir, config.name))
        makedirs(self.save_dir)
        increment_name = self.save_dir.split('/')[-1]
        self.wandb = Wandb(**config.wandb, name=increment_name, config=config)
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

    def train(self, config, pseudo_df=None):

        folds = KFold(csv_path=self.csv_path, img_path=self.img_path, **config.fold)

        train_df, val_df = folds[0] # 나중에는 fold 다 돌면서 진행해도 됨

        if pseudo_df:
            train_df = pd.concat([pseudo_df, train_df])

        # -- transform
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        train_transform = transform_module(augment=True, **config.augmentation.args)
        test_transform = transform_module(augment=True, **config.augmentation.args)

        train_set = MaskDataset(train_df, transform=train_transform, target=config.target)
        val_set = MaskDataset(val_df, transform=test_transform, target=config.target)

        train_loader = DataLoader(
            train_set, 
            num_workers=multiprocessing.cpu_count()//2,
            pin_memory=self.use_cuda,
            **config.data_loader
        )
        val_loader = DataLoader(
            val_set,
            num_workers=multiprocessing.cpu_count()//2,
            pin_memory=self.use_cuda,
            **config.val_data_loader
        )

        num_class = target_to_class_num(config.target)

        # -- model
        model_module = getattr(import_module("model"), config.model.name)  # default: BaseModel
        model = model_module(num_classes=num_class).to(self.device)
        model = torch.nn.DataParallel(model).cuda()

        # -- loss & metric
        loss_module = getattr(import_module("model"), criterion_entrypoint(config.loss.name))
        criterion = loss_module(**config.loss.args)
        opt_module = getattr(import_module("torch.optim"), config.optimizer.type)
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), **config.optimizer.args)

        # -- lr_scheduler
        sch_module = getattr(import_module("torch.optim.lr_scheduler"), config.lr_scheduler.type)
        scheduler = sch_module(optimizer, **config.lr_scheduler.args)

        best_val_acc = 0
        best_val_loss = np.inf
        best_f1 = 0

        self.wandb.watch(model)
        for epoch in range(config.epochs):
            # train loop
            model.train()

            loss_value = 0
            matches = 0

            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch[0].to(self.device), train_batch[1].to(self.device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                
                if (idx + 1) % config.log_interval == 0:
                    train_loss = loss_value / config.log_interval
                    train_acc = matches / config.data_loader.batch_size / config.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{config.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    loss_value = 0
                    matches = 0
                    self.wandb.write_log(train_loss, train_acc, current_lr)

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                label_list = []
                pred_list = []

                for val_batch in val_loader:
                    inputs, labels = val_batch[0].to(self.device), val_batch[1].to(self.device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()

                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    label_list.append(labels)
                    pred_list.append(preds)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                f1 = f1_score(torch.cat(label_list).to('cpu'), torch.cat(pred_list).to('cpu'), average = 'macro')
                best_val_loss = min(best_val_loss, val_loss)

                scheduler.step(val_loss)

                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{self.save_dir}/best_acc.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{self.save_dir}/last_acc.pth")
                if f1 > best_f1:
                    print(f"New best model for f1 : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{self.save_dir}/best_f1.pth")
                    best_f1 = f1
                torch.save(model.module.state_dict(), f"{self.save_dir}/last_f1.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                    f"F1 Score : {best_f1:.3f}\n"
                )
            torch.save(model.module.state_dict(), f"{self.save_dir}/epoch{epoch}.pth")
            self.wandb.write_log2(epoch, current_lr, val_loss, val_acc, f1)
        self.wandb.write_log3(best_val_acc, best_f1)
        