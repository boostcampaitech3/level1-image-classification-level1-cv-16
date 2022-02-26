import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import multiprocessing

import ttach as tta # TTA 라이브러리 추가

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device, args):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    if args.score == 'f1': # f1
        target_name = 'best_f1'
        model_path = os.path.join(saved_model, 'best_f1.pth')
    elif args.score == 'acc': # acc
        target_name = 'best_acc'
        model_path = os.path.join(saved_model, 'best_acc.pth')
    else: # epoch
        target_name = 'epoch' + str(args.target_epoch)
        statedict_name = 'epoch' + str(args.target_epoch) + '.pth'
        model_path = os.path.join(saved_model, target_name)

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, target_name


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    TTA_transform = tta.Compose( # augmentation for TTA
        [
            tta.HorizontalFlip(),
            # tta.VerticalFlip(),
            # tta.Scale(scales=[1, 2, 4]),
            # tta.Multiply(factors=[0.8, 1, 1.1]),        
            # tta.FiveCrops()
        ]
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model, target_name = load_model(model_dir, num_classes, device, args)

    if args.TTA == "True":
        print("TTA is applied...")
        model = tta.ClassificationTTAWrapper(model, TTA_transform)

    model.to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    model_name = args.model_dir.split('/')[-1]
    output_name = model_name + '_' + target_name + '.csv'

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 100)')
    parser.add_argument('--resize', type=tuple, default=(512, 512), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--score', type=str, default='acc', help='select the model at the point where the desired value (acc, f1, epoch)')
    parser.add_argument('--target_epoch', type=int, default=0, help='select input epoch model_dict') 
    parser.add_argument('--TTA', type=str, default="False", help="TTA (default: False")

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
