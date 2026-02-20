import argparse
import pathlib
import random
import sys
from argparse import Namespace
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from addict import Dict
from torchvision import models

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score, normalized_mutual_info_score, classification_report

from BYOL.feature_model import get_backbone
from BYOL.test_byol import get_oct_data_loaders, get_stl10_data_loaders

from SimCLR.models.resnet_simclr import FeatureModelSimCLR

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset, build_image_root

img_size_dict = {'stl10': 96,
                 'cifar10': 32,
                 'cifar100': 32}
num_cluster_dict = {'stl10': 10,
                    'cifar10': 10,
                    'cifar100': 100}
mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
# mean['oct'] = [43.51, 43.51, 43.51]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
# std['oct'] = [24.98, 24.98, 24.98]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)

class SupervisedModel(object):
    def __init__(self, args, ckp_file):
        self.ckp_file = ckp_file
        self.args = args
        # Define model
        if self.args.approach == 'byol':
            self.model, _ = get_backbone(args.arch, True)
            # Change first layer to take grayscale image
            if args.img_channel == 1:
                self.model = utils.update_backbone_channel(self.model, args.img_channel)
        elif self.args.approach == 'simclr':
            # Set out_dim to 4 due to pre-training on 4 classes
            self.model = FeatureModelSimCLR(arch=args.arch, out_dim=4, pretrained=False,
                                            img_channel=args.img_channel)
            # Skip changing the first layer, already done in FeatureModelSimCLR
        self.save_folder = self.args.save_folder
        self.finetune_best_weights_path = self.args.save_folder.joinpath(f'finetune_best_loss.pt')
        if self.finetune_best_weights_path.exists():
            self.finetune_best_weights = torch.load(self.finetune_best_weights_path, map_location=self.args.device)
        else:
            self.finetune_best_weights = None

        # Load weights
        print(f"Loading weights from {self.ckp_file}...")
        state_dict = torch.load(self.ckp_file, map_location=self.args.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)

        # Update classification head
        num_outputs = 1 if len(self.args.labels_dict.keys()) == 2 else len(self.args.labels_dict.keys())
        if self.args.approach == 'byol':
            self.model = utils.set_classifier_head(self.model, num_outputs)
        elif self.args.approach == 'simclr':
            self.model = self.model.backbone
            self.model = utils.set_classifier_head_SimCLR(self.model, num_outputs)
        self.model.to(args.device)

    def finetune(self, train_loader, valid_loader, criterion, opt):
        best_epoch = 0
        best_valid_loss = 1e6
        for epoch in range(self.args.epochs):
            print(f"\n================================\n"
                  f"Epoch {epoch}")
            if (epoch - best_epoch) >= self.args.patience:
                print(f'Loss has not improved for {self.args.patience} epochs. Training has stopped')
                print(f'Best loss was {best_valid_loss} @ epoch {best_epoch}')
                break
            avg_epoch_train_loss = []
            avg_epoch_valid_loss = []
            self.model.train()
            # with torch.autocast(device_type=f'cuda:{self.args.gpu_index}', dtype=torch.float16):
            for images, labels in tqdm(train_loader, desc='Training'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                opt.zero_grad()
                preds = self.model(images)
                batch_loss = criterion(preds, labels)
                batch_loss.backward()
                opt.step()
                avg_epoch_train_loss.append(batch_loss)
            avg_epoch_train_loss = float(torch.mean(torch.stack(avg_epoch_train_loss)).cpu().detach().numpy())
            print(f"Average epoch train loss: {avg_epoch_train_loss}")

            # Get validation loss
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(valid_loader, desc='Validation'):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    preds = self.model(images)
                    batch_loss = criterion(preds, labels)
                    avg_epoch_valid_loss.append(batch_loss)
                avg_epoch_valid_loss = float(torch.mean(torch.stack(avg_epoch_valid_loss)).cpu().detach().numpy())
                print(f"Average epoch valid loss: {avg_epoch_valid_loss}")

            if avg_epoch_valid_loss < best_valid_loss:
                print(f'New best loss achieved @ epoch {epoch}: {avg_epoch_valid_loss}')
                best_epoch = epoch
                best_valid_loss = avg_epoch_valid_loss
                self.finetune_best_weights = self.model.state_dict()
                torch.save(self.model.state_dict(), self.finetune_best_weights_path)

    def test(self, test_loader):
        # Update model weights
        print(f'Loading best model weghts...')
        self.model.load_state_dict(self.finetune_best_weights, strict=False)
        preds_all = []
        labels_all = []
        self.model.eval()
        print(f'Getting test set predictions...')
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.args.device)
                pred = self.model(images)
                preds_all.append(pred)
                labels_all.append(labels)
        preds_all = torch.concat(preds_all, dim=0).detach().to('cpu')
        labels_all = torch.concat(labels_all, dim=0).detach().to('cpu')
        return preds_all, labels_all


def get_stl10_data_loaders(root_path, batch_size=128, shuffle=False, download=False):
    train_dataset = datasets.STL10(root_path, split='train', download=download,
                                   transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10(root_path, split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_oct_data_loaders(root_path:pathlib.Path, args: argparse.Namespace, batch_size:int, shuffle=False):
    img_transforms = [transforms.ToTensor(),
                      transforms.Resize((args.img_reshape, args.img_reshape)),
                      transforms.Normalize(mean=mean[args.dataset_name],
                                           std=std[args.dataset_name])]
    if args.img_channel == 1:
        img_transforms.append(transforms.Grayscale())
    img_transforms = transforms.Compose(img_transforms)
    train_dataset = OCTDataset(root_path, 'train',
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False, # args.use_iipp,
                               num_same_area=-1,
                               transforms=img_transforms,
                               pre_sample=args.dataset_sample)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    valid_dataset = OCTDataset(root_path, 'valid',
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False,  # args.use_iipp,
                               num_same_area=-1,
                               transforms=img_transforms,
                               pre_sample=args.dataset_sample)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = OCTDataset(root_path, 'test',
                              args.map_df_paths, args.labels_dict,
                              ch_in=args.img_channel,
                              sample_within_image=args.sample_within_image,
                              use_iipp=False,
                              num_same_area=-1,
                              transforms=img_transforms,
                              pre_sample=args.dataset_sample)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, valid_loader, test_loader


def main():
    args = parser.parse_args()
    if args.config_path is None:
        args.config_path = pathlib.Path('config.yaml')
    config_file = pathlib.Path(args.config_path)

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    configs = utils.load_configs(config_file)
    if platform == "linux" or platform == "linux2":
        dataset_path = pathlib.Path(configs['finetune']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['finetune']['dataset_path_windows'])
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    args.dataset_name = configs['finetune']['dataset_name']
    if 'oct' in args.dataset_name:
        mean[args.dataset_name] = 3 * [configs['data']['img_mean'] / 255]
        std[args.dataset_name] = 3 * [configs['data']['img_std'] / 255]
        img_size_dict[args.dataset_name] = (512, ascan_per_group)
        num_cluster_dict[args.dataset_name] = len(labels)

    # Dataset
    # args.dataset_name = configs['finetune']['dataset_name']
    if args.dataset_name == 'oct':
        folder_name = 'OCT_lab_data'
    elif args.dataset_name == 'oct_clinical':
        folder_name = 'OCT_clinical_data'
        labels = ['Healthy', 'Lesion']
        num_cluster_dict[args.dataset_name] = len(labels)
        # Update pre processing
        pre_processing['no_noise'] = False  # M-Scans have already been cropped to remove noise
        pre_processing['ascan_sampling'] = 1
    else:
        folder_name = args.dataset_name
    args.data = pathlib.Path(dataset_path).joinpath(folder_name)
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {args.data.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: args.data.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    args.img_channel = configs['finetune']['img_channel']
    if 'oct' not in args.dataset_name:
        args.img_channel = 3
    args.sample_within_image = configs['finetune']['sample_within_image']
    args.img_reshape = configs['finetune']['img_reshape']
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512  # BYOL requires square images, so all images will be reshaped to 512x512
    args.use_iipp = configs['finetune']['use_iipp']
    args.ascan_per_group = ascan_per_group

    # Training params
    args.seed = configs['training']['random_seed']
    args.dataset_sample = configs['finetune']['dataset_sample']
    args.approach = configs['finetune']['approach']
    args.arch = configs['finetune']['arch']
    args.workers = configs['finetune']['num_workers']
    args.epochs = configs['finetune']['max_epochs']
    args.batch_size = configs['finetune']['batch_size']
    args.lr = configs['finetune']['lr']
    args.disable_cuda = configs['finetune']['disable_cuda']
    args.out_dim = num_cluster_dict[args.dataset_name]
    args.gpu_index = configs['finetune']['gpu_index']
    args.patience = configs['finetune']['patience']
    if args.approach == 'byol':
        approach_folder = 'BYOL'
    elif args.approach == 'simclr':
        approach_folder = 'SimCLR'
    args.save_folder = pathlib.Path().resolve().joinpath(approach_folder).joinpath(f'weights_{args.arch}')
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)

    if args.approach == 'byol':
        chkpt_file = list(args.save_folder.glob('byol_best_loss*.pt'))[-1]
    elif args.approach == 'simclr':
        chkpt_file = list(args.save_folder.glob('checkpoint_best*.pt'))[-1]

    # Set all random seeds
    print("Setting random seed...")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Create train and test sets
    if 'oct' in args.dataset_name:
        train_loader, valid_loader, test_loader = get_oct_data_loaders(args.data, args, args.batch_size,
                                                                       shuffle=False)
    else:
        train_loader, test_loader = get_stl10_data_loaders(args.data, args.batch_size, shuffle=False,
                                                           download=False)

    # Define model
    model = SupervisedModel(args, chkpt_file)

    # Finetune weights
    print(f"Finetune model")
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.model.parameters(), lr=args.lr)
    model.finetune(train_loader=train_loader, valid_loader=valid_loader, criterion=criterion, opt=opt)

    # Get test set performance
    test_logits, test_oh_labels = model.test(test_loader)
    # Convert from logits to predictions
    if len(labels) > 2:
        test_probs = F.softmax(test_logits, dim=1)
        test_preds = torch.argmax(test_probs, dim=1)
        test_labels = torch.argmax(test_oh_labels, dim=1)
    else:
        test_probs = F.sigmoid(test_logits)
        test_preds = test_probs > 0.5
        test_labels = test_oh_labels

    # Calculate metrics
    report = classification_report(test_labels, test_preds, target_names=labels)
    print(report)


if __name__ == "__main__":
    main()
