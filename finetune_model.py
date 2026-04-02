import argparse
import pathlib
import random
import sys
from sys import platform
import socket
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from BYOL.feature_model import get_backbone
from BYOL.test_byol import  get_stl10_data_loaders

from SimCLR.models.resnet_simclr import FeatureModelSimCLR

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_supervised_oct_data_loaders, build_image_root, RandomWrapAround, NormTransform, get_cross_valid_splits

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

        # Load weights
        state_dict = None
        if self.ckp_file is not None:
            print(f"Loading weights from {self.ckp_file}...")
            state_dict = torch.load(self.ckp_file, map_location=self.args.device)
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']

        # Define model
        if self.args.approach == 'byol':
            self.model, _ = get_backbone(args.arch, True)
            # Change first layer to take grayscale image
            if args.img_channel == 1:
                self.model = utils.update_backbone_channel(self.model, args.img_channel)
        elif self.args.approach == 'simclr':
            if state_dict is not None:
                last_layer = list(state_dict.keys())[-1]
                out_dim = state_dict[last_layer].shape[0]
            else:
                out_dim = len(args.labels_dict)
            self.model = FeatureModelSimCLR(arch=args.arch, out_dim=out_dim, pretrained=False,
                                            img_channel=args.img_channel)
            # Skip changing the first layer, already done in FeatureModelSimCLR
        self.save_folder = self.args.save_folder
        self.finetune_best_weights_path = self.args.save_folder.joinpath(f'finetune_best_loss_{args.dataset_name}.pt')
        if self.finetune_best_weights_path.exists():
            self.finetune_best_weights = torch.load(self.finetune_best_weights_path, map_location=self.args.device)
        else:
            self.finetune_best_weights = None

        # Load weights
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)

        # Update classification head
        num_outputs = 1 if len(self.args.labels_dict.keys()) == 2 else len(self.args.labels_dict.keys())
        if self.args.approach == 'byol':
            self.model = utils.set_classifier_head(self.model, num_outputs)
        elif self.args.approach == 'simclr':
            self.model = self.model.backbone
            self.model = utils.set_classifier_head_SimCLR(self.model, num_outputs)
        self.model.to(args.device)

    def finetune(self, train_loader, valid_loader, criterion, opt, scheduler=None):
        best_epoch = 0
        best_valid_loss = 1e6
        for epoch in range(self.args.epochs):
            print(f"\n================================\n"
                  f"Epoch {epoch}")
            if (epoch - best_epoch) >= self.args.patience+1:
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
                if labels.shape[-1] > 1:
                    # One-hot → class index
                    labels = torch.argmax(labels, dim=1)
                else:
                    # labels = labels.unsqueeze(1)
                    pass
                opt.zero_grad()
                outputs = self.model(images)
                batch_loss = criterion(outputs, labels)
                batch_loss.backward()
                opt.step()
                avg_epoch_train_loss.append(batch_loss)
            avg_epoch_train_loss = float(torch.mean(torch.stack(avg_epoch_train_loss)).cpu().detach().numpy())
            print(f"Average epoch train loss: {avg_epoch_train_loss}")

            # Get validation loss
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in tqdm(valid_loader, desc='Validation'):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    if labels.shape[-1] > 1:
                        # One-hot → class index
                        labels_idx = torch.argmax(labels, dim=1)
                    else:
                        labels_idx = labels
                    outputs = self.model(images)
                    batch_loss = criterion(outputs, labels_idx)
                    avg_epoch_valid_loss.append(batch_loss)
                    if labels.shape[-1] > 1:
                        preds = torch.argmax(outputs, dim=1)
                    else:
                        preds = (outputs > 0.5).to(torch.float16)
                    correct += (preds == labels_idx).sum().item()
                    total += labels.size(0)
                avg_epoch_valid_loss = float(torch.mean(torch.stack(avg_epoch_valid_loss)).cpu().detach().numpy())
                print(f"Average epoch valid loss: {avg_epoch_valid_loss}")
                print(f"Epoch valid accuracy: {correct/total}")


            if scheduler is not None:
                scheduler.step(avg_epoch_valid_loss)

            if avg_epoch_valid_loss < best_valid_loss:
                print(f'New best loss achieved @ epoch {epoch}: {avg_epoch_valid_loss}')
                best_epoch = epoch
                best_valid_loss = avg_epoch_valid_loss
                self.finetune_best_weights = self.model.state_dict()
                torch.save(self.model.state_dict(), self.finetune_best_weights_path)

    def test(self, test_loader):
        # Update model weights
        print(f'Loading best model weights...')
        self.model.load_state_dict(self.finetune_best_weights, strict=False)
        preds_all = []
        labels_all = []
        self.model.eval()
        print(f'Getting test set predictions...')
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.args.device)
                outputs = self.model(images)
                if labels.shape[-1] > 1:
                    # One-hot → class index
                    preds = torch.argmax(outputs, dim=1)
                    labels = torch.argmax(labels, dim=1)
                else:
                    preds = (outputs > 0.5).to(torch.float16)
                    labels = labels
                preds_all.append(preds)
                labels_all.append(labels)
        preds_all = torch.concat(preds_all, dim=0).detach().to('cpu')
        labels_all = torch.concat(labels_all, dim=0).detach().to('cpu')
        return preds_all, labels_all


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
        print(f"socket name: {socket.gethostname()}")
        if 'hpc' in socket.gethostname() or 'u00' in socket.gethostname():
            dataset_path = pathlib.Path(configs['finetune']['dataset_path_hpc'])
        else:
            dataset_path = pathlib.Path(configs['finetune']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['finetune']['dataset_path_windows'])
    args.sequential_split = configs['finetune']['sequential_split']
    labels = configs['data']['labels']
    trajectories = configs['data']['trajectories']
    ascan_per_group = configs['data']['ascan_per_group']
    overwrite_labels_path = pathlib.Path(configs['data']['overwrite_labels'])
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
    if args.dataset_name == 'oct':
        new_lbl_str = f'{overwrite_labels_path.stem}_' if overwrite_labels_path is not None else ''
        traj_str = f"{''.join([t.capitalize() for t in trajectories])}_" if len(trajectories) < 3 else ''
    else:
        new_lbl_str = ''
        traj_str = ''
    args.map_df_paths = {
        split: args.data.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{new_lbl_str}{traj_str}{ascan_per_group}scans.csv")
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
    args.ratio_sup = configs['finetune']['ratio_sup']
    args.ascan_per_group = ascan_per_group
    if overwrite_labels_path is not None:
        labels = pd.read_csv(args.map_df_paths['train'])['label'].unique().tolist()
        args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
        num_cluster_dict['oct'] = len(labels)

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
    if (platform == "linux" or platform == "linux2") and ('hpc' in socket.gethostname() or 'u00' in socket.gethostname()):
        print(f"socket name: {socket.gethostname()}")
        args.save_folder = pathlib.Path(r'/fibus/fs0/14/cab8351/OCT_classification').joinpath(approach_folder).joinpath(f'weights_{args.arch}')
    else:
        args.save_folder = pathlib.Path().resolve().joinpath(approach_folder).joinpath(f'weights_{args.arch}')
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)
    print(f"Saving weights to: {args.save_folder}")

    if args.approach == 'byol':
        chkpt_file = list(args.save_folder.glob('byol_best_loss*.pt'))[-1]
    elif args.approach == 'simclr':
        chkpt_file = list(args.save_folder.glob('checkpoint_best*.pt'))[-1]

    # Set all random seeds
    print("Setting random seed...")
    utils.set_random_seed(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.dataset_name == 'oct_clinical':
        # Generate cross-validation split
        cv_splits = get_cross_valid_splits(args, k=3)
        # cv_splits = [None]
    else:
        cv_splits = [None]

    # Create train and test sets
    for i, cv_split in enumerate(cv_splits):
        if len(cv_splits) == 1:
            cv_split_str = f'_split{i}'
        else:
            print("================================")
            print(f"Split {i}")
            print(cv_split)
            cv_split_str = f'_split{i}'
        if 'oct' in args.dataset_name:
            # train_aug = [v2.RandomEqualize(p=0.98)]
            train_aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomEqualize(p=0.5),
                RandomWrapAround(dim=-1, p=1.0),
                RandomWrapAround(dim=-2, p=1.0)
            ]
            test_aug = [
                transforms.RandomEqualize(p=0.0),
            ]
            if args.dataset_name == 'oct_clinical':
                # train_aug = train_aug + [transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),]
                train_aug = train_aug + [
                    # v2.RandomAdjustSharpness(sharpness_factor=3, p=0.3),
                    v2.RandomAutocontrast(p=0.5),
                ]
            train_loader, valid_loader, test_loader = get_supervised_oct_data_loaders(args.data, args, args.batch_size,
                                                                           train_aug=train_aug,
                                                                           test_aug=test_aug,
                                                                           mean=mean[args.dataset_name],
                                                                           std=std[args.dataset_name],
                                                                           ratio_sup=args.ratio_sup,
                                                                           shuffle=True,
                                                                           seq_split=args.sequential_split,
                                                                           overwrite_split=cv_split)
        else:
            train_loader, test_loader = get_stl10_data_loaders(args.data, args.batch_size, shuffle=False,
                                                               download=False)

        # Define model
        model = SupervisedModel(args, chkpt_file)

        # Finetune weights
        print(f"Finetune model")
        if args.dataset_name == 'oct_clinical':
            # Define pos_weights
            # https://www.codegenes.net/blog/pytorch-bcewithlogitsloss-pos_weight/#handling-class-imbalance
            class_counts = train_loader.dataset.map_df.groupby('label').agg(img_count=('img_relative_path', 'count'))
            pos_weights = torch.Tensor([class_counts.loc[1.0, 'img_count'] / class_counts.loc[0.0, 'img_count']]).to(
                args.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        opt = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=0.5,
            patience=2
        )
        model.finetune(train_loader=train_loader, valid_loader=valid_loader, criterion=criterion, opt=opt,
                       scheduler=scheduler)

        # Get test set performance
        test_preds, test_labels = model.test(test_loader)

        # Save predictions
        preds_df = pd.DataFrame.from_dict({'pred': test_preds.squeeze(-1), 'pred_labels': test_labels.squeeze(-1)}, orient='columns')
        preds_df = pd.concat([test_loader.dataset.map_df.copy(), preds_df], axis=1)
        assert len(preds_df[preds_df['pred_labels'] == preds_df['label']]) == len(preds_df)
        preds_df = preds_df.drop(columns=['pred_labels'])
        preds_path = f'preds_{args.dataset_name}_{int(args.ratio_sup * 100)}p{cv_split_str}.csv'
        preds_df.to_csv(args.save_folder.joinpath(preds_path), index=False)

        # Calculate metrics
        print(f"Test set results using {args.arch} backbone \n(Finetune from {args.approach}, with {args.ratio_sup*100}% of {args.dataset_name}):")
        report = classification_report(test_labels, test_preds, target_names=labels, digits=4, zero_division=np.nan)
        print(report)

        # Get confusion matrix display
        cm = confusion_matrix(test_labels, test_preds)
        cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        cm_plot.plot()
        plt.title('Confusion matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(args.save_folder.joinpath(f'confusion_matrix_{args.dataset_name}{cv_split_str}.png'))
        plt.show()


if __name__ == "__main__":
    main()
