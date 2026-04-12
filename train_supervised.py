import argparse
import pathlib
import random
import re
import sys
from argparse import Namespace
import socket
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from addict import Dict
from matplotlib.pyplot import xticks
from torchvision import models

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from BYOL.feature_model import get_backbone
from BYOL.test_byol import get_stl10_data_loaders

from finetune_model import SupervisedModel

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_supervised_oct_data_loaders, get_oct_data_loaders, build_image_root, RandomWrapAround, NormTransform, get_cross_valid_splits

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
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
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
parser.add_argument('--ratio_sup',#}{
                    help='Ratio of the dataset used for supervised training (between 0.05 and 0.2)',
                    type=float,)
parser.add_argument('--dataset_name',
                    help='Name of the dataset to use (either oct or oct_clinical)',
                    type=str)
parser.add_argument('--weight_init',
                    help='Initial model weights (either Random or DEFAULT)',
                    type=str)


class FullSupervisedModel(SupervisedModel):
    def __init__(self, args):
        super().__init__(args, None)
        self.args = args
        # Define model
        if self.args.weight_init is not None and len([w for w in ['random', 'default'] if w == self.args.weight_init.lower()]) == 1:
            pretrained = False if self.args.weight_init.lower() == 'random' else True
        else:
            pretrained = True
        self.model, _ = get_backbone(args.arch, pretrained)
        # Change first layer to take grayscale image
        if args.img_channel == 1:
            self.model = utils.update_backbone_channel(self.model, args.img_channel)

        self.save_folder = self.args.save_folder
        self.finetune_best_weights_path = self.args.save_folder.joinpath(f'supervised_best_loss_{args.dataset_name}.pt')
        if self.finetune_best_weights_path.exists():
            self.finetune_best_weights = torch.load(self.finetune_best_weights_path, map_location=self.args.device)
        else:
            self.finetune_best_weights = None

        # Update classification head
        if self.args.use_bce:
            num_outputs = 1 if len(self.args.labels_dict.keys()) == 2 else len(self.args.labels_dict.keys())
        else:
            num_outputs = len(self.args.labels_dict.keys())
        self.model = utils.set_classifier_head(self.model, num_outputs)
        self.model.to(args.device)


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
    args.dataset_name = configs['finetune']['dataset_name'] if args.dataset_name is None else args.dataset_name
    args.use_bce = configs['finetune']['use_bce'] if args.dataset_name == 'oct_clinical' else False
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
    args.ratio_sup = configs['finetune']['ratio_sup'] if args.ratio_sup is None else args.ratio_sup
    args.ascan_per_group = ascan_per_group
    if overwrite_labels_path is not None:
        labels = pd.read_csv(args.map_df_paths['train'])['label'].unique().tolist()
        args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
        num_cluster_dict['oct'] = len(labels)
        lbls_to_keep = None
        # lbls_to_keep = ['chicken_heart_muscle', 'chicken_stomach_outside'] # None
        # lbls_to_keep = ['chicken_heart_muscle', 'chicken_stomach_inside']
        # lbls_to_keep = ['chicken_stomach_outside', 'chicken_stomach_inside']
        # lbls_to_keep = ['chicken_heart_muscle', 'chicken_stomach_outside', 'chicken_stomach_inside']
        # lbls_to_keep = ['lamb_heart_muscle', 'chicken_stomach_inside']
        # lbls_to_keep = ['lamb_heart_muscle', 'lamb_heart_fat']
        # lbls_to_keep = ['lamb_heart_fat', 'chicken_stomach_inside']
        # lbls_to_keep = ['lamb_heart_muscle', 'chicken_stomach_outside', 'chicken_stomach_inside']
        # lbls_to_keep = ['chicken_heart_muscle', 'lamb_heart_muscle']
        # lbls_to_keep = ['lamb_liver', 'lamb_testicle']
    if lbls_to_keep is not None:
        if len(lbls_to_keep) == 2:
            args.use_bce = configs['finetune']['use_bce']
        labels = lbls_to_keep
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
    if (platform == "linux" or platform == "linux2") and (
            'hpc' in socket.gethostname() or 'u00' in socket.gethostname()):
        print(f"socket name: {socket.gethostname()}")
        args.save_folder = pathlib.Path(r'/fibus/fs0/14/cab8351/OCT_classification/supervised').joinpath(
            f'weights_{args.arch}')
    else:
        args.save_folder = pathlib.Path().resolve().joinpath('supervised').joinpath(f'weights_{args.arch}')
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)
    print(f"Saving weights to: {args.save_folder}")

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
            cv_split_str = f''
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
                train_aug = train_aug + [
                    v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25)], p = 0.5),
                    v2.RandomResizedCrop(size=args.img_reshape, scale=(0.75, 1), interpolation=InterpolationMode.BILINEAR),
                    # https://arxiv.org/abs/2409.13351v1
                    v2.RandomApply([v2.RandomAffine(degrees=5, interpolation=InterpolationMode.BILINEAR)], p=0.5),
                    # v2.RandomApply([v2.RandomAffine(degrees=5, shear=5, interpolation=InterpolationMode.BILINEAR)], p=0.5),
                    # v2.RandomApply([v2.ElasticTransform(alpha=100, sigma=[1e-5, 10])], p=0.5), # Long elastic
                    # v2.RandomApply([v2.ElasticTransform(alpha=20, sigma=[1e-5, 2])], p=0.5),  # Short elastic
                    # v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.5),
                    # v2.RandomApply([v2.GaussianNoise(mean=0, sigma=0.1, clip=True)], p=0.5), # https://doi.org/10.1002/eng2.70110
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
        model = FullSupervisedModel(args)

        # Update model weights name if sequential split
        if args.sequential_split:
            model.finetune_best_weights_path = model.finetune_best_weights_path.parent.joinpath(
                f"{model.finetune_best_weights_path.stem}_seqSplit{cv_split_str}.pt")

        if len(cv_splits) > 1:
            model.finetune_best_weights_path = model.finetune_best_weights_path.parent.joinpath(
                f"{model.finetune_best_weights_path.stem}{cv_split_str}.pt")

        # Update labels
        if lbls_to_keep is not None:
            for l in [train_loader, valid_loader, test_loader]:
                l.dataset.map_df = l.dataset.map_df[l.dataset.map_df['label_str'].isin(labels)].copy()
            # Add extra steps for lamb heart data
            if lbls_to_keep == ['lamb_heart_muscle', 'lamb_heart_fat']:
                print("Re-splitting areas for lamb classes")
                split_areas = {'train': ["area1", "area2", "area5", "area6"],
                               'valid': ["area3", "area7"],
                               'test': ["area4", "area10"],
                               }
                # Re-merge mapping dfs
                map_df = pd.concat([l.dataset.map_df for l in [train_loader, valid_loader, test_loader]], axis=0)
                map_df.loc[:, 'area_id'] = [int(re.sub(r'[^\d]+', '', a.split('_')[-1])) for a in map_df.loc[:, 'area']]

                # Cut away first 750k and last 200k ascans
                map_df = pd.merge(map_df, map_df.groupby('trajectory').agg(idx_max=('idx_end', 'max')).reset_index(),
                                  on='trajectory', how='left')
                # Keep only half of the remaining A-scans
                map_df_a2 = map_df[map_df['area_id'] == 2]
                map_df_a_other = map_df[map_df['area_id'] != 2]
                map_df_a_other = map_df_a_other[map_df_a_other['idx_end'] < map_df_a_other['idx_max'] / 2]
                map_df = pd.concat([map_df_a2, map_df_a_other], axis=0)

                # Re-split
                map_df.loc[:, 'split'] = ''
                for s, a in split_areas.items():
                    map_df.loc[map_df['area'].str.contains('|'.join(a)), 'split'] = s

                for s, l in zip(['train', 'valid', 'test'], [train_loader, valid_loader, test_loader]):
                    l.dataset.map_df = map_df[map_df['split'] == s].reset_index()

                print(f"Areas in train: {train_loader.dataset.map_df['area'].unique()}")
                print(f"Areas in valid: {valid_loader.dataset.map_df['area'].unique()}")
                print(f"Areas in test: {test_loader.dataset.map_df['area'].unique()}")

            # Update model weights name
            model.finetune_best_weights_path = model.finetune_best_weights_path.parent.joinpath(
                f"{model.finetune_best_weights_path.stem}_{'_'.join(labels)}.pt")

        # Train weights
        print(f"Train model")
        if args.dataset_name == 'oct_clinical':
            # Define pos_weights
            # https://www.codegenes.net/blog/pytorch-bcewithlogitsloss-pos_weight/#handling-class-imbalance
            class_counts = train_loader.dataset.map_df.groupby('label').agg(img_count=('img_relative_path', 'count'))
            # pos_weights = torch.Tensor([class_counts.loc[0.0, 'img_count'] / class_counts.loc[1.0, 'img_count']]).to(
            #     args.device)
            pos_weights = None
            if args.use_bce:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            if args.use_bce:
                criterion = nn.BCEWithLogitsLoss()
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
        test_preds, test_labels, test_outputs = model.test(test_loader)

        # Save predictions
        preds_df = pd.DataFrame.from_dict({'pred': test_preds.squeeze(-1), 'pred_labels': test_labels.squeeze(-1)}, orient='columns')
        preds_df = pd.concat([test_loader.dataset.map_df.copy(), preds_df], axis=1)
        assert len(preds_df[preds_df['pred_labels'] == preds_df['label']]) == len(preds_df)
        preds_df = preds_df.drop(columns=['pred_labels'])
        preds_path = f'preds_{args.dataset_name}_{int(args.ratio_sup*100)}p{cv_split_str}.csv'
        preds_df.to_csv(args.save_folder.joinpath(preds_path), index=False)

        # Calculate metrics
        print(f"Test set results using {args.arch} backbone \n(supervised, with {args.ratio_sup * 100}% of {args.dataset_name}):")
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
        cm_path = f"confusion_matrix_{args.dataset_name}{cv_split_str}.png"
        if args.sequential_split:
            cm_path = f"confusion_matrix_{args.dataset_name}_seqSplit{cv_split_str}.png"
        if lbls_to_keep is not None:
            cm_path = f"confusion_matrix_{args.dataset_name}_{'_'.join(lbls_to_keep)}{cv_split_str}.png"
            if args.sequential_split:
                cm_path = f"confusion_matrix_{args.dataset_name}_seqSplit_{'_'.join(lbls_to_keep)}{cv_split_str}.png"

        plt.savefig(args.save_folder.joinpath(cm_path))
        plt.show()
        plt.close()

        # Plot ROC curve if 2 classes
        if len(args.labels_dict) == 2 and args.use_bce:
            fpr, tpr, thresholds = roc_curve(test_labels, test_outputs)
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            roc_path = cm_path = f"roc_{args.dataset_name}{cv_split_str}.png"
            plt.savefig(args.save_folder.joinpath(roc_path))
            plt.show()
            plt.close()

        del(model)


if __name__ == "__main__":
    main()
