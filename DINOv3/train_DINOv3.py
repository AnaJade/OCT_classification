import argparse
import pathlib
import pynvml
import random
import sys
from sys import platform
import socket

import timm
from addict import Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from dinov3_model import DINO_LoRA

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_supervised_oct_data_loaders, build_image_root, RandomWrapAround, get_cross_valid_splits


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)
parser.add_argument('--ratio_sup',
                    help='Ratio of the dataset used for supervised training (between 0.05 and 0.2)',
                    type=float,)
parser.add_argument('--dataset_name',
                    help='Name of the dataset to use (either oct or oct_clinical)',
                    type=str)

# Img size and moco_dim (nb of classes) values based on the dataset
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



if __name__ == "__main__":
    args = parser.parse_args()
    if args.config_path is None:
        args.config_path = pathlib.Path('../config.yaml')
    config_file = pathlib.Path(args.config_path)

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    configs = utils.load_configs(config_file)
    if platform == "linux" or platform == "linux2":
        print(f"socket name: {socket.gethostname()}")
        if 'hpc' in socket.gethostname() or 'u00' in socket.gethostname():
            dataset_path = pathlib.Path(configs['DINO']['dataset_path_hpc'])
        else:
            dataset_path = pathlib.Path(configs['DINO']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['DINO']['dataset_path_windows'])
    print(f"dataset path: {dataset_path}")
    labels = configs['data']['labels']
    trajectories = configs['data']['trajectories']
    ascan_per_group = configs['data']['ascan_per_group']
    overwrite_labels_path = pathlib.Path(configs['data']['overwrite_labels'])
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    args.dataset_name = configs['DINO']['dataset_name'] if args.dataset_name is None else args.dataset_name
    args.use_bce = configs['DINO']['use_bce'] if args.dataset_name == 'oct_clinical' else False
    if 'oct' in args.dataset_name:
        mean[args.dataset_name] = 3 * [configs['data']['img_mean'] / 255]
        std[args.dataset_name] = 3 * [configs['data']['img_std'] / 255]
        img_size_dict[args.dataset_name] = (512, ascan_per_group)
        num_cluster_dict[args.dataset_name] = len(labels)

    # Dataset
    # args.dataset_name = configs['DINO']['dataset_name']
    args.scan_no_noise = configs['data']['pre_processing']['no_noise'] # Add to args for logging
    args.scan_use_movmean = configs['data']['pre_processing']['use_movmean']  # Add to args for logging
    args.scan_use_speckle = configs['data']['pre_processing']['use_speckle']  # Add to args for logging
    args.scan_sampling = configs['data']['pre_processing']['ascan_sampling']  # Add to args for logging
    if args.dataset_name == 'oct':
        folder_name = 'OCT_lab_data'
    elif args.dataset_name == 'oct_clinical':
        folder_name = 'OCT_clinical_data'
        labels = ['Healthy', 'Lesion']
        num_cluster_dict[args.dataset_name] = len(labels)
        # Update pre processing
        pre_processing['no_noise'] = False  # M-Scans have already been cropped to remove noise
        # pre_processing['ascan_sampling'] = 1
        args.scan_no_noise = False
        # args.scan_sampling = 1
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
    args.img_channel = configs['DINO']['img_channel']
    if 'oct' not in args.dataset_name:
        args.img_channel = 3
    args.sample_within_image = configs['DINO']['sample_within_image']
    args.img_reshape = configs['DINO']['img_reshape']
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512  # BYOL requires square images, so all images will be reshaped to 512x512
    args.ratio_sup = configs['DINO']['ratio_sup'] if args.ratio_sup is None else args.ratio_sup
    args.ascan_per_group = ascan_per_group
    if (args.dataset_name == 'oct') and (overwrite_labels_path is not None):
        labels = pd.read_csv(args.map_df_paths['train'])['label'].unique().tolist()
        args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
        num_cluster_dict['oct'] = len(labels)

    # Training params
    args.seed = configs['training']['random_seed']
    args.dataset_sample = configs['DINO']['dataset_sample']
    args.arch = configs['DINO']['arch']
    args.use_lora = configs['DINO']['use_lora']
    args.workers = configs['DINO']['num_workers']
    args.epochs = configs['DINO']['max_epochs']
    args.batch_size = configs['DINO']['batch_size']
    args.lr = configs['DINO']['lr']
    args.disable_cuda = configs['DINO']['disable_cuda']
    args.out_dim = num_cluster_dict[args.dataset_name]
    args.gpu_index = configs['DINO']['gpu_index']
    args.patience = configs['DINO']['patience']
    if (platform == "linux" or platform == "linux2") and ('hpc' in socket.gethostname() or 'u00' in socket.gethostname()):
        print(f"socket name: {socket.gethostname()}")
        args.save_folder = pathlib.Path(r'/fibus/fs0/14/cab8351/OCT_classification/DINOv3').joinpath(f'weights_{args.arch}')
    else:
        args.save_folder = pathlib.Path().resolve().joinpath(f'weights_{args.arch}')
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)
    print(f"Saving weights to: {args.save_folder}")

    wandb_log = configs['wandb']['wandb_log']
    project_name = configs['wandb']['project_name']
    if project_name != 'Test-project':
        project_name = 'OCT_DINO'

    # Set all random seeds
    print("Setting random seed...")
    utils.set_random_seed(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        # print('__CUDNN VERSION:', torch.backends.cudnn.version())
        # print('__Number CUDA Devices:', torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
        print('Selected GPU index:', args.gpu_index)
        print('__CUDA Device Name:', torch.cuda.get_device_name(args.gpu_index))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(args.gpu_index).total_memory / 1e9)
        print('Clearing cache...')
        torch.cuda.empty_cache()
        print('__CUDA Device Reserved Memory [GB]:', torch.cuda.memory_reserved(args.gpu_index) / 1e9)
        print('__CUDA Device Allocated Memory [GB]:', torch.cuda.memory_allocated(args.gpu_index) / 1e9)
        print('Stats with pynvml:')
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        pynvml.nvmlShutdown()
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.dataset_name == 'oct_clinical':
        # Generate cross-validation split
        cv_splits = get_cross_valid_splits(args, k=3)
        # cv_splits = [None]
    else:
        cv_splits = [None]

    for i, cv_split in enumerate(cv_splits):
        if len(cv_splits) == 1:
            cv_split_str = f''
        else:
            print("================================")
            print(f"Split {i}")
            print(cv_split)
            cv_split_str = f'_split{i}'

        # Create train, valid and test sets
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
                v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25)], p=0.5),
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
                                                                                  seq_split=False,
                                                                                  overwrite_split=cv_split)

        with torch.cuda.device(args.gpu_index):
            # Define model
            learner = DINO_LoRA(args, None, None)

            if len(cv_splits) > 1:
                learner.classifier_best_weights_path = learner.classifier_best_weights_path.parent.joinpath(
                    f"{learner.classifier_best_weights_path.stem}{cv_split_str}.pt")
                learner.lora_best_weights_path = learner.lora_best_weights_path.parent.joinpath(
                    f"{learner.lora_best_weights_path.stem}{cv_split_str}.pt")

            # Train linear layer and LoRA
            if args.dataset_name == 'oct_clinical':
                # Define pos_weights
                # https://www.codegenes.net/blog/pytorch-bcewithlogitsloss-pos_weight/#handling-class-imbalance
                class_counts = train_loader.dataset.map_df.groupby('label').agg(
                    img_count=('img_relative_path', 'count'))
                # pos_weights = torch.Tensor(
                #     [class_counts.loc[0.0, 'img_count'] / class_counts.loc[1.0, 'img_count']]).to(
                #     args.device)
                pos_weights = None
                if args.use_bce:
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                if args.use_bce:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
            opt = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.5,
                patience=2
            )
            learner.train(train_loader, valid_loader, criterion, opt, scheduler=scheduler, wandb_log=wandb_log, project_name=project_name)

            # Get test set performance
            test_preds, test_labels, test_outputs = learner.test(test_loader)

            # Save predictions
            preds_df = pd.DataFrame.from_dict({'pred': test_preds.squeeze(-1), 'pred_labels': test_labels.squeeze(-1)},
                                              orient='columns')
            preds_df = pd.concat([test_loader.dataset.map_df.copy(), preds_df], axis=1)
            assert len(preds_df[preds_df['pred_labels'] == preds_df['label']]) == len(preds_df)
            preds_df = preds_df.drop(columns=['pred_labels'])
            no_lora_str = '' if args.use_lora else '_noLoRA'
            preds_path = f'preds_{args.dataset_name}_{int(args.ratio_sup * 100)}p{no_lora_str}{cv_split_str}.csv'
            preds_df.to_csv(args.save_folder.joinpath(preds_path), index=False)

            # Calculate metrics
            print(f"Test set results using {args.arch} backbone:")
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

            del(learner)

