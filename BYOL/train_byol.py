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

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from byol_pytorch import BYOL
from overwrite_byol import BYOL_custom
from torchvision import models
from torchvision.datasets import STL10

from feature_model import get_backbone

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset, build_image_root


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
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
        if 'hpc' in socket.gethostname():
            dataset_path = pathlib.Path(configs['BYOL']['dataset_path_hpc'])
        else:
            dataset_path = pathlib.Path(configs['BYOL']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['BYOL']['dataset_path_windows'])
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    mean['oct'] = 3*[configs['data']['img_mean']/255]
    std['oct'] = 3*[configs['data']['img_std']/255]
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    # Dataset
    args.dataset_name = configs['BYOL']['dataset_name']
    args.scan_no_noise = configs['data']['pre_processing']['no_noise'] # Add to args for logging
    args.scan_use_movmean = configs['data']['pre_processing']['use_movmean']  # Add to args for logging
    args.scan_use_speckle = configs['data']['pre_processing']['use_speckle']  # Add to args for logging
    args.scan_sampling = configs['data']['pre_processing']['ascan_sampling']  # Add to args for logging
    dataset_root = pathlib.Path(dataset_path).joinpath(
        'OCT_lab_data' if args.dataset_name == 'oct' else args.dataset_name)
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {dataset_root.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: dataset_root.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    args.img_channel = configs['BYOL']['img_channel']
    if args.dataset_name != 'oct':
        args.img_channel = 3
    args.sample_within_image = configs['BYOL']['sample_within_image']
    args.img_reshape = configs['BYOL']['img_reshape']
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512 # BYOL requires square images, so all images will be reshaped to 512x512
    args.use_iipp = configs['BYOL']['use_iipp']
    args.ascan_per_group = ascan_per_group

    # Training params
    args.seed = configs['training']['random_seed']
    args.dataset_sample = configs['BYOL']['dataset_sample']
    args.arch = configs['BYOL']['arch']
    args.use_pretrained = configs['BYOL']['use_pretrained']
    args.workers = configs['BYOL']['num_workers']
    args.epochs = configs['BYOL']['max_epochs']
    args.batch_size = configs['BYOL']['batch_size']
    args.lr = configs['BYOL']['lr']
    args.disable_cuda = configs['BYOL']['disable_cuda']
    args.out_dim = num_cluster_dict[args.dataset_name]
    args.gpu_index = configs['BYOL']['gpu_index']
    args.patience = configs['BYOL']['patience']
    save_folder = pathlib.Path().resolve().joinpath(f'weights_{args.arch}')
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

    wandb_log = configs['wandb']['wandb_log']
    project_name = configs['wandb']['project_name']
    if project_name != 'Test-project':
        project_name = 'OCT_BYOL'

    # Set all random seeds
    print("Setting random seed...")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # Dataloader
    if args.dataset_name == 'oct':
        img_transforms = [transforms.ToTensor(), # scales pixel values to [0, 1]
                          transforms.Resize((args.img_reshape, args.img_reshape)),
                          transforms.Normalize(mean=mean[args.dataset_name],
                                               std=std[args.dataset_name])]
        if (args.sample_within_image <= 0) and (args.img_reshape <= 480):
            img_transforms.insert(1, transforms.CenterCrop(480))
        if args.img_channel == 1:
            img_transforms.append(transforms.Grayscale())
        img_transforms = transforms.Compose(img_transforms)
        train_dataset = OCTDataset(dataset_root, 'train',
                                   args.map_df_paths, args.labels_dict,
                                   ch_in=args.img_channel,
                                   sample_within_image=args.sample_within_image,
                                   use_iipp=args.use_iipp,
                                   num_same_area=-1,
                                   transforms=img_transforms,
                                   pre_sample=args.dataset_sample)
    elif args.dataset_name == 'stl10':
        args.img_size = img_size_dict[args.dataset_name]
        img_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean[args.dataset_name],
                                                                  std=std[args.dataset_name])])
        train_dataset = STL10(dataset_root, split="train",
                              transform=img_transforms,
                              download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, drop_last=False, shuffle=True)

    with torch.cuda.device(args.gpu_index):
        feature_model, feature_layer = get_backbone(args.arch, args.use_pretrained)

        # Change first layer to take grayscale image
        if args.img_channel == 1:
            feature_model = utils.update_backbone_channel(feature_model, args.img_channel)

        # Augmentations (from iipp paper, sec 3.3.1):
        #   vertical_flip(p=0.3), due to some scans being flipped because the probe was too close
        #   brightness(p=0.8)
        #   contrast(p=0.8, max_rel_change=0.4)
        #   rotate(p=0.5, max_angle=8deg)
        #   crop_centrally(p=0.5, res=188x236)
        #   hori_flip(p=0.5)
        #   random_crop(scale=[0.25, 1], aspect_ratio=[3/4, 4/3]
        #   resize(192x192)
        # No gaussian blur, hue, saturation and colour droppings
        aug = [transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.3), # Used to counter flipped scans
               transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
               transforms.RandomApply([transforms.RandomRotation(degrees=8),
                                       # transforms.CenterCrop(size=(188, 236)), # Used in the paper, but not really applicable here
                                       transforms.RandomHorizontalFlip()], p=0.5),
               ]
        aug = transforms.Compose(aug)

        learner = BYOL_custom(
            feature_model.cuda(args.gpu_index),
            ch_in=args.img_channel,
            use_iipp=args.use_iipp,
            image_size=args.img_size,
            hidden_layer=feature_layer,
            augment_fn=aug,
            augment_fn2=aug
        )

        opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
        # opt = torch.optim.SGD(learner.parameters(), lr=args.lr)

        # Train
        if wandb_log:
            utils.wandb_init(project_name, hyperparams=vars(args))
        best_epoch = 0
        best_loss = 1e6
        for e in range(args.epochs):
            print(f"\n================================\n"
                  f"Epoch {e}")
            if (e - best_epoch) >= args.patience:
                print(f'Loss has not improved for {args.patience} epochs. Training has stopped')
                print(f'Best loss was {best_loss} @ epoch {best_epoch}')
                break
            avg_epoch_loss = []
            if e > 0:
                if args.use_iipp:
                    train_loader.dataset.create_iipp_map_df()
            for images, _ in tqdm(train_loader):
                # images = torch.randn(20, 3, 256, 256)
                with torch.autocast(device_type=f'cuda:{args.gpu_index}', dtype=torch.float16):
                    if args.use_iipp:
                        images, meta_data = images
                    images = images.to(args.device)
                    loss = learner(images)
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average() # update moving average of target encoder
                avg_epoch_loss.append(loss)

                if wandb_log:
                    utils.wandb_log('batch', loss=loss)

            avg_epoch_loss = float(torch.mean(torch.stack(avg_epoch_loss)).cpu().detach().numpy())
            if wandb_log:
                utils.wandb_log('epoch', loss=avg_epoch_loss)
            if avg_epoch_loss < best_loss:
                print(f'New best loss achieved @ epoch {e}: {avg_epoch_loss}')
                best_epoch = e
                best_loss = avg_epoch_loss
                torch.save(feature_model.state_dict(), save_folder.joinpath(f'byol_best_loss.pt'))
            if (e+1)%10 == 0:
                torch.save(feature_model.state_dict(), save_folder.joinpath('byol_{:04d}.pth.tar'.format(e)))

        # save your improved network
        torch.save(feature_model.state_dict(), save_folder.joinpath('byol_{:04d}_last.pth.tar'.format(e)))
        # Update best model name to include epoch
        best_weights_path = save_folder.joinpath(f'byol_best_loss.pt')
        best_weights_path.rename(best_weights_path.parent.joinpath(f'byol_best_loss_{best_epoch:04d}.pt'))