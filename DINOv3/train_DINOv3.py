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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import STL10
from sklearn.metrics import classification_report

from DINOv3.dinov3_model import DINO_LoRA

# DEBUG
from transformers import VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType
from transformers import LlamaTokenizer, LlamaForCausalLM

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset, build_image_root
from finetune_model import get_oct_data_loaders, get_stl10_data_loaders


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
        print(f"socket name: {socket.gethostname()}")
        if 'hpc' in socket.gethostname() or 'u00' in socket.gethostname():
            dataset_path = pathlib.Path(configs['DINO']['dataset_path_hpc'])
        else:
            dataset_path = pathlib.Path(configs['DINO']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['DINO']['dataset_path_windows'])
    print(f"dataset path: {dataset_path}")
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    args.dataset_name = configs['DINO']['dataset_name']
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
        pre_processing['ascan_sampling'] = 1
        args.scan_no_noise = False
        args.scan_sampling = 1
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
    args.img_channel = configs['DINO']['img_channel']
    if 'oct' not in args.dataset_name:
        args.img_channel = 3
    args.sample_within_image = configs['DINO']['sample_within_image']
    args.img_reshape = configs['DINO']['img_reshape']
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512  # BYOL requires square images, so all images will be reshaped to 512x512
    args.ascan_per_group = ascan_per_group

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
        args.save_folder = pathlib.Path(r'/fibus/fs0/14/cab8351/OCT_classification/DINO').joinpath(f'weights_{args.arch}')
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

    # Create train, valid and test sets
    img_transforms = [transforms.ToTensor(),
                      transforms.Resize((args.img_reshape, args.img_reshape)),
                      transforms.Normalize(mean=mean[args.dataset_name],
                                           std=std[args.dataset_name])]
    if args.img_channel == 1:
        img_transforms.append(transforms.Grayscale())
    img_transforms = transforms.Compose(img_transforms)
    train_dataset = OCTDataset(args.data, 'train',
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False,  # args.use_iipp,
                               num_same_area=-1,
                               transforms=img_transforms,
                               pre_sample=args.dataset_sample)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=0, drop_last=False, shuffle=True)

    valid_dataset = OCTDataset(args.data, 'valid',
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False,  # args.use_iipp,
                               num_same_area=-1,
                               transforms=img_transforms,
                               pre_sample=args.dataset_sample)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=0, drop_last=False, shuffle=False)

    test_dataset = OCTDataset(args.data, 'test',
                              args.map_df_paths, args.labels_dict,
                              ch_in=args.img_channel,
                              sample_within_image=args.sample_within_image,
                              use_iipp=False,
                              num_same_area=-1,
                              transforms=img_transforms,
                              pre_sample=args.dataset_sample)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=0, drop_last=False, shuffle=False)

with torch.cuda.device(args.gpu_index):
        # feature_model, feature_layer = get_backbone(args.arch, args.use_pretrained)
        # feature_model = timm.create_model('convnext_tiny.dinov3_lvd1689m',
        #                               pretrained=True,
        #                               num_classes=0)
        # DEBUG
        # Find where dino matches convnextt
        # convnextt = models.convnext_tiny(weights='DEFAULT')
        # children_convnextt = [*convnextt.children()]
        # children_dino_convnext = [*feature_model.children()]
        # modules_convnextt = dict([*convnextt.named_modules()])
        # modules_dino = dict([*feature_model.named_modules()])

        # vit-s
        # dino_vit = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)
        # vitb = models.vit_b_16(weights='DEFAULT')
        # children_vitb = [*vitb.children()]
        # children_dino_vitb = [*dino_vit.children()]
        # modules_vitb = dict([*vitb.named_modules()])
        # modules_dino_vitb = dict([*dino_vit.named_modules()])


        # Change first layer to take grayscale image
        # if args.img_channel == 1:
        #     # feature_model = utils.update_backbone_channel(feature_model, args.img_channel)
        #     pass

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
        # aug = [transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.3), # Used to counter flipped scans
        #        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
        #        transforms.RandomApply([transforms.RandomRotation(degrees=8),
        #                                # transforms.CenterCrop(size=(188, 236)), # Used in the paper, but not really applicable here
        #                                transforms.RandomHorizontalFlip()], p=0.5),
        #        ]
        # aug = transforms.Compose(aug)

        # Define model
        learner = DINO_LoRA(args, None, None)

        # Train linear layer and LoRA
        criterion = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=0.1) # , eps=6e-5)
        learner.train(train_loader, valid_loader, criterion, opt)

        # Get test set performance
        test_logits, test_oh_labels = learner.test(test_loader)
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

        # Train
        # if wandb_log:
        #     utils.wandb_init(project_name, hyperparams=vars(args))
        # best_epoch = 0
        # best_loss = 1e6
        # for e in range(args.epochs):
        #     print(f"\n================================\n"
        #           f"Epoch {e}")
        #     if (e - best_epoch) >= args.patience:
        #         print(f'Loss has not improved for {args.patience} epochs. Training has stopped')
        #         print(f'Best loss was {best_loss} @ epoch {best_epoch}')
        #         break
        #     avg_epoch_loss = []
        #     for images, labels in tqdm(train_loader, desc='Training'):
        #         # images = torch.randn(20, 3, 256, 256)
        #         # with torch.autocast(device_type=f'cuda:{args.gpu_index}', dtype=torch.float16):
        #         # with torch.autograd.detect_anomaly():
        #         images = images.to(args.device)
        #         labels = labels.to(args.device)
        #         loss = learner(images)
        #         opt.zero_grad()
        #         loss.backward()
        #         opt.step()
        #         learner.update_moving_average() # update moving average of target encoder
        #         avg_epoch_loss.append(loss)
#
        #         if wandb_log:
        #             utils.wandb_log('batch', loss=loss)
#
        #     avg_epoch_loss = float(torch.mean(torch.stack(avg_epoch_loss)).cpu().detach().numpy())
        #     if wandb_log:
        #         utils.wandb_log('epoch', loss=avg_epoch_loss)
        #     if avg_epoch_loss < best_loss:
        #         print(f'New best loss achieved @ epoch {e}: {avg_epoch_loss}')
        #         best_epoch = e
        #         best_loss = avg_epoch_loss
        #         torch.save(feature_model.state_dict(), save_folder.joinpath(f'byol_best_loss.pt'))
        #     if (e+1)%10 == 0:
        #         torch.save(feature_model.state_dict(), save_folder.joinpath('byol_{:04d}.pth.tar'.format(e)))
#
        # # save your improved network
        # torch.save(feature_model.state_dict(), save_folder.joinpath('byol_{:04d}_last.pth.tar'.format(e)))
        # # Update best model name to include epoch
        # best_weights_path = save_folder.joinpath(f'byol_best_loss.pt')
        # best_weights_path.rename(best_weights_path.parent.joinpath(f'byol_best_loss_{best_epoch:04d}.pt'))