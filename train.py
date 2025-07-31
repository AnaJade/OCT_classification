import argparse
import pathlib
import platform
import pandas as pd

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader
from torchvision.io import read_image

# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import utils
import utils_data
from utils_data import MandibleDataset, NormTransform
from SiameseNet import SiameseNetwork, train_model


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)
    # config_file = pathlib.Path("siamese_net/config.yaml")

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    # Load configs
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_train = configs['data']['trajectories_train']
    anno_paths_valid = configs['data']['trajectories_valid']
    anno_paths_test = configs['data']['trajectories_test']
    resize_img_h = configs['data']['resize_img']['img_h']
    resize_img_w = configs['data']['resize_img']['img_w']
    grayscale = configs['data']['grayscale']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    weights_file_addon = configs['training']['weights_file_addon']
    use_pretrained = configs['training']['use_pretrained']
    pre_trained_weights = configs['training']['pre_trained_weights']
    cam_inputs = configs['training']['cam_inputs']
    train_bs = configs['training']['train_bs']
    valid_bs = configs['training']['valid_bs']
    test_bs = configs['training']['test_bs']
    rename_side = True if 'center_rmBackground' in cam_inputs else False

    nb_epochs = configs['training']['max_epochs']
    patience = configs['training']['patience']
    lr = configs['training']['lr']
    scheduler_step_size = configs['training']['lr_scheduler']['step_size']
    scheduler_gamma = configs['training']['lr_scheduler']['gamma']

    wandb_log = configs['wandb']['wandb_log']
    project_name = configs['wandb']['project_name']

    if rename_side:
        cam_inputs[-1] = 'Side'
    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}"
    print(f'Weights will be saved to: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        min_max_pos = utils_data.get_dataset_min_max_pos(configs)
        print(f'min_max_pos: {min_max_pos}')
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_train = utils_data.merge_annotations(dataset_root, anno_paths_train)
    annotations_valid = utils_data.merge_annotations(dataset_root, anno_paths_valid)
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)

    # Check if all images are available
    unavailable_imgs = set()
    all_annotations = pd.concat([annotations_train, annotations_valid, annotations_test], axis=0)
    annotation_stems = set(all_annotations.index.tolist())
    for c in cam_inputs:
        imgs = list(dataset_root.joinpath(c).glob('*.jpg'))
        img_stems = set([i.stem[:-2] for i in imgs])
        cam_unavailable_imgs = set(annotation_stems) ^ set(img_stems)
        if cam_unavailable_imgs:
            unavailable_imgs.update(cam_unavailable_imgs)
    if unavailable_imgs:
        unavailable_imgs_train = set(annotations_train.index.tolist()) & unavailable_imgs
        unavailable_imgs_valid = set(annotations_valid.index.tolist()) & unavailable_imgs
        unavailable_imgs_test = set(annotations_test.index.tolist()) & unavailable_imgs
        annotations_train.drop(index=list(unavailable_imgs_train), inplace=True)
        annotations_valid.drop(index=list(unavailable_imgs_valid), inplace=True)
        annotations_test.drop(index=list(unavailable_imgs_test), inplace=True)
        print(f'Missing train images: {list(unavailable_imgs_train)}')
        print(f'Missing valid images: {list(unavailable_imgs_valid)}')
        print(f'Missing test images: {list(unavailable_imgs_test)}')

    # Create dataset object
    print("Initializing dataset object...")
    # Create dataset objects
    if rename_side:
        cam_inputs[-1] = 'center_rmBackground'
    if grayscale:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 torchvision.transforms.Grayscale(),
                                 NormTransform()])  # Remember to also change the annotations for other transforms
    else:
        transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                 NormTransform()])
    dataset_train = MandibleDataset(dataset_root, cam_inputs, annotations_train, min_max_pos, transforms)
    dataset_valid = MandibleDataset(dataset_root, cam_inputs, annotations_valid, min_max_pos, transforms)
    dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, min_max_pos, transforms)

    print("Creating dataloader...")
    if platform.system() == 'Windows':
        dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=False, num_workers=0)
        dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=0)
        dataloader_test = DataLoader(dataset_valid, batch_size=test_bs, shuffle=False, num_workers=0)
    else:
        dataloader_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True, num_workers=4)
        dataloader_valid = DataLoader(dataset_valid, batch_size=valid_bs, shuffle=False, num_workers=4)
        dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)

    # Define the model
    model = SiameseNetwork(configs)
    if use_pretrained and pre_trained_weights != '':
        print(f'Initial weight values set to {pre_trained_weights}.pth')
        model.load_state_dict(torch.load(f"siamese_net/model_weights/{pre_trained_weights}.pth"))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # Define the criterion, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params, lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    print("Training model...")
    print(f"Logging on wandb: {wandb_log}")
    model_fit = train_model(configs, model, [dataloader_train, dataloader_valid, dataloader_test],
                            device, criterion, optimizer, scheduler)

    # Save model (not needed, since the model is saved during training
    """
    print("Saving best model...")
    # weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{configs['training']['num_fc_hidden_units']}"
    torch.save(model_fit.state_dict(), f"siamese_net/model_weights/{weights_file}.pth")
    """


