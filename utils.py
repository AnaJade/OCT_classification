import pathlib

import torch
import yaml
import numpy as np
import pandas as pd
import wandb

from scipy.spatial.transform import Rotation
from torch import nn


def load_configs(config_path: pathlib.Path) -> dict:
    """
    Load the configs in the yaml file, and returns them in a dictionary
    :param config_path: path to the config file as a pathlib path
    :return: dict with all configs in the file
    """
    # Read yaml config
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    return configs


def load_frame_time_steps(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the image time frame csv file
    :param csv_path: path to the time_steps csv file
    :return: dataframe with the file data
    """
    csv = pd.read_csv(csv_path, header=None, names=['time'])
    return csv


def load_log_file(txt_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the robot position log txt file
    :param txt_path: path to the robot pose log file
    :return: dataframe with the file data
    """
    col_names = ['time', 'x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    txt = pd.read_table(txt_path, sep=',', header=None, names=col_names)
    return txt


def wandb_init(project_name: str, hyperparams:dict):
    wandb.init(
        # Choose wandb project
        project=project_name,
        # Add hyperparameter tracking
        config=hyperparams
    )


def wandb_log(phase: str, **kwargs):
    """
    Log the given parameters
    :param phase: Either batch or epoch
    :param kwargs: Values to be logged
    :return:
    """
    # Append phase at the end of the param names
    log_data = {f'{key}_{phase}': value for key, value in kwargs.items()}
    wandb.log(log_data)


def update_backbone_channel(feature_model, ch_in: int):
    """
    Update first layer input to accept either greyscale or rgb images
    :param feature_model: model to be updated
    :param arch: model name
    :param ch_in: desired input channel size
    :return: updated model
    """
    arch = feature_model.__class__.__name__
    if 'ResNet' in arch:
        layer = feature_model.conv1
    elif ('VisionTransformer' in arch) and ('PyramidVisionTransformer' not in arch):
        layer = feature_model.conv_proj
    elif ('EfficientNet' in arch) or ('SwinTransformer' in arch) or ('ConvNeXt' in arch):
        layer = feature_model.features[0][0]
    elif 'PyramidVisionTransformer' in arch:
        layer = feature_model.patch_embed.proj

    layer.in_channels = ch_in
    if ch_in == 1:
        layer.weight = nn.Parameter(torch.mean(layer.weight, dim=1, keepdim=True))
    elif ch_in == 3:
        layer.weight = nn.Parameter(torch.concat(3*[layer.weight], dim=1))

    return feature_model


def set_classifier_head(feature_model, num_classes):
    """
    Update classifier head to be a single layer
    :param feature_model: model to be updated
    :param num_classes: number of outputs in the last layer
    :return:
    """
    arch = feature_model.__class__.__name__
    if 'ResNet' in arch:
        dim_mlp = feature_model.fc.in_features
        feature_model.fc = nn.Sequential(nn.Linear(dim_mlp, num_classes))
    elif ('VisionTransformer' in arch) and ('PyramidVisionTransformer' not in arch):
        dim_mlp = feature_model.heads.head.in_features
        feature_model.heads.head = nn.Sequential(nn.Linear(dim_mlp, num_classes))
    elif 'EfficientNet' in arch:
        dim_mlp = feature_model.classifier[1].in_features
        feature_model.classifier = nn.Sequential(
            list(feature_model.classifier)[0],
            nn.Linear(dim_mlp, num_classes))
    elif 'SwinTransformer' in arch:
        dim_mlp = feature_model.head.in_features
        feature_model.head = nn.Sequential(nn.Linear(dim_mlp, num_classes))
    elif 'ConvNeXt' in arch:
        dim_mlp = feature_model.classifier[2].in_features
        feature_model.classifier = nn.Sequential(
            *list(feature_model.classifier)[:2],
            nn.Linear(dim_mlp, num_classes))
    elif 'PyramidVisionTransformer' in arch:
        dim_mlp = feature_model.head.in_features
        feature_model.head = nn.Sequential(nn.Linear(dim_mlp, num_classes))
    """
    if 'ResNet' in arch:
        feature_model.fc.out_features = num_classes
    elif ('VisionTransformer' in arch) and ('PyramidVisionTransformer' not in arch):
        feature_model.heads.head.out_features = num_classes
    elif ('EfficientNet' in arch) or ('ConvNeXt' in arch):
        feature_model.classifier[-1].out_features = num_classes
    elif ('SwinTransformer' in arch) or ('PyramidVisionTransformer' in arch):
        feature_model.head.out_features = num_classes
        feature_model.num_classes = num_classes
    """

    return feature_model


def calculate_metrics(preds, labels):
    pass


if __name__ == '__main__':
    print()
