import pathlib
import sys
import timm
import torch
from torch import nn
from torchvision import models
from torchvision.datasets import STL10

from overwrite_ViT import vit_b_16_byol, vit_b_16_512_byol, vit_h_14_byol


def get_backbone(arch: str, pretrained: bool):
    """
    Return base backbone based on desired architecture
    :param arch: model name
    :param pretrained: whether default pretrained weights should be used
    :return: model, feature_layer
    """
    # Define model
    weights = 'DEFAULT' if pretrained else None
    if arch == 'resnet18':
        feature_model = models.resnet18(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'resnet50':
        feature_model = models.resnet50(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'vitb16':
        feature_model = vit_b_16_byol(weights=weights)
        feature_layer = 1
    elif arch == 'vitb16_512':
        feature_model = vit_b_16_512_byol(weights=weights)
        feature_layer = 1
    elif arch == 'efficientnetV2s':
        feature_model = models.efficientnet_v2_s(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'efficientnetb3':
        feature_model = models.efficientnet_b3(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'swinv2t':
        feature_model = models.swin_v2_t(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'convnextt':
        feature_model = models.convnext_tiny(weights=weights)
        feature_layer = 'avgpool'
    elif arch == 'pvtv2b0':
        feature_model = timm.create_model("pvt_v2_b0", pretrained=pretrained)
        feature_layer = 'head_drop'
    else:
        print(f'{arch} not a valid model')
        feature_model, feature_layer = None, None

    """
    # Debug
    # Test feature layer
    if type(feature_layer) == str:
        modules = dict([*feature_model.named_modules()])
        layer = modules.get(feature_layer, None)
    elif type(feature_layer) == int:
        children = [*feature_model.net.children()]
        layer = children[feature_layer]
    print(layer)
    """

    return [feature_model, feature_layer]





