
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple

import torchvision.transforms.functional

from layers import Flatten, L2Norm, GeM


CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "EfficientNet_B0": 1280,
    "EfficientNet_B1": 1280,
    "EfficientNet_B2": 1408,
    "EfficientNet_B3": 1536,
    "EfficientNet_B4": 1792,
    "EfficientNet_B5": 2048,
    "EfficientNet_B6": 2304,
    "EfficientNet_B7": 2560,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone, False)
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
    
    def forward(self, image: torch.Tensor):
        image = self.backbone(image)
        image = self.aggregation(image)
        return image.detach()

def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

def get_backbone(backbone_name : str, train_all_layers : bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")

        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        if train_all_layers:
            logging.debug("Train all layers of the VGG-16")
        else:
            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    elif backbone_name.startswith("EfficientNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.features.named_children():
                if name == "5": # Freeze layers before block 5
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only the last three blocks of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
    
    backbone = torch.nn.Sequential(*layers)
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

