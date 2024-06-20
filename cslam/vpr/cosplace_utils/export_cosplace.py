import numpy as np

import os
from os.path import join, exists, isfile, realpath, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import sys
import pickle
import sklearn
from sklearn.neighbors import NearestNeighbors
from cslam.vpr.cosplace_utils.network import GeoLocalizationNet

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
            
if __name__ == '__main__':
    model = torch.jit.script(GeoLocalizationNet("resnet18", 64))
    model.save('cosplace_model_resnet18.pt')
    checkpoint = torch.load(
        "/models/resnet18_64.pth", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    w = {k: v for k, v in model.state_dict().items()}
    torch.save(w, "resnet18_64.pth")
    
    model = torch.jit.script(GeoLocalizationNet("resnet101", 512))
    model.save('cosplace_model_resnet101.pt')
    checkpoint = torch.load(
        "/models/resnet101_512.pth", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    w = {k: v for k, v in model.state_dict().items()}
    torch.save(w, "resnet101_512.pth")