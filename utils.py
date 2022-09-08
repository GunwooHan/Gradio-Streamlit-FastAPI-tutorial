import io
import base64
from PIL import Image

import numpy as np
import torchvision


CLASS_COLOR = np.array([[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128],
        [153, 0, 51], [102, 255, 153] , [255, 51, 153], [102, 204, 255],[0, 102, 51],
        [255, 153, 204]], np.uint8)

def get_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(520),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def visualize(segmentation_map):
    if len(segmentation_map.shape) != 2:
         raise ValueError('Expect 2-D input label')
    
    return CLASS_COLOR[segmentation_map]