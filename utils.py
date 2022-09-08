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

def image2bytes(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    return decoded

def bytes2image(bytes):
    decoded_bytes = base64.b64decode(bytes),
    img = np.array(io.BytesIO(decoded_bytes), dtype=np.uint8)
    return img