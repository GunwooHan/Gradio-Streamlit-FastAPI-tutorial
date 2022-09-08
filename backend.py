from typing import Any

import torch
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

from utils import get_transform, visualize


app = FastAPI()

weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

transform = get_transform()

class InputData(BaseModel):
    image: Any

@app.post("/predict")
async def predict(data: InputData):
    img = np.array(data.image, dtype=np.uint8)
    tensor_img = transform(img).unsqueeze(0)
    output = model(tensor_img)['out'].squeeze(0)
    output = torch.argmax(output, dim=0)
    label_map = visualize(output) 
    return JSONResponse({'segmentation' : label_map.tolist()})
