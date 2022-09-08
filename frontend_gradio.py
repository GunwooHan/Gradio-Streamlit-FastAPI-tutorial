from PIL import Image
import json
import urllib

import numpy as np
import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"

URL = urllib.parse.urljoin(BACKEND_URL, "predict")

def segmentation(img):
    response_predict = requests.post(url=URL, data=json.dumps({"image": img.tolist()}))
    if response_predict.ok:
        res = response_predict.json()
        segmentation_map = np.array(res['segmentation'])
        return segmentation_map
    else:
        return None

def main():    
    demo = gr.Interface(segmentation, gr.Image(), "image")
    demo.launch()


if __name__ == '__main__':
    main()