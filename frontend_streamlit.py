from PIL import Image
import json
import urllib

import cv2
import numpy as np
import streamlit as st
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
    src_uploaded_file = st.file_uploader('Upload an image')
    if src_uploaded_file is not None:
        pil_img = Image.open(src_uploaded_file)
        st.image(pil_img, width=250)
        np_img = np.array(pil_img)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    if st.button("Predict"):
        try:
            response_predict = requests.post(
                url=URL,
                data=json.dumps({"image": np_img.tolist()}),
            )


            if response_predict.ok:
                res = response_predict.json()
                segmentation_map = np.array(res['segmentation'], dtype=np.uint8)
                result_img = Image.fromarray(segmentation_map)
                st.image(result_img, width=250)
            else:
                st.write("Some error occured")

        except ConnectionError:
            st.write("Couldn't reach backend")


if __name__ == '__main__':
    main()