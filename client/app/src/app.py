import os
import requests
import streamlit as st
from torchvision.transforms import ToPILImage, ToTensor
from src.utils import to_uint8, image_to_tensor, list_to_tensor, bounding_box

API_URL = os.getenv("APP_PREDICTION_SERVICE")

class front:

    file = None

    @classmethod
    def ask_for_file(cls):
        """Lorem ipsum"""
        label = "Upload an image file with png or jpg extension"
        with st.beta_container():
            cls.file = st.file_uploader(label, type=["png", "jpg"])
        return cls.file

    @classmethod
    def inference(cls):
        """Lorem ipsum"""        
        prediction = cls._predict(cls.file)
        return cls._bounding(cls.file, prediction)

    def _predict(file):
        response = requests.post(API_URL, files={"file": file})
        return response.json()
   
    def _bounding(file, y):
        x, y = image_to_tensor(file), list_to_tensor(y["boxes"])
        return bounding_box(x, y)

    def hide():
        css = "<style> .e1tzin5v3 { display: none; } </style>"
        st.markdown(css, unsafe_allow_html=True)


def flow():
    
    front.ask_for_file()

    def response(app):

        if front.file is not None:
            front.hide()
            return app
        
        return lambda: None

    return response