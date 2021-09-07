from torch import Tensor
from fastapi import FastAPI, Depends
from src.preprocessing import file_feature
from src.predict import infering, thresholding


app = FastAPI()


@app.post("/image/predict")
def image_predict(feature: Tensor = Depends(file_feature)):
    prediction = infering(feature)
    return thresholding(prediction)
