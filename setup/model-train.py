from setup.GLOBAL import *
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

model = YOLO("yolo26n-seg.pt")

model.train(
    data=PATH_YAML,
    epochs=100,
    plots=True,  
    device=0,
    name= 'Model_tumor_yolo',
    project=PATH_MODEL_TRAIN,
)