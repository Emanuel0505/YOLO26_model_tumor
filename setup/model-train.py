from GLOBAL import *
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

    #config para melhor treino
    optimizer='AdamW',
    lr0=0.001,
    freeze=10,
    degrees=15,

)