from GLOBAL import *
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

model = YOLO("yolo26s-seg.pt")

model.train(
    data=PATH_YAML,
    epochs=200,
    plots=True,  
    device=0,

    #diretorio
    name= 'Model_tumor_yolo',
    project=PATH_MODEL_TRAIN,
    exist_ok=False,

    #config para melhor treino
    optimizer='AdamW',
    batch=32,
    lr0=0.001,
    freeze=10,
    degrees=15,
    mask_ratio=1,
    cls_pw=1,
    patience=50,
)