import os
import kagglehub

DATASET_ORINGIN = kagglehub.dataset_download("indk214/brain-tumor-dataset-segmentation-and-classification")

YOLO_DATASET = 'brain-tumor-dataset-YOLO'

CLASS = {'glioma': 0,  'meningioma': 1, 'pituitary': 2}

SPLIT_VAL_AND_TEST = 0.3
SPLIT_TEST = 0.1

PATH_YAML = os.path.join(YOLO_DATASET, 'data.yaml')

DRIVE = False 

PATH_MODEL_TRAIN = '/content/drive/MyDrive/visaocomputacional/Projeto-Deteccao_e_Tumor_no_Cerebro' if DRIVE else '../runs/train'
