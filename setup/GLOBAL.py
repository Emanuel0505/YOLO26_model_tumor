import os
import kagglehub

DATASET_ORINGIN = kagglehub.dataset_download("indk214/brain-tumor-dataset-segmentation-and-classification")

YOLO_DATASET = 'brain-tumor-dataset-YOLO'

CLASS = {'glioma': 0,  'meningioma': 1, 'pituitary': 2}

SPLIT_VAL_AND_TEST = 0.3
SPLIT_TEST = 0.1

NAME_MODEL = 'Model_tumor_yolo'


#diretorios

PATH_YAML = os.path.join(YOLO_DATASET, 'data.yaml')

PATH_MODEL_TRAIN = '../train'

PATH_MODEL_VALIDATION = '../validation'
