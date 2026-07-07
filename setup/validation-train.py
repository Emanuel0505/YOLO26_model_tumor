from GLOBAL import NAME_MODEL, PATH_YAML, PATH_MODEL_VALIDATION
from ultralytics import YOLO
import os

model = YOLO(os.path.join('runs', 'train', NAME_MODEL,'weights', 'best.pt'))

metricas = model.val(
    data=PATH_YAML,
    imgsz=640,          
    batch=32,           
    conf=0.25,          
    iou=0.6,            
    plots=True,         
    visualize=True,     
    project=PATH_MODEL_VALIDATION,
    name="result_val"
)

print("\n🎯 MÉTRICAS DA SEGMENTAÇÃO DO TUMOR:")
print(f"mAP50-95 (Precisão Rigorosa Máscara): {metricas.seg.map:.4f}")
print(f"mAP50 (Acurácia Básica Máscara): {metricas.seg.map50:.4f}")
print(f"Precisão (Máscara): {metricas.seg.p.mean():.4f}")
print(f"Recall / Sensibilidade (Máscara): {metricas.seg.r.mean():.4f}")