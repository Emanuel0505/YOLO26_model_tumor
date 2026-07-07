
from GLOBAL import DATASET_ORINGIN, YOLO_DATASET, SPLIT_TEST, SPLIT_VAL_AND_TEST, PATH_YAML, CLASS
from sklearn.model_selection import train_test_split
import cv2
import kagglehub
import os
import shutil
import yaml

def search_folder(base_path, folder):
  """
  Buscar a pasta do caminho ignorando maiusculas
  """
  for root, dirs, files in os.walk(base_path):
    for d in dirs:
      if d.lower() == folder.lower():
        return os.path.join(root, d)
  return None

def create_dataset_folders(path):
  """
    Criar as estrutura das pastas do Dataset para o YOLO
  """
  folders= ['train', 'val', 'test']
  for folder in folders:
    os.makedirs(os.path.join(path, 'images', folder), exist_ok=True)
    os.makedirs(os.path.join(path, 'labels', folder), exist_ok=True)

def polygon_mask(path):
  """
  Converter a imagem da mascara em coordenadas do YOLO
  """
  mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if mask is None:
    return []

  H, W = mask.shape[:2]

  _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  polygons = []
  img_area = H * W

  for contour in contours:
    area = cv2.contourArea(contour)

    if area < 50:
      continue

    if area > img_area * 0.9:
      continue

    flat_contour = contour.reshape(-1, 2)
    polygon = []

    for x, y in flat_contour:
      polygon.extend([x / W, y / H])

    if len(polygon) >= 6:
      polygons.append(polygon)

  return polygons

  H, W = mask.shape
  _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  polygons = []
  for contour in contours:
    if cv2.contourArea(contour) > 50:
      flat_contour = contour.flatten().tolist()
      polygon = []

      for i in range(0, len(flat_contour), 2):
        x = flat_contour[i] / W
        y = flat_contour[i+1] / H
        polygon.extend([x, y])

      polygons.append(polygon)

  return polygons

def process_dataset():
  """
  Processar o dataset para o YOLO
  """
  folder_segmentation = search_folder(DATASET_ORINGIN, 'segmentation')
  folder_classification = search_folder(DATASET_ORINGIN, 'classification')

  create_dataset_folders(YOLO_DATASET)
  all_images = []

  if folder_segmentation:

    for class_name in os.listdir(folder_segmentation):
      folder_class = os.path.join(folder_segmentation, class_name)

      if not os.path.isdir(folder_class):
        continue

      for cn in class_name.split():
        if cn.lower() in CLASS:
          print(f"Classe '{class_name}' encontrada na lista de classes.")
          class_id = CLASS[cn.lower()]
          continue

      all_files = os.listdir(folder_class)

      masks = [f for f in all_files if 'mask' in f.lower()]
      images = [f for f in all_files if 'mask' not in f.lower()]

      for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        mask = None

        for m in masks:
          if base_name in m:
            mask = m
            break

        if mask:
          path_img = os.path.join(folder_class, img_name)
          path_mask = os.path.join(folder_class, mask)
          all_images.append((path_img, path_mask, class_id))
  else:
    print("Pasta 'segmentation' não encontrada.")

  if folder_classification:
    folders_notumor = [
        os.path.join(folder_classification, 'Training', 'notumor'),
        os.path.join(folder_classification, 'Testing', 'notumor'),
    ]

    for folder in folders_notumor:
      images_notumor = os.listdir(folder)
      for img_name in images_notumor:
        path_img = os.path.join(folder, img_name)
        all_images.append((path_img, None, None))

  print(f"Total de pares (Imagem+Máscara) encontrados: {len(all_images)}")

  train, temp_data = train_test_split(all_images, test_size=SPLIT_VAL_AND_TEST, random_state=42)
  val, test = train_test_split(temp_data, test_size=SPLIT_TEST, random_state=42)

  def copy_and_convert(dataset, split_name):
    print(f"A processar conjunto de {split_name}...")

    for img_path, mask_path, class_id in dataset:
      base_name = os.path.basename(img_path)
      txt_name = os.path.splitext(base_name)[0] + '.txt'

      destination_img = os.path.join(YOLO_DATASET, 'images', split_name, base_name)
      destination_txt = os.path.join(YOLO_DATASET, 'labels', split_name, txt_name)

      shutil.copy(img_path, destination_img)

      if mask_path is None or class_id is None:
        open(destination_txt, 'w').close()
      else:
        polygons = polygon_mask(mask_path)
        with open(destination_txt, 'w') as f:
          for poly in polygons:
            row = f'{class_id}' + ' '.join([f'{coord: .6f}' for coord in poly]) + '\n'
            f.write(row)

  copy_and_convert(train, 'train')
  copy_and_convert(val, 'val')
  copy_and_convert(test, 'test')

  print("\n✅ Conversão concluída! O dataset pronto para o YOLO está na pasta:", os.path.abspath(YOLO_DATASET))

process_dataset()

data_yaml = {
    'path': os.path.abspath(YOLO_DATASET),
    'train': 'images/train',
    'val':  'images/val',
    'test': 'images/test',
    'nc': len(CLASS),
    'names': list(CLASS.keys())
}

with open(PATH_YAML, 'w') as f:
  yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
