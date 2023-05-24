import torch
from pathlib import Path
from yolov5 import train

def train_yolov5(dataset_path, checkpoint):
    train.run(imgsz=640, data='preprocessing_folders/'+dataset_path+'/dataset.yaml', device=0, workers=1)
    return True

# import fiftyone as fo
# name = "velos-train-dataset"
# dataset_dir = "./downloads/train/"
# dataset_type = fo.types.YOLOv5Dataset
#
# dataset = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir,
#     dataset_type=dataset_type,
#     name=name,
# )
# session = fo.launch_app(dataset)
# session.wait()


