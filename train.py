import csv
import os
import shutil

from yolov5 import train
import fiftyone as fo
from yolov5.utils.dataloaders import autosplit

models_dir = "models"

def train_yolov5(dataset_path, checkpoint):
    dataset = fo.Dataset.from_dir(
        dataset_dir="./downloads/train/",
        dataset_type=fo.types.YOLOv5Dataset,
        name="velos-train-dataset",
    )
    session = fo.launch_app(dataset)
    autosplit('preprocessing_folders/'+dataset_path+'/images', weights=(0.8, 0.2, 0.0))
    train.run(imgsz=640, data='preprocessing_folders/'+dataset_path+'/dataset.yaml', device=0, workers=1, batch_size=8)

    train_dir = "runs/train"
    subfolders = next(os.walk(train_dir))[1]
    last_model = subfolders[-1]

    line_count = 0
    results_csv = os.path.join(train_dir, last_model, "results.csv")
    if os.path.isfile(results_csv):
        csv_reader = csv.reader(results_csv)
        for row in csv_reader:
            line_count += 1
        if line_count < 20:
            return False
        best_checkpoints = os.path.join(train_dir, last_model, "best.pt")
        if os.path.exists(best_checkpoints):
            destination_path = os.path.join(models_dir, "best.pt")
            shutil.copy(best_checkpoints, destination_path)
    return True




