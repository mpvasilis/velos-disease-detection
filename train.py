import torch
from pathlib import Path
from yolov5 import train

def train_yolov5(dataset_path, checkpoint):
    data_path = Path('preprocessing_folders/'+dataset_path)
    yolov5_path = Path("yolov5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = yolov5_path / "yolov5s.pt"
    project_name = "velos_train_"+checkpoint

    # Train the model
    train.train(
        str(data_path),
        img_size=640,
        batch_size=16,
        epochs=100,
        weights=weights_path,
        project=project_name,
        name="yolov5s_finetune",
        device=device,
        multi_scale=False,
        evolve=False,
        exist_ok=True,
    )
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


