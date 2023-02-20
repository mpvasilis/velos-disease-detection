import fiftyone as fo
name = "velos-train-dataset"
dataset_dir = "./downloads/train"
dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)
session = fo.launch_app(dataset)