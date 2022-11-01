import logging
import torch

class DiseaseDetection:

    def __init__(self, model_dir_or_repo, model_name, images_to_detect):
        self.model_dir_or_repo = model_dir_or_repo
        self.model_name = model_name
        self.images_to_detect = images_to_detect
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


    def detect(self):
        results = self.model(self.images_to_detect)
        results.print()
        res = results.pandas()
        print(res.xywh)