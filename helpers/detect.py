import logging
import os
import yolov5

import torch

class DiseaseDetection:

    def __init__(self,  images_to_detect, mission):
        self.images_to_detect = images_to_detect
        self.model = yolov5.load(os.path.join('model', 'best.pt'))
        self.mission = mission


    def detect(self):
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image
        results = self.model(self.images_to_detect)
        results.print()
        results.save(save_dir='detection_results/'+str(self.mission))
        return results.pandas()