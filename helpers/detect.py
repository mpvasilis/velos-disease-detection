import logging
import os
import yolov5

import torch

class DiseaseDetection:

    def __init__(self,  images_to_detect, mission):
        self.images_to_detect = images_to_detect
        self.model = yolov5.load(os.path.join('model', 'best.pt')) #, device=0
        self.mission = mission


    def detect(self):
        self.model.conf = 0.15  # NMS confidence threshold
        self.model.iou = 0.34  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 100  # maximum number of detections per image

        for i in range(30):
            if "rust" in self.model.names[i] or "blight" in self.model.names[i] or "rot" in self.model.names[i] or "spot" in self.model.names[i]:
                self.model.names[i] = 'Skoriasis'
        results = self.model(self.images_to_detect)
        results.print()
        results.save(save_dir='detection_results/'+str(self.mission))
        return results.pandas()