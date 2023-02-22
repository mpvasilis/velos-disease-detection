import os
import shutil
import json
from PIL import Image
import requests
import logging

from helpers.Annotations import coco_to_yolo, classes


class Train:

    def __init__(self, url, jwt):
        self.url = url
        self.jwt = jwt
        self.responseUAV = None
        self.responseUGV = None
        self.classes = []
        self.annotations = []
        log = logging.getLogger("VELOS")
        log.debug("API initialised to", self.url)

    def GetUAVImageDetails(self):
        url = self.url + "/api/UAVImage/GetUAVImageDetails"
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        self.responseUAV = response.json()
        return response.json()

    def DownloadUAVImage(self, ImageName):
        url = self.url + "/api/UAVImage/Download?ImageName=" + str(ImageName)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        if response.status_code == 200:
            with open("./downloads/train/data/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True

    def DownloadUAVImages(self):
        images = []
        if self.responseUAV is not None:
            if not os.path.exists("./downloads/train/data/"):
                os.makedirs("./downloads/train/data/")
            print(self.responseUAV)
            for image in self.responseUAV:
                image_path = "./downloads/train/data/" + image['filename']
                print(image['annotation'])
                im = Image.open(image_path)
                image_width, image_height = im.size
                f = open("./downloads/train/data/"+image['filename'][:len(image['filename']) - 3] + "txt", "w")
                for annotation in json.loads(image['annotation']):
                    # print(annotation)
                    # print(annotation['mark']['x'])
                    # print(annotation['mark']['y'])
                    # print(annotation['mark']['width'])
                    # print(annotation['mark']['height'])
                    yolo = coco_to_yolo(float(annotation['mark']['x']), float(annotation['mark']['y']),
                                        float(annotation['mark']['width']), float(annotation['mark']['height']),
                                        image_width, image_height)  # <[98 345 420 462] (322x117) | Image: (?x?)>

                    if 'comment' in annotation:
                        print(classes[annotation['comment']], yolo)
                        f.write(classes[annotation['comment']]+" "+" ".join(yolo))
                    else:
                        print("Annotation", annotation['id'], "of image", image['filename'], "does not have class")
                f.close()
                if not os.path.exists(image_path):
                    self.DownloadUAVImage(image['filename'])
                if os.path.exists(image_path):
                    images.append(image_path)
        else:
            raise Exception("[DownloadUAVImages] Empty response.")
        return images

    def DownloadUGVImages(self):
        images = []
        if self.responseUGV is not None:
            if not os.path.exists("./downloads/train/data/"):
                os.makedirs("./downloads/train/data/")
            print(self.responseUGV)
            for image in self.responseUGV:
                image_path = "./downloads/train/data/" + image['filename']
                if not os.path.exists(image_path):
                    self.DownloadUGVImage(image['filename'])
                if os.path.exists(image_path):
                    images.append(image_path)
        else:
            raise Exception("[DownloadUGVImages] Empty response.")
        return images

    def GetUGVImageDetails(self):
        url = self.url + "/api/UGVImage/GetUGVImageDetails"
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        self.responseUGV = response.json()
        return response.json()

    def DownloadUGVImage(self, ImageName):
        url = self.url + "/api/UGVImage/Download?ImageName=" + str(ImageName)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        if response.status_code == 200:
            with open("./downloads/train/data/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True
