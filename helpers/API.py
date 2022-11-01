import os
import shutil
import json

import requests
import logging

class API:

    def __init__(self, url, jwt):
        self.url = url
        self.jwt = jwt
        self.responseUAV = None
        self.mission = None
        log = logging.getLogger("VELOS")
        log.debug("API initialised to",self.url)

    def GetUAVImageDetails(self, mission):
        self.mission = str(mission)
        url = self.url + "/api/UAVImage/GetUAVImageDetails?mission=" + str(mission)
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
            with open("./downloads/"+self.mission+"/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True

    def DownloadUAVImages(self):
        images = []
        if self.responseUAV is not None:
            if not os.path.exists("./downloads/"+self.mission+"/"):
                os.makedirs("./downloads/"+self.mission+"/")
            print(self.responseUAV)
            for image in self.responseUAV:
                image_path = "./downloads/" + self.mission + "/"+image['filename']
                if not os.path.exists(image_path):
                    self.DownloadUAVImage(image['filename'])
                if os.path.exists(image_path):
                    images.append(image_path)
        else:
            raise Exception("[DownloadUAVImages] Empty response from mission",self.mission)
        return images



    def GetUGVImageDetails(self, mission):
        url = self.url + "/api/UGVImage/GetUGVImageDetails?mission=" + str(mission)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        return response.json()

    def DownloadUGVImage(self, ImageName):
        url = self.url + "/api/UGVImage/Download?ImageName=" + str(ImageName)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        if response.status_code == 200:
            with open("./downloads/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True
