import shutil

import requests
import logging

class API:

    def __init__(self, url, jwt):
        self.url = url
        self.jwt = jwt
        log = logging.getLogger("VELOS")
        log.debug("API initialised to",self.url)

    def GetUAVImageDetails(self, mission):
        url = self.url + "/api/UAVImage/GetUAVImageDetails?mission=" + str(mission)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        return response.json()

    def DownloadUAVImage(self, ImageName):
        url = self.url + "/api/UAVImage/Download?ImageName=" + str(ImageName)
        payload = {}
        headers = {
            'Authorization': 'Bearer ' + self.jwt
        }
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        if response.status_code == 200:
            with open("../downloads/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True

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
            with open("../downloads/" + ImageName, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
        return True
