import os
from unittest import TestCase
from dotenv import load_dotenv
from helpers.Train import Train


class TestTrain(TestCase):
    def test_download_all_images(self):
        load_dotenv()
        API_BASE_URL = os.getenv('API_BASE_URL')
        JWT = os.getenv('JWT')
        api = Train(API_BASE_URL,JWT)
        api.GetUAVImageDetails()
        result = api.DownloadUAVImages()
        api.GetUGVImageDetails()
        result = api.DownloadUGVImages()
        print(result)

