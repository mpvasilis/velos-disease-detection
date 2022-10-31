import os
from unittest import TestCase
from dotenv import load_dotenv
from helpers.API import API


class Test(TestCase):
    def test_get_uavimage_details(self):
        load_dotenv()
        API_BASE_URL = os.getenv('API_BASE_URL')
        JWT = os.getenv('JWT')
        api = API(API_BASE_URL, JWT)
        result = api.GetUAVImageDetails(1)
        print(result)

    def test_download_uavimage(self):
        load_dotenv()
        API_BASE_URL = os.getenv('API_BASE_URL')
        JWT = os.getenv('JWT')
        api = API(API_BASE_URL, JWT)
        result = api.DownloadUAVImage("100_27122DJI_0289.JPG")
        print(result)