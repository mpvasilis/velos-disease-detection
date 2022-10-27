from unittest import TestCase
from helpers import API


class Test(TestCase):
    def test_get_uavimage_details(self):
        result = API.GetUAVImageDetails(1)
        print(result)

    def test_download_uavimage(self):
        result = API.DownloadUAVImage("100_27122DJI_0289.JPG")
        print(result)