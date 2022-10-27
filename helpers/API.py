import shutil

import requests


def GetUAVImageDetails(mission):
    url = "http://83.212.19.14:8081/api/UAVImage/GetUAVImageDetails?mission=" + str(mission)
    payload = {}
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJuYW1laWQiOiJhaXVzZXIiLCJ1bmlxdWVfbmFtZSI6ImFpdXNlciIsInJvbGUiOiJBSXVzZXIiLCJuYmYiOjE2NjY4NjE2MzAsImV4cCI6MTY2NzQ3MDAzMCwiaWF0IjoxNjY2ODYxNjMwfQ.JlaSVlrCrAm6jYuAUbdjgTVQX7bTjfTk-hbWj1DBWN4nSzVTf-YPxQ6QEqjerl4nBKPDQb1kCv8yFKn7hIYnlg'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()


def DownloadUAVImage(ImageName):
    url = "http://83.212.19.14:8081/api/UAVImage/Download?ImageName=" + str(ImageName)
    payload = {}
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJuYW1laWQiOiJhaXVzZXIiLCJ1bmlxdWVfbmFtZSI6ImFpdXNlciIsInJvbGUiOiJBSXVzZXIiLCJuYmYiOjE2NjY4NjE2MzAsImV4cCI6MTY2NzQ3MDAzMCwiaWF0IjoxNjY2ODYxNjMwfQ.JlaSVlrCrAm6jYuAUbdjgTVQX7bTjfTk-hbWj1DBWN4nSzVTf-YPxQ6QEqjerl4nBKPDQb1kCv8yFKn7hIYnlg'
    }
    response = requests.request("POST", url, headers=headers, data=payload, stream=True)
    if response.status_code == 200:
        with open("../downloads/"+ImageName, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    return True
