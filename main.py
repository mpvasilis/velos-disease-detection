import json
import os
from flask import Flask, request, jsonify
app = Flask(__name__)
from dotenv import load_dotenv
from helpers.API import API
import logging
from helpers.detect import DiseaseDetection
from flask import jsonify

load_dotenv()
API_BASE_URL = os.getenv('API_BASE_URL')
JWT = os.getenv('JWT')
logging.basicConfig(level=os.getenv('LOGLEVEL'))
log = logging.getLogger("VELOS")

@app.route('/detectDieasesForMission')
def detectDieasesForMission():
    mission = request.args.get('mission')
    api = API(API_BASE_URL, JWT)
    UAVImages = api.GetUAVImageDetails(mission)
    if len(UAVImages)>0:
        images = api.DownloadUAVImages()
        disease_detection = DiseaseDetection('ultralytics/yolov5', 'yolov5s',images)
        results = disease_detection.detect()
        print(results.xywh)
        return jsonify("Added to queue") #TODO: Queue will be implemeted
    else:
        return jsonify("Empty image list")



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
