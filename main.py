import json
import os
from flask import Flask, request, jsonify

from image_preprocessing.image_preprocessing import parallel_preprocessing
from train import train_yolov5

app = Flask(__name__)
from dotenv import load_dotenv
from helpers.API import API
import logging
from helpers.detect import DiseaseDetection
from flask import jsonify
from helpers.Train import Train

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
        disease_detection = DiseaseDetection(images,mission)
        results = disease_detection.detect()
        print(results.xywh)
        return jsonify("Added to queue") #TODO: Queue will be implemeted
    else:
        # disease_detection = DiseaseDetection(["downloads/train/images/2_10107DJI_0274.JPG"], 1)
        # results = disease_detection.detect()
        # print(results.xywh)
        return jsonify("Empty image list")



@app.route('/train', methods=['GET'])
def train():
    checkpoint = request.args.get('checkpoint')
    input_pre = 'train'
    output_pre = 'combined_methods'
    print("Downloading UAV images....")
    api = Train(API_BASE_URL, JWT)
    api.GetUAVImageDetails()
    result = api.DownloadUAVImages()
    api.GetUGVImageDetails()
    print("Downloading UGV images....")
    result = api.DownloadUGVImages()
    print(result)
    print("Preprocessing....")
    parallel_preprocessing(['get_Laplacian','get_crop'],input_pre,output_pre)
    print("Training model....")
    train_yolov5(output_pre, checkpoint)
    return jsonify("Training started.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
