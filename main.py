import os
from flask import Flask, request, jsonify
app = Flask(__name__)
from dotenv import load_dotenv
from helpers.API import API
import logging

log = logging.getLogger("VELOS")

@app.route('/receiveCallForDetection')
def receiveCallForDetection():
    mission = request.args.get('mission')
    return mission


if __name__ == '__main__':
    load_dotenv()
    API_BASE_URL = os.getenv('API_BASE_URL')
    JWT = os.getenv('JWT')
    api = API(API_BASE_URL,JWT)
    logging.basicConfig(level=os.getenv('LOGLEVEL'))
    app.run(host='0.0.0.0', threaded=True, debug=True)
