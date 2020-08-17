import os
os.environ["GEVENT_RESOLVER"] = "ares"

from gevent import monkey
monkey.patch_all()

import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
import time
from datetime import datetime
from gevent.pywsgi import WSGIServer
from flask import Flask, request, render_template
import requests
from io import BytesIO

app = Flask(__name__)


fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
  features.append(pickle.load(open(feature_path, 'rb')))
  img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

@app.route('/', methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def index(path=None):
  local, serve = request.args.get('local', False), request.args.get('serve', '/')
  print(local, serve, path)

  if request.method == 'POST':
    file = request.files['query_img']
    img = Image.open(file.stream)
    img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
    img.save(img_path)
  elif path is None or path == 'favicon.ico':
    return render_template('index.html')
  else:
    if local:
      img_path = 'static/uploaded/' + path
      img = Image.open(img_path)
    else:
      response = requests.get(path)
      img = Image.open(BytesIO(response.content))
      img_path=path
  
  query = fe.extract(img)
  dists = np.linalg.norm(features - query, axis=1)
  ids = np.argsort(dists)[:30]
  scores = [(dists[id], serve + img_paths[id]) for id in ids]
  
  print(scores)
  
  return render_template('index.html',
                          ts=int(time.time()),
                          query_path=img_path,
                          scores=scores)


# @app.route('/dj/<path:path>', methods=['GET'])
# def dj(path):
#     if path == 'lucky':
#       img_path = '/static/uploaded/Q?t=' + datetime.now().isoformat()
#       img = Image.open('static/uploaded/Q')
#     else:
#       response = requests.get(path)
#       img = Image.open(BytesIO(response.content))
#       img_path=path
#     query = fe.extract(img)
#     dists = np.linalg.norm(features - query, axis=1)  # Do search
#     ids = np.argsort(dists)[:30] # Top 30 results
#     scores = [(dists[id], img_paths[id]) for id in ids]
#     print scores
#     return render_template('index.html',
#                             query_path=img_path,
#                             scores=map(lambda (a, b): (a, 'http://127.0.0.1:8080/'+b), scores))


# @app.route('/other/<path:path>', methods=['GET'])
# def other(path):
#     if path == 'lucky':
#       img_path = '/static/uploaded/Q?t=' + datetime.now().isoformat()
#       img = Image.open('static/uploaded/Q')
#     else:
#       response = requests.get(path)
#       img = Image.open(BytesIO(response.content))
#       img_path=path
#     query = fe.extract(img)
#     dists = np.linalg.norm(features - query, axis=1)  # Do search
#     ids = np.argsort(dists)[:30] # Top 30 results
#     scores = [(dists[id], '/' + img_paths[id]) for id in ids]
#     print scores
#     return render_template('index.html',
#                             query_path=img_path,
#                             scores=scores)

if __name__=="__main__":
    http_server = WSGIServer(('', 4005), app)
    http_server.serve_forever()
    # app.run("0.0.0.0", port=4005)
