import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

fe = FeatureExtractor()
last = time.time()
for i, img_path in enumerate(sorted(glob.glob('static/img/*.jpg'))):
    print(i, img_path)
    elapsed = last - time.time();
    print(-elapsed, -i / elapsed)
    try:
        img = Image.open(img_path)  # PIL image
        feature = fe.extract(img)
        feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
        pickle.dump(feature, open(feature_path, 'wb'))
    except Exception as e:
        print(e)
        