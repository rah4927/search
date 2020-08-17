#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:41:18 2019

@author: rah4927
"""

import multiprocessing
from gensim.models import Word2Vec
import tensorflow as tf
import argparse 
import urllib.request as urllib2  # the lib that handles the url stuff
from tensorflow.python.lib.io import file_io

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

def model(sentence):
    w2v_model = Word2Vec(min_count=1, # let's keep all the asins
                         window=2, # 2 is fine, we only have two each line anyways
                         size=64, # start small for faster training, we'll increase later
                         sample=1e-5,  # let's keep 1e-5, we'll tweak later
                         alpha=0.03, 
                         min_alpha=0.03, # approx_epoch, 
                         negative=5, # keep small
                         workers=cores-1)
    
    
    w2v_model.build_vocab(sentence, progress_per=10000)
    w2v_model.init_sims(replace=True)
    return w2v_model

def main(job_dir,**args):
    with tf.device('/device:GPU:0'):
        # read from file
        sentence = []
                
        target_url = "https://00e9e64bac33fe0c7c6d29dd2eeb30708944964417a34175fe-apidata.googleusercontent.com/download/storage/v1/b/word2vec-test/o/asin_pairs.txt?qk=AD5uMEuaW22pTOang5kjorT1CRv_-Fi67pWFSastu1dxQxNVGwJV_yyv5c0CL-chMubw-JiF_XohvQ59_KohE7mCiOOcK1Sv7GAxgk-RgDaoqz1LmCXfJDxykUCzbp14Rww4srgqBILjeXh0Yz4rIQQb-E2Isqorsq4XKVexzXLOKV4HRGH9czoMdQepHBHJQxghR7BotlZ5pqkoIKisk3zZnY4eNXubA0aLB1hBlR1tFHrIAVLixb60sj1FS18v3uQS_bVKMMcVQPDu1HhhgIFksnOoBDokc1g5lbcYAN9Sh1OcexDUHS3EbuADM7Ax20n9fOj7IZpDB8Ck_FYkHQB9_ZzJ6Qwo7j4m7KJP8Hq-JuhiW5qTuXEl1HWqy2qEvqrJznorDK2edxFup-CJHOr59OF6qqQs0CVE6TjyCCNyNlCEC72laaQePFwjSiERt8gSuwXwZLv5T1XoiIM-Q-IySjqAiQAE_jyU-BGFQ_IYyWLDXKSpHAnEf9P7jJox3pmTHFd0ODTa9C8DgtB3ZnxAwvXmZshXTF9TwAlUXvAa7lbHuYRiYoA8X1IdLRg3K9dLpZi-OTZRLGjMjoPbORzq63HgwrY4UGbAzFHTT5_gBumvnXr3Q3OK3hexTYxdAjZTD0jgKMFnc7tbEWcwwxah7S562UQo1j9YKP9LBbTCZd6OX9adAfxKIuv8jV7tDZ2DYL4Fyh1xXARXQUC4PDoWYmvbD-J6bGNgpsp22bHAMZRFjjk7xhz3-hI5l6qbBu8xFzQnJ2lAzWCYDZ840YkzYj-5vqNhmg"
        data = urllib2.urlopen(target_url) # it's a file like object and works just like a file
        
        for line in data: # files are iterable
            line = "".join( chr(x) for x in line)
            a = line.split(',')
            li = []
            for x in a:
                li.append(x.strip())
            sentence.append(li)
        
        w2v_model = model(sentence) 
        w2v_model.save('w2v_model')
        
        # Save model on to google storage
        with file_io.FileIO('w2v_model', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/w2v_model', mode='w+') as output_f:
                output_f.write(input_f.read())
        

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)