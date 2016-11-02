#!/usr/bin/env python

import threading
import time
import numpy as np
import VideoThread
import os
import glob
import caffe

class AgeThread(threading.Thread):
    def __init__(self, videoThread):
        
        threading.Thread.__init__(self)
        
        print "Initializing age recognition thread..."
        self.videoThread = videoThread
        
	#caffe.set_mode_cpu()
        caffe.set_mode_gpu()
    
        # Model file and parameters are written by trainDnn.py   
        # Take the most recent parameter set       
        dcnnPath = "./dcnn_age"
        paramFiles = glob.glob(dcnnPath + os.sep + "*.caffemodel")
        paramFiles = sorted(paramFiles, key=lambda x:os.path.getctime(x))
        
        MODEL_FILE = dcnnPath + os.sep + "deploy.prototxt"
        PRETRAINED = paramFiles[-1]
        MEAN_FILE = dcnnPath + os.sep + "mean.binaryproto"
        
        blob = caffe.proto.caffe_pb2.BlobProto()
        with open(MEAN_FILE, 'rb') as f:
            data = f.read()
            
        blob.ParseFromString(data)
        # mean = np.array( caffe.io.blobproto_to_array(blob) ) [0]
        # Added simple mean
        mean = np.array([93.5940, 104.7624, 129.1863])
    
        # Initialize net             
        self.net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(224,224), mean=mean)


    def run(self):
        caffe.set_mode_gpu()

        while self.videoThread.isTerminated() == False:
	
	    while self.videoThread.isTerminated() == False and self.videoThread.getEventReady() == True:
                time.sleep(0.1)
                print("Age recognition sleep")
 	
            #print "Detecting..."
            crop = None
            
            while crop == None:
                
                crop, rectangle = self.videoThread.getCropEx(0)
                time.sleep(0.05)
                if crop == None: # No crops available yet
                    time.sleep(0.1)
    
            crop = crop.astype(np.float32)
            
            out = self.net.predict([crop], oversample = False).ravel()
            age = np.dot(out, range(101))
            
            self.videoThread.setAge(age)
            
            
	    



