#!/usr/bin/env python

import threading
import time
import numpy as np
import VideoThread
import os
import glob
import caffe

class GenderThread(threading.Thread):
    def __init__(self, videoThread):
        
        threading.Thread.__init__(self)
        
        print "Initializing recognition thread..."
        self.videoThread = videoThread
        
	#caffe.set_mode_cpu()
        caffe.set_mode_gpu()
        caffe.set_device(0)
    
        # Model file and parameters are written by trainDnn.py  
        # Take the most recent parameter set

	genderPath = "./dcnn_gender"
	genderParamFiles = glob.glob(genderPath + os.sep + "*.caffemodel")
        genderParamFiles = sorted(genderParamFiles, key=lambda x:os.path.getctime(x))

	MODEL_FILE_GENDER = genderPath + os.sep + "deploy_gender.prototxt"
        PRETRAINED_GENDER = genderParamFiles[-1]
        MEAN_FILE_GENDER = genderPath + os.sep + "mean.binaryproto"
		 
	proto_data = open(MEAN_FILE_GENDER, 'rb').read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean  = caffe.io.blobproto_to_array(a)[0]

        # Initialize net             
        self.gender_net = caffe.Classifier(MODEL_FILE_GENDER, PRETRAINED_GENDER, image_dims=(227,227),)
        

    def run(self):
        caffe.set_mode_gpu()
       
        while self.videoThread.isTerminated() == False:
	
	    while self.videoThread.isTerminated() == False and self.videoThread.getEventReady() == True:
                time.sleep(0.1)
                print("Gender recognition sleep")
 	
            #print "Detecting..."
            crop = None
            
            while crop == None:
                
                crop, rectangle = self.videoThread.getCropEx(1)
                time.sleep(0.05)
                if crop == None: # No crops available yet
                    time.sleep(0.1)
    
            crop = crop.astype(np.float32)
     
            propabilities = self.gender_net.predict([crop], oversample = False).ravel() #[Male, Female]
            self.videoThread.setGender(propabilities)
            
	    


