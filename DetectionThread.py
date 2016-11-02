#!/usr/bin/env python

import cv2
import threading
import time
import random
import VideoThread


class DetectionThread(threading.Thread):
    def __init__(self, videoThread):
        
        threading.Thread.__init__(self)
        time.sleep(random.random())
        print "Initializing detection thread..."
        self.videoThread = videoThread
        
        # Create the haar cascade
        cascPath = "./cascades/haarcascade_frontalface_alt.xml"
        self.detector = cv2.CascadeClassifier(cascPath)
            
        # Main loop:
        self.numDetections = 0
        self.numNoDetections = 0
        
    def run(self):
       
        while self.videoThread.isTerminated() == False:
        
            #print "Detecting..."
            frame = None
            
            while frame == None:
                frame = self.videoThread.getFrameEx()
                time.sleep(0.05) # delay added
                if frame == None: # No frames available yet
                    time.sleep(0.1)
                    print("Detection: sleep") 
                    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces

            boxes = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(100, 100),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            detections = []
            
            for box in boxes:
                x,y,w,h = box
                
                # take 40% more
                x_more = min(x-max(x-0.2*w, 0), min(x+0.2*w, frame.shape[1])-x)
                y_more = min(y-max(y-0.2*h, 0), min(y+0.2*h, frame.shape[0])-y)
                x -= x_more
                y -= y_more
                w += x_more*2
                h += y_more*2

                crop = frame[y:y+min(h,w), x:x+min(h,w), ...]
                detections.append((box, crop))

            if len(detections) > 0:
                self.videoThread.setDetections(detections)
            else:
                self.videoThread.setEventReady(False)
                self.videoThread.setCounter()
                  
            
