#!/usr/bin/env python

import cv2
import cv2.cv as cv
from copy import deepcopy
import threading
import DetectionThread
import AgeThread
import GenderThread
import time
import os
import glob

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, age, gender, gender_label, counter, COUNTER_LIMIT, thickness = 1):
    if counter >= COUNTER_LIMIT:
        vari = (0, 255, 0)
        message = "Ready"
    else:
        vari = (0, 0, 255)
        message = "Recognizing..."    

    for x, y, w, h in rects:
        
        cv2.rectangle(img, (x, y), (x+w, y+h), vari, thickness)        
        
        #print "rectangle drawn: ", (x,y,w,h)

        if age is not None:
	    #print age
            annotation = "Age: %.0f" % (age)
            txtLoc = (x + 5,y + 30)
            
            cv2.putText(img, 
                        annotation, 
                        txtLoc, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        vari,
			2)
            
            cv2.putText(img, 
                        message, 
                        (x, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        vari,
			2)

        if gender is not None:
	    #print age
            annotation = "%.0f %%" % (gender)
            txtLoc = (x,y + h + 60)
            
            cv2.putText(img, 
                        annotation, 
                        txtLoc, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        vari,
			2)
            
            cv2.putText(img, 
                        "Gender:", 
                        (x, y + h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        vari,
	    		2)
            
            cv2.putText(img, 
                        gender_label, 
                        (x + 100, y + h + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        vari,
	    		2)  
                                
    return img
    
class VideoThread(threading.Thread):
     def __init__(self):
        threading.Thread.__init__(self)
        self.windowCaption = "Age Estimation in the Wild"
        cv2.namedWindow(self.windowCaption)
        
        self.mutex = threading.Lock()
        self.cropMutex = threading.Lock()
        self.dataMutex = threading.Lock()
        self.crops = []
        self.rectangles = []
        self.lastDetection = time.time();
        self.detections = []
        self.frames = []
        self.frameRate = 0
        self.detectionRate = 0  
        self.rateInterval = 10.0
        self.isDetected = False
        self.frame = None
        self.frameEx = None
        self.age = 0
        self.ages = []
        self.gender_male = []
        self.gender_female = []
        self.gender = 0
        self.gender_label = "" 
        self.counter = 0
        self.inputmode = "webcam"
	self.downsampleFactor = 2
	self.COUNTER_LIMIT = 20 # Number of predictions used to calculate average age
        # self.ROI = [560, 0, 800, 1080] # x, y, w, h  #1920x1080
        #self.ROI = [0, 0, 1280, 720] # x, y, w, h      #1280x720
        self.ROI = [0, 0, 640, 480] # x, y, w, h      #640x480

        # When eventReady is true, program stops making new estimations and ui box color changes to green
        self.eventReady = False
        
        print "Initializing video capture..."
        
        if self.inputmode == "webcam":
            self.video = cv2.VideoCapture(0) # 0: Laptop camera, 1: USB-camera 
            #self.video.set(3, 1280) #1280 #1920 Default: 640
            #self.video.set(4, 720)  #720  #1080 Default: 480

            self.video.set(3, 640) #1280 #1920 Default: 640
            self.video.set(4, 480)  #720  #1080 Default: 480
        elif self.inputmode == "ueye":
            self.video = pyuEye.pyuEye()    
        elif self.inputmode == "file":
            inputPath = config.get("General", "inputpath", ".")
            self.fileList = glob.glob(os.path.join(inputPath, "*.jpg"))
                    
        self.terminated = False
        
        threadPool = []
        
        for detectionThreads in range(1):
            detectionThread = DetectionThread.DetectionThread(self)
            detectionThread.start()
            threadPool.append(detectionThread)                

        recThreadPool = []
        
        for ageThreads in range(1):
            ageThread = AgeThread.AgeThread(self)
            ageThread.start()
            recThreadPool.append(ageThread)
       
        genderThreadPool = []  

        for genderThreads in range(1):
            genderThread = GenderThread.GenderThread(self)
            genderThread.start()
            genderThreadPool.append(genderThread) 
            
     def annotate(self, frame):
         text = "Frame rate: %.1f" % self.frameRate
         textColor = (0,255,0)
         font = cv2.FONT_HERSHEY_SIMPLEX
         size = 0.5
         thickness = 2
         textSize = cv2.getTextSize(text, font, size, thickness)
         height = textSize[1]         
         location = (0,frame.shape[0] - 4*height)
         cv2.putText(frame, text, location, font, size, textColor,
            thickness=thickness)
                     
         text = "Detection rate: %.1f" % self.detectionRate
         location = (0,frame.shape[0] - height)
         cv2.putText(frame, text, location, font, size, textColor,
            thickness=thickness)
         
     def run(self):
	 
         while not self.terminated:

             self.grab()    
  
             # if in file mode, wait until the grabbed frame is detected.
             if self.inputmode == "file":
                 while self.isDetected == False:
                     time.sleep(0.1)
                     
             frame = self.getFrame()
             if len(frame.shape) == 2:
                 frame = cv2.cvtColor(frame, cv.CV_GRAY2RGB)

             self.dataMutex.acquire()

             if len(self.ages) > 0:
                 self.age = sum(self.ages)/len(self.ages)            

	     if len(self.gender_male) > 0 and len(self.gender_female) > 0:
             	male = (sum(self.gender_male)/len(self.gender_male))*100
                female = (sum(self.gender_female)/len(self.gender_female))*100
                if male > female:
                    self.gender_label = "Male"
                    self.gender = male
                else:
                    self.gender_label = "Female"
                    self.gender = female 

             self.dataMutex.release()		      
             boxAge = time.time() - self.lastDetection          
             crop = None
             seconds = 2         

             if boxAge < seconds and self.rectangles is not None and len(self.rectangles) > 0:           
                 x,y,w,h = self.rectangles[0]              
                 frame = draw_detections(frame.astype('uint8'), self.rectangles, self.age, self.gender, self.gender_label, self.counter, self.COUNTER_LIMIT, 3) 
             
             if self.ROI is not None:
                 x, y, w, h = self.ROI
                 thickness = 2
                 frame = frame.astype('uint8')
                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness)        

             self.annotate(frame)
             cv2.imshow(self.windowCaption, frame)
             
             # User input:
             key = cv2.waitKey(1)             	
                 
             if key == 1048603: #ESC(27) #Exit program
                 self.terminated = True
	     elif key == 1048586: # Enter(10) #Reset predictions
	         self.setEventReady(False)
                 self.setCounter()
  
     def grab(self):                  
         
         if self.inputmode == "webcam":
            stat, frame = self.video.read()
            frame = frame[:, ::-1, ...]
         elif self.inputmode == "ueye":
            frame = self.video.grab()
         elif self.inputmode == "file":
            # Do not advance to next file until detected.
            while self.isDetected == False:
                time.sleep(0.1)
                
            frame = cv2.imread(self.fileList[0])
            self.fileList = self.fileList[1:]
         
         self.mutex.acquire()         
         self.frame = frame
         self.isDetected = False
         self.mutex.release()
         
         timeNow = time.time()
         self.frames.append(timeNow)
         self.frames = [x for x in self.frames if timeNow - x < self.rateInterval]
         self.frameRate = len(self.frames) / self.rateInterval
         
     def getFrame(self):
         self.mutex.acquire()
         result = self.frame
         self.mutex.release()
         return result

     def getFrameEx(self):
         if self.isDetected == True:
             return None # Another thread has detected this already.

         self.mutex.acquire()
         result = deepcopy(self.frame) 
         self.isDetected = True # Flag this frame so no other thread receives the same picture.
         self.mutex.release()

	 if result is not None:
             x, y, w, h = self.ROI
             result = result[y:y+h, x:x+w, ...]
	     result = cv2.resize(result, (result.shape[1] / self.downsampleFactor, result.shape[0] / self.downsampleFactor))

         return result

     def getCropEx(self, purpose):
         
         crop = None
         rectangle = None
         
         self.cropMutex.acquire()

         if self.crops is not None and len(self.crops) > 0 and not self.processed[purpose]:
             crop = deepcopy(self.crops[0])
             rectangle = deepcopy(self.rectangles[0])
             self.processed[purpose] = True

         self.cropMutex.release()
         
         return (crop, rectangle)

     def setAge(self, age):
         self.ages.append(age)
     
     def setGender(self, genders):
         self.gender_male.append(genders[0])
         self.gender_female.append(genders[1])      
        
     def setDetections(self, detections):       
         rectangles = []
         crops = []

         for detection in detections:        
	     scaledBox = [c * self.downsampleFactor for c in detection[0]]
             scaledBox[0] += self.ROI[0]
             scaledBox[1] += self.ROI[1]

             rectangles.append(scaledBox)
             crops.append(detection[1])
             #cv2.imwrite('aux.jpg', detection[1])

         self.cropMutex.acquire()
         self.rectangles = rectangles
         self.crops = crops
         self.processed = [False]*2
         self.cropMutex.release()
                  
         # Store timestamps of finished frames for estimating framerate
         timeNow = time.time()
         self.detections.append(timeNow)
         self.lastDetection = timeNow
         
         self.detections = [x for x in self.detections if timeNow - x < self.rateInterval]
         self.detectionRate = len(self.detections) / self.rateInterval

         if self.counter < self.COUNTER_LIMIT:
             self.counter += 1
             print(self.counter)
         else:
             self.setEventReady(True)
        
     def isTerminated(self):
         return self.terminated
     
     def getEventReady(self):
         return self.eventReady

     def setEventReady(self, status):
         self.eventReady = status
 
     def setCounter(self):
         self.dataMutex.acquire()
         self.counter = 0

	 if len(self.ages) > 0:	
             self.age = self.ages[-1]
         else:
	     self.age = 0
 
         self.ages = []
         self.gender_male = []
         self.gender_female = []
         self.dataMutex.release()
                   
