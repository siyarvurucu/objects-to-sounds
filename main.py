from __future__ import print_function
import sys
import cv2
import mido
import numpy as np
from random import random, randint

import matplotlib.pyplot as plt

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker


# Set video to load
videoPath = "singingobjects.avi"                           # local video file
#videoPath = "http://192.168.1.4:4747/mjpegfeed?640x480"   # this is for droidcam app
#videoPath = "http://10.42.0.186:4747/mjpegfeed?640x480"   # this is for droidcam app
#videoPath = 0                                             # this is for webcam. sometimes it is on '1'
# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)
  
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

# connect to Puredata
outport = mido.open_output('Pure Data:Pure Data Midi-In 1 128:0')


## Select boxes
bboxes = []
colors = [] 


# Specify the tracker type
trackerType = "TLD"    

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
 

def drum(centers,sizes):
    # creating drum messages according to center distances and standard deviations of object sizes 
    drummessages = []
    msg = mido.Message('control_change',channel=1,control=24,value = 16) # resets drum layout
    drummessages.append(msg)
    # print("sizes:")
    # print(sizes)
    sizes_diff = np.abs(sizes-np.roll(sizes,1))
    # print("sizes diff:")
    # print(sizes_diff)
    if centers.shape[1]==1:
        centers_diff = np.array([0])
    else:
        # print(centers)
        # print(centers-np.roll(centers,1,axis=0)**2)
        centers_diff = np.sqrt(np.sum((centers-np.roll(centers,1,axis=1))**2,axis=0))/30
    stds = 1+np.sqrt(sizes_diff)/20
    x = np.arange(-16,32,1)
    if sizes.shape[0]==2:
        centers_diff = [centers_diff[0]]
        sizes = sizes[0]
        sizes_diff = sizes_diff[0]
        stds = [stds[0]]
    for i,mu,sig in zip(np.arange(len(stds)),centers_diff,stds):
        print("instrument no:"+ str(i))
        print("mu")
        print(mu)
        print("sig")
        print(sig)
        y = sig*gaussian(x,mu,sig)
        y = np.sum(y.reshape((-1,16)),axis=0)
        for j in range(16):
            if random()<y[j]:
                msg = mido.Message('control_change',channel=1,control=i,value = j)
                drummessages.append(msg)
    
    return drummessages
    
def generateMessages(boxes,count):
    boxes = np.asarray(boxes)
    centers = np.array([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
    #sizes = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    sizes = (boxes[:,2])*(boxes[:,3])
    messages = []
    if count==0:
        messages.extend(drum(centers,sizes))
    #msg = mido.Message('note_on', note=int(abs(boxes[0][0]/5)))
    #messages.append(msg)
    return messages
    
def getObject(frame):
    while True:
      # draw bounding boxes over objects
      # selectROI's default behaviour is to draw box starting from the center
      # when fromCenter is set to false, you can draw box starting from top left corner
      bbox = cv2.selectROI('MultiTracker', frame)
      bboxes.append(bbox)
      colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
      multiTracker.add(createTrackerByName(trackerType), frame, bbox)

      print("Press q to quit selecting boxes and start tracking")
      print("Press any other key to select next object")
      k = cv2.waitKey(0) & 0xFF
      if (k == 113):  # q is pressed
        break

def measure(frame):
    success, boxes = multiTracker.update(frame)
    points = []
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        cv2.imshow('MultiTracker', frame)
    boxes = np.asarray(boxes)
    print("boxes:")
    print(boxes)
    centers = np.array([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
    sizes = (boxes[:,2])*(boxes[:,3])
    sizes_diff = np.abs(sizes-np.roll(sizes,1))
    
    print("centers.shape:" + str(centers.shape)) 
    if centers.shape[1]==1:
        centers_diff = np.array([0])
    else:
        centers_diff = np.sqrt(np.sum((centers-np.roll(centers,1,axis=1))**2,axis=0))/30 
    print(centers)
    print(centers_diff)
    x = np.arange(-16,32,1)
    stds = 1+np.sqrt(sizes_diff)/20
    if sizes.shape[0]==2:
        centers_diff = [centers_diff[0]]
        sizes = sizes[0]
        sizes_diff = sizes_diff[0]
        stds = [stds[0]]
    

    print("centers distances:")
    print(centers_diff)
    print("sizes:")
    print(sizes)
    print("sizes difference:")
    print(sizes_diff)
    print("stds:")
    print(stds)
    for i,mu,sig in zip(np.arange(len(stds)),centers_diff,stds):       
        y = sig*gaussian(x,mu,sig)
        y = np.sum(y.reshape((-1,16)),axis=0)
        plt.plot(y)
        plt.show()
    while True:
        print("Press 'm' to continue")
        k = cv2.waitKey(0) & 0xFF
        if (k == 109):  # m
            break

def show():
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # get updated location of objects in subsequent frames
        
        success, boxes = multiTracker.update(frame)
        points = []

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        cv2.imshow('MultiTracker', frame)  
        
        
        if boxes != ():
            count += 1
            if count%40==0:
                messages = generateMessages(boxes,count%40)
                for msg in messages:
                    #print(msg)
                    outport.send(msg)
                
        k = cv2.waitKey(10) & 0xFF  # (msec) delay
        
        if (k == 103):  # g is pressed
            getObject(frame)
        if (k == 113):  # q is pressed
            break
        if (k == 109):  # m is pressed
            measure(frame)
print("Press g to select object to track")
show()

#print('Selected bounding boxes {}'.format(bboxes))




# Initialize MultiTracker 
# for bbox in bboxes:
  # multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Process video and track objects
# while cap.isOpened():
  # success, frame = cap.read()
  # if not success:
    # break
  
  # # get updated location of objects in subsequent frames
  # success, boxes = multiTracker.update(frame)

  # # draw tracked objects
  # for i, newbox in enumerate(boxes):
    # p1 = (int(newbox[0]), int(newbox[1]))
    # p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    # cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    # #msg = mido.Message('note_on', note=newbox[0])
    # #outport.send(msg)
  # # show frame
  # cv2.imshow('MultiTracker', frame)
  

  # # quit on ESC button
  # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    # break


