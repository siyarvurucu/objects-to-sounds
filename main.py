from __future__ import print_function
import sys
import cv2
import mido
import numpy as np
from random import random, randint
import time
import matplotlib.pyplot as plt
from threading import Thread

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
    a = np.arange(-16,32,1)
    if sizes.shape[0]==2:
        centers_diff = [centers_diff[0]]
        sizes = sizes[0]
        sizes_diff = sizes_diff[0]
        stds = [stds[0]]
        
    for i,mu,sig in zip(np.arange(len(stds)),centers_diff,stds):
        # print("instrument no:"+ str(i))
        # print("mu")
        # print(mu)
        # print("sig")
        # print(sig)
        y = sig*gaussian(a,mu,sig)
        y = np.sum(y.reshape((-1,16)),axis=0)
        for j in range(16):
            if random()<y[j]:
                msg = mido.Message('control_change',channel=1,control=i,value = j)
                drummessages.append(msg)
    
    return drummessages

def lead(boxes,scale):
    boxes = np.asarray(boxes)
    centers = np.array([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
    msgs = []
    
    xyratios = boxes[:,2]/boxes[:,3]
    if len(xyratios)==1:
        xyratios_diff = xyratios*2
    else:
        xyratios_diff = 2*(xyratios[1:]-xyratios[:-1])
    centers_diff = np.sqrt(np.sum((centers-np.roll(centers,1,axis=1))**2,axis=0))
    if centers_diff.shape[0]==2:
        centers_diff = [centers_diff[0]]
    
    prev_note = 7
    octave = 0
    for i in range(16):
        center_diff = centers_diff[i%(len(xyratios_diff))]
        rest_probability = np.exp(-center_diff/347.43)
        # print(center_diff)
        # print("prob")
        # print(rest_probability)
        if random()>rest_probability:
            msg = mido.Message('note_off', channel=0)
            msgs.append(msg)
            msg = mido.Message('note_off', channel=0)
            msgs.append(msg)
            continue
            
        diff = xyratios_diff[i%(len(xyratios_diff))]
        sample = int(np.random.normal(diff,2,1))
        next_note = prev_note+sample
        while (next_note)>14:
            next_note = next_note-7
            octave += 1
        while (next_note)<0:
            next_note = next_note+7
            octave += -1
        if octave>2:
            octave = 0
        if octave<-1:
            octave = 0    
        msg = mido.Message('note_on', channel=0, note=48+scale[next_note]+12*octave,velocity=80) 
        msgs.append(msg)
        msg = mido.Message('note_off', channel=0, note=48+scale[next_note]+12*octave,velocity=80)
        msgs.append(msg)        
        prev_note = next_note
    return(msgs)

def chords(boxes,sizes,scale):
    onmsgs = []
    offmsgs = []
    xy = np.random.randint(2)
    centers = boxes[:,xy] + boxes[:,xy+2]/2
    for c in centers:
        sample = int(np.random.normal(4*(c-(240+xy*80))/(240+xy*80),2,1)) # index to pick a note from scale. depends on x position of object and sampled from normal dist   
        msg = mido.Message('note_on', channel=0, note=36+scale[sample]) 
        onmsgs.append(msg)
        msg = mido.Message('note_off', channel=0, note=36+scale[sample])
        offmsgs.append(msg)     
    return onmsgs,offmsgs
    
def generateMessages(boxes,scale):
    boxes = np.asarray(boxes)
    centers = np.array([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
    #sizes = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    sizes = (boxes[:,2])*(boxes[:,3])
    
    drummsg = drum(centers,sizes)
    chordsmsg_on,chordsmsg_off = chords(boxes,sizes,scale)
    #msg = mido.Message('note_on', note=int(abs(boxes[0][0]/5)))
    #messages.append(msg)
    return drummsg, chordsmsg_on,chordsmsg_off

def generateScale(boxes):
    major = np.array([0,2,4,5,7,9,11])
    major = np.concatenate((major-12,major,np.array([12])),axis=0)  # spanning 2 octaves intervals
    return major
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
        plt.title("probability distribution of intrument %d on 16 beats" %i)
        plt.show()
    while True:
        print("Press 'm' to continue")
        k = cv2.waitKey(0) & 0xFF
        if (k == 109):  # m
            break

def leadplay(leadmessages): 
    duration = [1]
    idur = 0
    offmsg = mido.Message('note_off', channel=0, note=60) # dummy off message
    while leadmessages:
        if time.time()%(duration[idur]*0.24)>((duration[idur]*0.24)-0.005) or time.time()%(duration[idur]*0.24)<0.005:           
            outport.send(offmsg)
            msg = leadmessages.pop(0)
            outport.send(msg)
            offmsg = leadmessages.pop(0)
            
            idur = (idur+1)%1
            time.sleep(0.01)
        else:
            time.sleep(0.01)
    outport.send(offmsg)  
    
def show():
    count = 0
    globstart = time.time()
    drumtime = globstart
    leadtime = globstart
    scaletime = globstart
    c = globstart + 16
    leadmessages = []
    chordsmsg_off = []
    #offmsg = mido.Message('note_off', channel=0, note=60) # dummy off message
    leadplayer= Thread(target=leadplay,args=(leadmessages,))
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
        
        if (c-scaletime)>15.36:
            scale = generateScale(boxes)
            scaletime = time.time()
           
        c = time.time() # current time
        
        if boxes != ():
            if (c-drumtime)>3.74:
                drumtime = time.time()
                for msg in chordsmsg_off:
                    outport.send(msg)
                drummessages, chordsmsg_on,chordsmsg_off = generateMessages(boxes,scale)
                for msg in drummessages:
                    outport.send(msg)
                for msg in chordsmsg_on:
                    outport.send(msg)   
        c = time.time() 
                    
        
        if boxes != ():  # creating new lead note messages
            if (c-leadtime)>(8*0.240):
                #outport.send(offmsg)
                leadtime = time.time()
                if len(leadmessages)<8:                   
                    leadmessages.extend(lead(boxes,scale))
                # else:
                    # msg = leadmessages.pop(0)
                    # outport.send(msg)
                    # offmsg = leadmessages.pop(0)
        
        
        
        if not leadplayer.is_alive():
            leadplayer= Thread(target=leadplay,args=(leadmessages,))
            leadplayer.start()           
        k = cv2.waitKey(10) & 0xFF  # (msec) delay
        
        if (k == 103):  # g is pressed
            getObject(frame)
        if (k == 113):  # q is pressed
            break
        if (k == 109):  # m is pressed
            measure(frame)
    
    while leadplayer.is_alive(): # wait for lead thread to push all messages
        time.sleep(0.1)
    msg = mido.Message('control_change',channel=0,control=24,value = 16) # disable drum machine
    outport.send(msg)
    for msg in chordsmsg_off:  # send any note off messages left
        outport.send(msg)
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


