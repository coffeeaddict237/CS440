from collections import deque
import numpy as np
import cv2
#from matplotlib import pyplot as plt

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
pts = deque(maxlen=16)
cap = cv2.VideoCapture(0)

def isThumbsUp(w,h):
    if w in range(100, 200) and h in range(220, 320):
        return True
    return False

def isHeart(w,h):
    if w in range(300, 400) and h in range(55, 170):
        return True
    return False



def isWave(pts):
    if list(pts)[0] is None: 
        return False

    start = pts[0]
    end = pts[len(pts)-1]

    if start is None or end is None:
        return False

    if np.abs(start[1] - end[1]) in range(50):
        # check if distance is greater than 400 pixels:
        if np.abs(start[0] - end[0]) > 250:
            return True

    return False


while(True):

    ret, frame = cap.read()

    if not ret:
    	break

    # frame = imutils.resize(frame, width=600)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # static gesture:
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    #cv2.imshow('frame2', skin)

    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    #create a matrix of countours (triples of their boundaries)
    contours = cv2.findContours(skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #check detected object
    for c in contours[1]:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        #print cv2.contourArea(c)
        x,y,w,h = rect
        #print(x,y,w,h) 
        
        #step to print to screen
        if isThumbsUp(w,h):
            #GOOD JOB! You detected a thumbs up!
            cv2.putText(frame,'GOOD JOB!',(x+w+10,y+h),0,3,(255,255,255)) 

        if isHeart(w,h):
            #THANK YOU! I love you too!
            cv2.putText(frame,'THANK YOU!',(x+w+10,y+h),0,3,(255,255,255)) 
        
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

    # step to remove unwanted erosions
    mask = cv2.inRange(converted, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only continue if gesture is found
    # step to find contour and compute minimum enclosing circle and centroid
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # show the frame to my screen
    pts.appendleft(center)

    for i in range(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None:
            continue
        #debugging
        #print('current: ' + str(pts[i]))
        #thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
        #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    #print(pts)
    
    if isWave(pts):
        cv2.putText(frame,'HELLO',(100,100),0,5,(255,255,255)) 
    # Display
    cv2.imshow('frame', frame)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close everything
cap.release()
cv2.destroyAllWindows()