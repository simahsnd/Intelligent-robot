
import numpy as np
from Motor import *
import cv2

 

cap = cv2.VideoCapture(-1)

cap.set(3, 150)

cap.set(4, 110)

 
PWM=Motor() 
while(True):

    ret, frame = cap.read()

    crop = frame[60:120, 0:160]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    Gblur = cv2.GaussianBlur(gray,(5,5),0)

    ret,thresh = cv2.threshold(Gblur,60,255,cv2.THRESH_BINARY_INV)


    contours,hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)


    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)

        coorx = int(M['m10']/M['m00'])

        coory = int(M['m01']/M['m00'])

        cv2.line(crop,(coorx,0),(coorx,720),(255,0,0),1)

        cv2.line(crop,(0,coory),(1280,coory),(255,0,0),1)

        cv2.drawContours(crop, contours, -1, (0,255,0), 1)

        if coorx >= 120:
            PWM.setMotorModel(1000,1000,-500,-500)
            print ("Turn Left!")

 

        if coorx < 120 and coorx > 50:
            PWM.setMotorModel(500,500,500,500) 
            print ("On Track!")

 

        if coorx <= 50:
            PWM.setMotorModel(-500,-500,1000,1000)
            print ("Turn Right")

    else:
        print ("There is no line")


    cv2.imshow('Road',crop)

    if cv2.waitKey(1) == 27:
        PWM.setMotorModel(0,0,0,0)
        break
    #PWM.setMotorModel(0,0,0,0)
PWM.setMotorModel(0,0,0,0)
