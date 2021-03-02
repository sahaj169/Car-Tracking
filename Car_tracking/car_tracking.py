import cv2
import numpy as np


# create our body classifier 

car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# initiate video capture for video files 

cap = cv2.VideoCapture('training_video.mp4')

#loop once video is successfully loaded 

while cap.isOpened():
    # read first frame 
    ret , frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # pass frame to our car classifier 
    cars = car_classifier.detectMultiScale(gray,1.4,2)

    # extract bounding boxes for any bodies identified
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.imshow('Cars',frame)

    if cv2.waitKey(1)== 13: # 12 is the enter key
        break

cap.release()
cv2.destroyAllWindows()

