import cv2
import numpy as np
import sys
import os

def play_movie(path):
    from os import startfile
    startfile(path)
    
class Video(object):
    def __init__(self,path):
        self.path = path

    def play(self):
        from os import startfile
        startfile(self.path)

class Movie_MP4(Video):
    type = "MP4"
movie1 = Movie_MP4(r"F:\blink.mp4")
movie2 = Movie_MP4(r"F:\smile.mp4")
movie3 = Movie_MP4(r"F:\hello.mp4")
movie1.play()

facePath = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"
smilePath = "C:\opencv\sources\data\haarcascades\haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

sF = 1.05

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        #flags=cv2.CV_HAAR_SCALE_IMAGE
        flags=0
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            #flags=cv2.CV_HAAR_SCALE_IMAGE
            flags=0
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            print "Found", len(smile), "smiles!"
            movie2.play()
            movie3.play()
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
            #print "!!!!!!!!!!!!!!!!!"

    #cv2.cv.Flip(frame, None, 1)
    cv2.imshow('Smile Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
