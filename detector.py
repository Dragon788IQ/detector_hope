#Deteccion de espermatozoides en video
import cv2 as cv
import imutils
import numpy as np

cap = cv.VideoCapture("videos/video_2.mp4")
#fgbg = cv.bgsegm.createBackgroundSubtractorMOG2()
fgbg = cv.createBackgroundSubtractorMOG2()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
#result = cv.VideoWriter('filename.avi', cv.VideoWriter_fourcc(*'MJPG'),10, size)
result = cv.VideoWriter('output.avi', -1, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 1080)
    if ret == False: break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    fgmask = cv.dilate(fgmask, None, iterations=2)

    cnts = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv.contourArea(cnt) > 20:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    #cv.drawContours(-1, (0, 255, 0), 2)
    result.write(frame)
    cv.imshow("Frame", frame)
    cv.imshow("Fgmask", fgmask)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
result.release()
cv.destroyAllWindows()