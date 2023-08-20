#!/usr/bin/env python3P
import cv2                 #import libraries

#loading the cascades
face_cascade = cv2.CascadeClassifier('/Users/sadhanasharma/Downloads/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/sadhanasharma/Downloads/Module_1_Face_Recognition/haarcascade_eye.xml')

#defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5 )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #roi = region of inter
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3 )
        for (ex, ey, ew, eh) in faces:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

#face recognition with webcam
video_capture = cv2.VideoCapture(1)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()









