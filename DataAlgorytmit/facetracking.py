import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import uuid
import pickle

#OpenCV cv2.data folderi
#face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("facemodel.yml")

labels = {"henkilö":1}
#luetaan labeleista henkilöiden nimet labels.pickle tiedostosta
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h)in faces:
        #x-y square
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        #roi_gray = gray[y-50:y+200, x-50:x+200]
        roi_color = frame[y:y+h,x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        #conf 80-85, saa usein henkilön oikein mutta sekoittaa vielä välillä henkilöt keskenään(enemmän koulutusdataa?)
        if conf>= 60 and conf <= 125 :
            print(conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x-10,y-10),font,1,color,stroke,cv2.LINE_AA)

        #kuvan asettelun katselmointia varten
        #img_item = "kasvokuva.png"
        #bug? 3+ hlö kuvassa aiheuttaa hidastuksia ha suuri väkijoukko aiheuttaa mahdollisen crashin
        #cv2.imwrite(img_item,roi_gray)

        #BGR
        color = (0,0,200)
        stroke = 3
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color, stroke)

    #Framen adjustaaminen 250x250 kokoon
    '''frame = frame[85:85+250,185:185+250,:]'''

    #kansio jonne otetut kuvat tallenetaan
    '''POSPATH = os.path.join('kuvakansiokasvot','positive')'''

    #koulutuskuvien ottaminen videosta/feedistä
    '''if cv2.waitKey(1) & 0XFF == ord("a"):
        imgname = os.path.join(POSPATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname,frame)'''

    #Break video/exit
    cv2.imshow('Face Tracking', frame)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()