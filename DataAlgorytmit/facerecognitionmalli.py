import os
import numpy as np
import cv2
from PIL import Image
import pickle

dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(dir,"kuvakansiofaceid")
#Opencv kansiosta data/xml-profiilit ja sieltä frontalface/profile
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
#pip install opencv-contrib-python for cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root,file)
            #path->rootin sijaan antaa verrattavan kuvan numeron
            label = os.path.basename(root).replace(" ","-").lower()
            print(label,path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id +=1

            id_ = label_ids[label]
            print(label_ids)

            #convert("L") = grayscale
            pil_image = Image.open(path).convert("L")
            #kansioista otetut kuvat asetetaan 250x250 pikseleihin
            size = (250,250)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8")
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for ( x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

'''print(y_labels)
print(x_train'''

#pickle hakee labelit kansioista
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

#malli koulutetaan ja tallenetaan
recognizer.train(x_train,np.array(y_labels))
recognizer.save("facemodel.yml")