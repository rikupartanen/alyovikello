import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

#Ajan muuttaminen stringiin jotta se on helppo näyttää verkkokäyttöliittymässä
now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

#'Kirjautuminen' omaan firebase-projektiin firebase-sdk.json tiedostolla
cred = credentials.Certificate('firebase-sdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

categories = [0,1] # 0.1 liikettä/henkilö kuvassa/ 1.0 ei liikettä/ei henkilöä kuvassa
img_path= r'C:\\Users\\samul\\Desktop\\Koulu\\Uusi kansio\\koneoppiminen\\Python\\testipython\\kuvakansio2\\IMG_20211118_155355.jpg'
model = tf.keras.models.load_model("CNNmodel1.model")

#Otetaan kuva
test_image = image.load_img(img_path, target_size=(400, 400,1),color_mode="grayscale")
plt.imshow(test_image)
plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

#Kuvan laitto koneoppimismodelin läpi
result = model.predict(test_image, batch_size=1)
result_categories = categories[np.argmax(result)]
print(result)
print(result_categories)

#Seuraava pätkä postaa firebasen databaseen seuraavat tiedot
if result_categories == 1:
    doc_ref = db.collection('restaurants').document()
    doc_ref.set({
    u'name': 'Liike tunnistettu',
    u'category': u'',
    u'photo': u'https://firebasestorage.googleapis.com/v0/b/alyovikello.appspot.com/o/oamk.png?alt=media&token=aa9a5c3d-8298-44fe-a6e2-b7399c0830d1',
    u'avgRating': u'0',
    u'city': date_time
    })

else:
    print("ei henkilöä kuvassa")