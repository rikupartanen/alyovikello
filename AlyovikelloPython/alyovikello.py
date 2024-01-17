from gpiozero import Button
from signal import pause
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os, shutil
import pickle
import uuid
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app, storage
 
#Ladataan konvoluutioneuroverkko
model = tf.keras.models.load_model("CNNmodel1.model")
 
#Muutetaan kuvanottoaika stringiksi jotta helpompi käyttää databasessa
now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
 
 
#Kirjaudutaan firebaseen firebase-sdk.json tiedostolla joka on samassa kansiossa
cred = credentials.Certificate('firebase-sdk.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'alyovikello.appspot.com'})
db = firestore.client()
 
#Definataan napin painallus
def button_press():
    print("nappia painettu")
    face_recognition()
   
   
#Definataan napin painalluksen jälkeen tapahtuva kasvontunnistus
def face_recognition():
	#Asetetaan opencv:n tarjoama cascademalli jota käytetään sitten 
#kasvojen sijainnin paikantamiseen
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
	#Ladataan malli kansiosta
    recognizer.read("facemodel.yml")
 
    labels = {"henkilö":1}
    #luetaan labeleista henkilöiden nimet labels.pickle tiedostosta
    with open("labels.pickle",'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
	#käynnistetään kamera
    cap = cv2.VideoCapture(0)
	#aloitetaan timer algoritmin sammuttamista varten
    t0 = time.time()
    while cap.isOpened():
        ret,frame = cap.read()
        firebase_face_image= frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	#Toistetaan kasvojen pisteiden kautta
        for (x, y, w, h)in faces:
		#Kasvot laitetaan framen sisään jonka koko on y->y+h ja x->x+w
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
		#Alustetaan predictaaminen mallista
            id_, conf = recognizer.predict(roi_gray)
            #Asetetaan varmuusprosentti ohjelmalle )
            if conf>= 70 and conf <= 100 :
                print(conf)
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
			#Nimen tilalle asetetaan labels[id_] tiedostosta oikea nimi
                name = labels[id_]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame,name,(x-10,y-10),font,1,color,stroke,cv2.LINE_AA)
 
            #Asetetaan kasvojen koordinattien kohdalle neliö joka
            #värjätään
            color = (0,0,200)
            stroke = 3
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color, stroke)
 
        t1 = time.time()
	    #kasvojentunnistus toimii vain nappia painettaessa ja 10s ajan
        break_timer = t1 -t0
        if break_timer >= 10:
                    print("kuvan face upload")
			        #Kuva nimetään UUID-nimellä ja tallentaan kansioon
				    #ennen kuin se voidaan lähettää firebaseen
                    image_path = r'/home/pi/alyovikello/firebasekuvat'
                    #firebase_image = str(uuid.uuid4().hex)
                    firebase_face_image = os.path.join(image_path,'{}.jpg'.format(uuid.uuid4().hex))
                    cv2.imwrite(firebase_face_image,frame)
                    fileName = firebase_face_image
                    bucket = storage.bucket()
                    blob = bucket.blob(fileName)
                    blob.upload_from_filename(fileName)
                    blob.make_public()
 
                    #Tiedot kuvan otosta lähetetään databaseen
                    doc_ref = db.collection('restaurants').document()
                    doc_ref.set({
                    u'name': 'Nappia painettu ',
                    u'category': labels[id_],
                    u'photo': blob.public_url,
                    u'avgRating': u'0',
                    u'city': date_time
        })
		            #Kuva poistetaan kansiosta
                    folder = r'/home/pi/alyovikello/firebasekuvat'
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder,filename)
 
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Faile to delete %s. Reason : %s' % (file_path, e))
                    print("reloop")
                    break
 
 
 
    cap.release()
    cv2.destroyAllWindows()
 
 
#Definataan jatkuvasti käytössä oleva liikkeentunnistus
def motion_detect():
 
 
    time.sleep(0.1)
    motion_cap = cv2.VideoCapture(0)
    ret, motion_frame = motion_cap.read()
    #Kun kamera lähtee käyntiin, kameran framesta otetaan kuva ja se tallenetaan
    if ret:
        
        cv2.waitKey(0)
        #Kameran ottama frame tallenetaan
        cv2.imwrite('testi.jpg', motion_frame)
    #Annetaan sovellukselle aikaa ladata kuva ennen kuin siirrytään eteenpäin
    time.sleep(1)
 
   
    categories = [0,1] # 0.1 liikettä/henkilö kuvassa/ 1.0 ei liikettä/ei henkilöä kuvassa
    img_path = r'/home/pi/alyovikello/testi.jpg'
    #Kuva luetaan firebasen käyttöä varten
    firebase_image = cv2.imread(img_path)
 
	#kuva ladataan kansiosta tensorflown liikkeen tunnistusta varten
    test_image = image.load_img(img_path, target_size=(400, 400,1),color_mode="grayscale")
   
	#kuva muutetaan arrayksi
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
 
	#ennustetaan käyttämällä CNNmodel1.model neuroverkkoa
    result = model.predict(test_image, batch_size=1)
	#Asetetaan kategoriat 0 ja 1 tuloksille. Eli henkilö tai ei		#henkilöä
    result_categories = categories[np.argmax(result)]
    print(result)
    print(result_categories)
    if result_categories == 1:
	    #henkilö tunnistettiin ja aloitetaan kuvan siirtäminen firebaseen
	    #Asetetaan kuvakansion sijainti
        image_path = r'/home/pi/alyovikello/firebasekuvat'
        #firebase_imagelle asetetaan kansio ja sen nimi muutetaan uniikiksi nimeksi
        firebase_image = os.path.join(image_path,'{}.jpg'.format(uuid.uuid4().hex))
	    #firebase_image tallennetaan kansioon ennen kuin se voidaan
        #lähettää firebaseen
        #cv2.imwrite tallentaa kuvan kyseiseen kansioon
        cv2.imwrite(firebase_image,motion_frame)
        fileName = firebase_image
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)
        blob.make_public()
 
	    #Lähetetään kuvan tiedot firebasen databaseen
        doc_ref = db.collection('restaurants').document()
        doc_ref.set({
        u'name': 'Liike tunnistettu',
        u'category': u'',
        u'photo': blob.public_url,
        u'avgRating': u'0',
        u'city': date_time
        })
	    #tallennetut kuvat poistetaan niille tarkoitetusta kansiosta
        folder = r'/home/pi/alyovikello/firebasekuvat'
        for filename in os.listdir(folder):
                file_path = os.path.join(folder,filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason : %s' % (file_path, e))
    else:
	#algoritmi ei tunnistanut henkilöä
        print("ei henkilöä kuvassa")
 
   
#Koodin aloitus ja liikkeen tunnistuksen looppaus
start = time.time()
while motion_detect():
    print("kuva")
    continue
 
#Tunnistetaan napin painaminen
Nappi = Button(18)
 
try:
 
    Nappi.when_pressed = button_press
    pause()
 
finally:
    pass