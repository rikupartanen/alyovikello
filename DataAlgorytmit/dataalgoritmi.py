import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

categories = [0,1] # 0.1 liikettä/henkilö kuvassa/ 1.0 ei liikettä/ei henkilöä kuvassa
img_path= r'C:\\Users\\samul\\Desktop\\Koulu\\Uusi kansio\\koneoppiminen\\Python\\testipython\\kuvakansio2\\IMG_20211118_155355.jpg'
model = tf.keras.models.load_model("CNNmodel1.model")

test_image = image.load_img(img_path, target_size=(400, 400,1),color_mode="grayscale")
#plt.imshow(test_image)
#plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


result = model.predict(test_image, batch_size=1)

result_categories = categories[np.argmax(result)]
print(result)
print(result_categories)
if result_categories == 1:
    print("henkilö kuvassa")
else:
    print("ei henkilöä kuvassa")