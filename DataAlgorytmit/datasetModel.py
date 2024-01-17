import tensorflow
import numpy as np
import tensorflow as tf
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt


directory = 'C:\\Users\\Samuli\\Desktop\\Koulu\\Dokumentit\\Koneoppiminen\\Python\\dataalgoritmi\\kuvakansio'
nPictures = 81                # Total number of pictures
split = 0.2                   # 20% test and 80 % training data
nTraining = (int)((1-split)*nPictures) 
nValidation = (int)(split*nPictures)

training_data = tf.keras.preprocessing.image_dataset_from_directory(
directory,                # juurihakemisto, jonka alta löytyy kunkin dataluokan omat hakemistot
batch_size=nTraining,     # data jaetaan tämän kokoisin batcheihin.
color_mode="grayscale",
shuffle=True,
seed=1,
validation_split = split, # tämän arvo = 0.2
subset = 'training',      # tämä kertoo, että tällä kertaa datasta otetaan 80% eli yksi batch
image_size=(256, 256)
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
directory,                # juurihakemisto, jonka alta löytyy kunkin dataluokan omat hakemistot
batch_size=nValidation,     # data jaetaan tämän kokoisin batcheihin.
color_mode="grayscale",
shuffle=True,
seed=2,
validation_split = split, # tämän arvo = 0.2
subset = 'validation',      # tämä kertoo, että tällä kertaa datasta otetaan 80% eli yksi batch
image_size=(256, 256)
)



# determine the shape of the input images
in_shape = [256,256,1]
# determine the number of classes





########################################
# 5 step process step 1: Define model
########################################
model = Sequential()
#model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(10, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='relu', kernel_initializer='he_uniform'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#######################################
# 5 step process step 2: Compile model (define loss and optimizer)
#######################################
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#######################################
# 5 step process step 3: Fit the model = train model
#######################################

model.fit(training_data, epochs=40, batch_size=5, verbose=2)

#######################################
# 5 step process step 4: Evaluate the model
#######################################
loss, acc = model.evaluate(validation_data, verbose=2)
print('Accuracy: %.3f' % acc)
#######################################
# 5 step process: Make a prediction
#######################################
'''
image = x_test[3]
plt.figure(2)
plt.imshow(image)
yhat = model.predict(asarray([image]))
print('Predicted: class=%d' % argmax(yhat))
plt.figure(1)
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    label = y_test[i]
    plt.title(label)
plt.show()
'''
print(model.summary())