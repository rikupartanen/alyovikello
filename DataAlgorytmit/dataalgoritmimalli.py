import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

directory = 'C:\\Users\\Samuli\\Desktop\\Koulu\\Dokumentit\\Koneoppiminen\\Python\\dataalgoritmi\\kuvakansio'
nPictures = 81
split = 0.2
nTraining = (int)((1-split)*nPictures) 
nValidation = (int)(split*nPictures)

training_data = tf.keras.preprocessing.image_dataset_from_directory(
directory,
batch_size=nTraining,
color_mode="grayscale",
shuffle=True,
seed=1,
validation_split = split,
subset = 'training',
image_size=(400, 400)
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
directory,
batch_size=nValidation,
color_mode="grayscale",
shuffle=True,
seed=2,
validation_split = split,
subset = 'validation',
image_size=(400, 400)
)


in_shape = [400,400,1]
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape =in_shape))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_data, epochs=40, batch_size=5, verbose=2)


loss, acc = model.evaluate(validation_data, verbose=2)
print('Accuracy: %.3f' % acc)
model.summary()
model.save('CNNmodel2.model')