from keras import *
import matplotlib.pyplot as plt
from tensorflow.keras import *
import numpy as np
import cv2
from genDat import *
backend.backend()
backend.image_data_format()

# class model


class idmodel():
    def __init__(self):
        print("building model")
        self.model = Sequential()
        self.model.add(layers.Conv2D(
            128, 2, activation='relu', input_shape=(48, 78, 1)))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Conv2D(64, 2, activation='relu'))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Conv2D(32, 2, activation='relu'))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Conv2D(16, 2, activation='relu'))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        self.model.compile(optimizer='adam',
                           loss="binary_crossentropy",
                           metrics=['acc'],)

        print("model compiled")

    def train(self, x_train, y_train, nombre_epochs=10, val=0.1):
        x_train = x_train.reshape(x_train.shape[0], 48, 78, 1)
        print('training...')
        training = self.model.fit(x_train, y_train, epochs=nombre_epochs,
                                  validation_split=val)
        print("done")

    def pred(self, dat):
        pred = (self.model.predict(dat) > 0.5).astype("int32")
        print(pred)

    def save(self):
        self.model.save_weights('../weights/wgt.h5')
        print("weights saved")

    def load(self):
        self.model.load_weights('../weights/wgt.h5')
        print("weights loaded")


# code test
x_train = genDat()
y_train = np.array(x_train[1])
x_train = np.array(x_train[0])

cnn = idmodel()
#cnn.train(x_train, y_train)
cnn.load()

mn = genUsable("manu/mesh0.dat")
cnn.pred(mn)
# cnn.save()
