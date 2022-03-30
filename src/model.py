from keras import *
import matplotlib.pyplot as plt
from tensorflow.keras import *
import numpy as np
import cv2
from genDat import *
backend.backend()
backend.image_data_format()

print("lib imported")
# Le modèle initial en comprenant les # est meilleur mais trop lent
model = Sequential()
# layers.Flatten()
model.add(layers.Conv2D(128, 2, activation='relu', input_shape=(48, 78, 1)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, 2, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 2, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(16, 2, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))


print(model.summary())

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['acc'],)

print("model compiled")


nombre_epochs = 30
x_train = genDat()
y_train = np.array(x_train[1])
x_train = np.array(x_train[0])

x_train = x_train.reshape(x_train.shape[0], 48, 78, 1)

print('training...')
training = model.fit(x_train, y_train, epochs=nombre_epochs,
                     validation_split=0.1)


train_acc = training.history['acc']

xc = range(nombre_epochs)

plt.figure()
plt.plot(xc, train_acc, xc, training.history['val_acc'], 'red')
plt.show()


# img = preprocessing.image.load_img('voiture.jpg')

# img = np.array(img) / 255.0

# img = cv2.resize(img, dsize=(224, 224))
# img = np.array(img).reshape(1, 224, 224, 3)
# predictions = (model.predict(img) > 0.5).astype("int32")
# print(predictions)

# img = preprocessing.image.load_img('avion.jfif')

# img = np.array(img) / 255.0

# img = cv2.resize(img, dsize=(224, 224))
# img = np.array(img).reshape(1, 224, 224, 3)
# predictions = (model.predict(img) > 0.5).astype("int32")
# print(predictions)

# scores = model.evaluate(test_generator, verbose=0)

# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
