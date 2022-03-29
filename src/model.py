from keras import *
import matplotlib.pyplot as plt
from tensorflow.keras import *
import numpy as np
import cv2
backend.backend()
backend.image_data_format()
# Le modèle initial en comprenant les # est meilleur mais trop lent
model = Sequential([

    layers.Conv2D(128, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['acc'],)


batch_size = 20
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=batch_size,)

train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255, shear_range=10, rotation_range=10, horizontal_flip=True, vertical_flip=True)
# ajouter de l'aléatoire shear_range=30,rescale = 1./255,rotation_range=30, horizontal_flip=True, vertical_flip = True
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=batch_size,)


nombre_epochs = 20

training = model.fit(
    train_generator, validation_data=test_generator, epochs=nombre_epochs)


train_acc = training.history['acc']

xc = range(nombre_epochs)

plt.figure()
plt.plot(xc, train_acc, xc, training.history['val_acc'], 'red')
plt.show()


img = preprocessing.image.load_img('voiture.jpg')

img = np.array(img) / 255.0

img = cv2.resize(img, dsize=(224, 224))
img = np.array(img).reshape(1, 224, 224, 3)
predictions = (model.predict(img) > 0.5).astype("int32")
print(predictions)

img = preprocessing.image.load_img('avion.jfif')

img = np.array(img) / 255.0

img = cv2.resize(img, dsize=(224, 224))
img = np.array(img).reshape(1, 224, 224, 3)
predictions = (model.predict(img) > 0.5).astype("int32")
print(predictions)

scores = model.evaluate(test_generator, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
