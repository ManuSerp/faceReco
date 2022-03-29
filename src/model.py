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


# il faut redemarrer le kernel pour réinitialiser le modèle
nombre_epochs = 20
scores_train = []
scores_test = []
for i in range(1, nombre_epochs+1):
    training = model.fit(train_generator, epochs=1,
                         validation_data=test_generator)
    scores_train.append(training.history['acc'])
    scores_test.append(training.history['val_acc'])

plt.ylim([0.4, 1])
plt.plot(range(1, nombre_epochs+1), scores_train,
         'red', range(1, nombre_epochs+1), scores_test)
plt.show()


backend.backend()
backend.image_data_format()
liste = [0.8, 0.5, 0.2, 0.1]
score_test = []
score_train = []
batch_size = 10
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(224, 224),
    batch_size=batch_size,)


for i in liste:
    model = Sequential([

        #layers.Conv2D(128,4, activation='relu'),
        # layers.MaxPooling2D(),
        #layers.Conv2D(64,4, activation='relu'),
        # layers.MaxPooling2D(),
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
    train_datagen = preprocessing.image.ImageDataGenerator(
        validation_split=i, shear_range=10, rescale=1./255, rotation_range=10, horizontal_flip=True, vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=batch_size)
    training = model.fit(train_generator, epochs=10,
                         validation_data=test_generator)
    score_train.append(model.evaluate(train_generator, verbose=0)[1])
    print(score_test.append(model.evaluate(test_generator, verbose=0)[1]))

    print(i)

train_acc = training.history['acc']
xc = range(4)

plt.figure()
plt.ylim([0.3, 1])
plt.plot(xc, score_test)
plt.show()
