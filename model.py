from keras.layers import Input, Dense, Activation, Conv2D, AveragePooling2D, Flatten
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Model
from keras.datasets import mnist
import pickle
import tensorflow as tf
import numpy as np
import cv2


def load_data():
    file = open("drive/MyDrive/labeld_data", "rb")
    (samples, labels) = pickle.load(file)
    return (samples, labels)


(samples, labels) = load_data()
samples = np.array(samples)
# samples = to_categorical(samples)
samples, test, labels, test_labels = train_test_split(
    samples, labels, test_size=0.1, shuffle=True, random_state=5)
labels = to_categorical(labels)


def build_model(input_shape):

    x_input = Input(shape=input_shape, name='input')

    x = Conv2D(filters=16, kernel_size=(2, 2), strides=1,
               padding='valid', name='conv2')(x_input)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad2')(x)

    x = Flatten()(x)

    x = Dense(units=120, name='fc_1')(x)

    x = Activation('relu', name='relu_1')(x)
    # x = Dropout(rate = 0.5)

    x = Dense(units=84, name='fc_2')(x)
    x = Activation('relu', name='relu_2')(x)
    # x = Dropout(rate = 0.5)
    x = Dense(units=48, name='fc_3')(x)
    x = Activation('relu', name='relu_3')(x)

    outputs = Dense(units=2, name='softmax', activation='softmax')(x)

    model = Model(inputs=x_input, outputs=outputs)
    model.summary()

    return model


model = build_model(input_shape=(45, 30, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=samples, y=labels, epochs=10)
model.save('model')
sample = test[150]
plt.figure(figsize=(3, 3))
plt.imshow(sample)
plt.show()
batch = np.expand_dims(sample, axis=0)
prediction = model.predict(batch)
print(np.argmax(prediction[0]))
