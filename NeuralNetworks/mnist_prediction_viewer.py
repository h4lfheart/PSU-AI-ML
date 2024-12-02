import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

model = keras.models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer = 'adam',
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.sparse_categorical_accuracy()])
model.fit(x_train, y_train, epochs = 5)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Finished training w/ a loss of {loss} and accuracy of {accuracy}%")

probabilities = model.predict(x_test)
predictions = np.argmax(probabilities, axis=1)

display_image = plt.imshow(x_test[0], cmap='gray')
title = plt.title(f"Prediction: None")
current_predict_index = 0


def load(predict_index):
    global title
    title.set_text(f"Prediction: {predictions[predict_index]}")

    global display_image
    display_image.set_data(x_test[predict_index])

    plt.draw()

load(0)

def next_prediction(event):
    global current_predict_index

    if current_predict_index == len(predictions) - 1:
        return

    current_predict_index += 1
    load(current_predict_index)

def prev_prediction(event):
    global current_predict_index

    if current_predict_index == 0:
        return

    current_predict_index -= 1
    load(current_predict_index)

next_button = plt.Button(plt.axes((0.01, 0.85, 0.2, 0.1)), 'Next Prediction')
next_button.on_clicked(next_prediction)

prev_button = plt.Button(plt.axes((0.01, 0.7, 0.2, 0.1)), 'Prev Prediction')
prev_button.on_clicked(prev_prediction)
plt.show()