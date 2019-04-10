"""
This guide trains a neural network model to classify images of clothing, like sneakers and shirts.
It's okay if you don't understand all the details,
this is a fast-paced overview of a complete TensorFlow program with the details explained as we go.

This guide uses tf.keras, a high-level API to build and train models in TensorFlow.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images.shape
# len(train_labels)
# train_labels
# test_images.shape
# len(test_labels)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, we divide the values by 255.
# It's important that the training set and the testing set are preprocessed in the same way:
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# A sequential model:
# model = keras.Sequential([

# transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
#     keras.layers.Flatten(input_shape=(28, 28)),

# These are densely-connected, or fully-connected, neural layers.
# The first Dense layer has 128 nodes (or neurons).
#     keras.layers.Dense(128, activation=tf.nn.relu),

# The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])


"""
Loss should be < 10
Loss function
  —This measures how accurate the model is during training.
  We want to minimize this function to "steer" the model in the right direction.
Optimizer
  —This is how the model is updated based on the data it sees and its loss function.
Metrics
  —Used to monitor the training and testing steps.
  The following example uses accuracy, the fraction of the images that are correctly classified.
  
  For more info:https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
  For loss :https://zhuanlan.zhihu.com/p/34667893
  1.mean_squared_error
  2.mean_absolute_error
  3.mean_absolute_percentage_error
  4.mean_squared_logarithmic_error
  5.squared_hinge
  6.hinge
  7.categorical_hinge
  8.logcosh
  9.categorical_crossentropy
  10.sparse_categorical_crossentropy
  11.binary_crossentropy
  12.kullback_leibler_divergence
  13.poisson
  14.cosine_proximity
  15.
  
"""


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


"""
Train the model
Training the neural network model requires the following steps:

Feed the training data to the model—in this example, the train_images and train_labels arrays.
The model learns to associate images and labels.
We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.

To start training, call the model.fit method—the model is "fit" to the training data:
"""
# model.fit(train_images, train_labels, epochs=5)


"""
Overfitting is when a machine learning model performs worse on new data than on their training data.

"""
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)



# model.save('my_model.h5')



new_model = keras.models.load_model('my_model.h5')
new_model.summary()
predictions = new_model.predict(test_images)

# predictions = model.predict(test_images)

# predictions[0]
#
# np.argmax(predictions[0])
#
# test_labels[0]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = new_model.predict(img)
print(predictions_single)

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

