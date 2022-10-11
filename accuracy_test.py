import csv
import random

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import save_img
from sklearn.metrics import classification_report

from cem_helper_func import *
from constants import *

# Dataset
# dataset = "cifar10"
dataset = "imagenet32"
# dataset = "imagenet32-pre"
# dataset = "mnist"

# Model
# model = "Resnet50v2"
# model = "MobileNetV2"
model = "ResNet20v1"
# model = "ResNet50v2-pre"

# Surrogate and Reference models
surrogate_no = 8
reference_no = 3  # Original 18

# Data preparation
(x_train, y_train), (x_test, y_test) = prepare_data(dataset=dataset, subtract_pixel_mean=True)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Model Preparation
source_model, reference_models, surrogate_models = prepare_models(x_train, y_train, x_test, y_test,
                                                                  dataset,
                                                                  model,
                                                                  input_shape=input_shape,
                                                                  use_saved=True,
                                                                  surrogate_no=surrogate_no,
                                                                  reference_no=reference_no)

data, label = x_test, y_test

# Score trained model.
scores = source_model.evaluate(data, label, verbose=1)
print('Source Model')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

for no, reference_model in enumerate(reference_models):

    # Score trained model.
    scores = reference_model.evaluate(data, label, verbose=1)
    print('Ref No:', no)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

for no, surrogate_model in enumerate(surrogate_models):

    # Score trained model.
    scores = surrogate_model.evaluate(data, label, verbose=1)
    print('Sur No:', no)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])



# scores = source_model.evaluate(x_test, y_test, verbose=1)
#
# new_data = np.r_[x_train, x_test]
# new_label = np.r_[y_train, y_test]
#
# predictions0 = source_model.predict(new_data)
#
# for prediction in predictions0:
#     for i in range(len(prediction)):
#         if i == np.argmax(prediction):
#             prediction[i] = 1
#         else:
#             prediction[i] = 0
#
# accuracy0 = classification_report(new_label, predictions0)
#
# predictions1 = source_model.predict(x_train)
#
# for prediction in predictions1:
#     for i in range(len(prediction)):
#         if i == np.argmax(prediction):
#             prediction[i] = 1
#         else:
#             prediction[i] = 0
#
# accuracy1 = classification_report(y_train, predictions1)
#
# predictions2 = source_model.predict(x_test)
#
# for prediction in predictions2:
#     for i in range(len(prediction)):
#         if i == np.argmax(prediction):
#             prediction[i] = 1
#         else:
#             prediction[i] = 0
#
# accuracy2 = classification_report(y_test, predictions2)

print("asd")
