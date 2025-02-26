#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:32:55 2025

@author: anandrei
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow import keras 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% Import model

model_filepath = r'/home/anandrei/deep_learning_alpha/projekt/model_checkpoints/chess_pieces.keras'
model = keras.models.load_model(model_filepath)

model.summary()

#%% Plot model


keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB", # left-right aka horizontal plot; 'TB' => vertical plot
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=False
)

#%% Load the test data

batch_size = 640
class_names_list = ['e', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

test_data = keras.utils.image_dataset_from_directory(
    r'/home/anandrei/deep_learning_alpha/projekt/data/pieces_test/',  # path to images
    labels='inferred',              # labels are generated from the directory structure
    label_mode='categorical',       # 'categorical' => categorical cross-entropy 
    class_names=class_names_list,   # such that i can control the order of the class names
    color_mode='rgb',               # alternatives: 'grayscale', 'rgba'
    batch_size=batch_size,
    image_size=(50, 50),        
    shuffle=False,                  # shuffle images before each epoch
    seed=0,                         # shuffle seed
    validation_split= None,         # percentage of validation data
    #subset='both',                 # return a tuple of datasets (train, val)
    interpolation='bilinear',       # interpolation method used when resizing images
    follow_links=False,             # follow folder structure?
    crop_to_aspect_ratio=False
    )

# 1280000 images

#%% Accuracy and f1 scores

'''# returns [loss, accuracy, array of f1 scores]
_, accuracy, f1_scores = model.evaluate(test_data) 


print(f'Model accuracy = {accuracy}')
print(f'F1 scores = {f1_scores}')'''


#%% Generate Predictions (~860 sec)

test_predictions = model.predict(test_data) # 1H-encoded

#%% Calculate accuracy

y_pred = np.argmax(test_predictions, axis=1) # cardinal encoded
y_true = np.argmax(np.concatenate([y for x, y in test_data], axis=0), axis=1)

print(f'Accuracy = {1 - np.sum(y_true != y_pred) / y_pred.size}') # 0.9999953
print(f'Mislabeled pieces in the test set: {np.sum(y_true != y_pred)} out of 1280000') # 6


#%% Calculate and plot the confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_mat = confusion_matrix(
    y_true, 
    y_pred, 
    #normalize = 'pred' # normalize over predicted label (columns) 
    )

# the numbers on the diagonals are O(10^3) or higher => set them to 0
np.fill_diagonal(conf_mat, 0) 

disp = ConfusionMatrixDisplay(conf_mat, display_labels = class_names_list)

disp.plot()
plt.title("Test Data Confusion Matrix")
plt.show()






