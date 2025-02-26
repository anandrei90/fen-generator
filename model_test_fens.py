#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:48:41 2025

@author: anandrei
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow import keras 
import time 
from skimage.util import view_as_blocks
from os import listdir
from os.path import isfile, join
from PIL import Image

model_filepath = r'/home/anandrei/deep_learning_alpha/projekt/model_checkpoints/chess_pieces.keras'
model = keras.models.load_model(model_filepath)

model.summary()


#%% Function definitions


def get_fen_labels_from_dir(path):
    # get rid of the .jpeg ending
    fen_labels = [f[:-5] for f in listdir(path) if isfile(join(path, f))]
    # sort the list for usage in image_dataset_from_directory
    return sorted(fen_labels)


def one_dim_array_to_fen(array, separator = '-'): 
    # array is a vector with 64 1-char long strings
    # organized the string along the 8 rows of the chess boards
    array = [''.join(array[i:i+8]) for i in range(0, 64, 8)]
    # reconstruct fen row by row
    fen = separator.join(array)
    # replace n * 'e' with n
    for i in range(8,0,-1): # go backwards, otherwise 'eeee' -> 1111 and not 4
        fen = fen.replace('e'*i, str(i))
    
    return fen


def fens_from_chessboards(test_data):
    # test_data needs to be a generator (tf.Dataset object)
    
    # initialize accumulators
    correct_predictions = 0 # needed to calculate accuracy
    dataset_length = 0 # needed to calculate accuracy
    predicted_fens = np.array([]) # here i store all the predicted fens
    
    
    # iterate over the generator
    for batch, labels in test_data: # labels come out as b(yte)-strings!!
            
        images = batch.numpy() # (100, 400, 400, 3)
        labels = labels.numpy()# (100,) => array of fens (byte-strings)
        # decode the strings to utf-8 (aka 'normal' strings)
        labels = np.array([label.decode('utf-8') for label in labels])
        # split the boards in squares
        squares = view_as_blocks(images,(1,50,50,3)).reshape(64*images.shape[0], 50, 50, 3)
        # predict the pieces using the CNN
        predictions = model.predict(squares) # 1H-encoded, (6400,13)
        predictions = np.argmax(predictions, axis=1) # cardinal encoded, (6400,)
        # original class names, needed for reconstructing the fens
        predictions_piece = [class_names_list[i] for i in predictions] # (6400,)
        
        predictions_piece = np.array(predictions_piece).reshape(-1,64) # (batch.shape[0], 64)
        
        prediction_fen = np.array([one_dim_array_to_fen(row) for row in predictions_piece])
        
        predicted_fens = np.concatenate((predicted_fens, prediction_fen))
        correct_predictions += np.sum(prediction_fen == labels)
        dataset_length += labels.size
    
    fen_accuracy = correct_predictions / dataset_length
    
    return fen_accuracy, predicted_fens

#%% Load the chess board image dataset


class_names_list = ['e', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

test_chess_boards_path = r'/home/anandrei/deep_learning_alpha/projekt/data/test/'


fen_test_data = keras.utils.image_dataset_from_directory(
    test_chess_boards_path,  # path to images
    labels = get_fen_labels_from_dir(test_chess_boards_path),              
    label_mode='int',
    color_mode='rgb',               # alternatives: 'grayscale', 'rgba'
    batch_size=100,
    image_size=(400, 400),        
    shuffle=False,
    validation_split= None,         # percentage of validation data
    interpolation='bilinear',       # interpolation method used when resizing images
    follow_links=False,             # follow folder structure?
    crop_to_aspect_ratio=False
    )

#%% Compare predicted fens to true fens

start = time.time()
accuracy, fens_pred = fens_from_chessboards(fen_test_data)

print(f'Duration = {time.time() - start} sec') # 16-17 min

print(f'Accuracy = {accuracy}') # 0.9996 (8 wrong fens out of 20000)
print(f'First 3 fens = {fens_pred[:3]}')

#%% Plot the boards with the wrongly predicted fens


fens_true = get_fen_labels_from_dir(test_chess_boards_path)

confused_indexes = np.where(fens_true != fens_pred)[0] # 8 wrongly predicted fens

filepaths = [test_chess_boards_path + fens_true[i] + '.jpeg' for i in confused_indexes]

for i, index in enumerate(confused_indexes):
    img = Image.open(filepaths[i])
    plt.imshow(img)
    plt.title(fens_pred[index])
    plt.axis("off")
    plt.show()


