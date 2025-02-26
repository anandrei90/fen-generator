#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:48:05 2025

@author: anandrei
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from tensorflow import keras
import time 


batch_size = 640
class_names_list = ['e', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

train_data = keras.utils.image_dataset_from_directory(
    r'/home/anandrei/deep_learning_alpha/projekt/data/pieces_train_no_duplicates/',  # path to images
    labels='inferred',              # labels are generated from the directory structure
    label_mode='categorical',       # 'categorical' => categorical cross-entropy 
    class_names=class_names_list,   # such that i can control the order of the class names
    color_mode='rgb',               # alternatives: 'grayscale', 'rgba'
    
    batch_size=batch_size,
    image_size=(50, 50),        
    shuffle=True,                   # shuffle images before each epoch
    seed=0,                         # shuffle seed
    validation_split=0.2,           # percentage of validation data
    subset='both',                  # return a tuple of datasets (train, val)
    interpolation='bilinear',       # interpolation method used when resizing images
    follow_links=False,             # follow folder structure?
    crop_to_aspect_ratio=False
    )

# 711716 images, 13 classes


#%% Test the shape of the batch

# train_data[0] is the train tf.dataset, train[1] the x-val tf.dataset
batch = train_data[0].take(1)

print('batch shape (inputs) = ', list(batch)[0][0].numpy().shape)
print('batch shape (label) = ', list(batch)[0][1].numpy().shape)
print('label type = ', list(batch)[0][1].numpy().dtype)

#%% Samples per class: 42.8% empty squares, rest of classes have 3.5 - 5.6 % each

all_labels = []
for x, y in train_data[0]:
  all_labels.append(np.argmax(y, axis = -1))
  
all_labels = np.concatenate(all_labels, axis = 0) 

#%% Plot histogram to visualize class imbalance

label_counts = [np.sum(all_labels == i) for i in range(len(class_names_list))]

plt.bar(class_names_list, label_counts)

for i in range(len(class_names_list)):
        plt.text(x = i, # x position of the text
                 y = label_counts[i] + 5000,  # y position of the text
                 s = f'{round(100*label_counts[i] / len(all_labels),1)} %', 
                 ha = 'center',
                 fontsize=8)
plt.title("Distribution of Samples in Classes")
#plt.yscale('log')
plt.show()

#%% Show a few chess piece images

plt.figure(figsize=(24, 18))
for images, labels in train_data[0]:
    
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(images[i].numpy().astype(int))
        label = np.argmax(labels.numpy()[i])
        plt.title(train_data[0].class_names[label], fontsize=18)
        plt.axis('off')
    plt.show()
    break

#%% Construct the model



model = keras.models.Sequential()
model.add(keras.layers.Input((50,50,3)))

model.add(keras.layers.Rescaling(scale = 1./255))


model.add(keras.layers.Conv2D(filters=18, 
                              kernel_size = (6,6),
                              strides = (1,1),
                              activation = 'relu',
                              kernel_regularizer = None)) #keras.regularizers.L2(1e-2) 


# 45*45*18 

model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

# 15*15*18

model.add(keras.layers.Conv2D(filters=36,
                              kernel_size = (4,4),
                              strides = (1,1),
                              activation = 'relu',
                              kernel_regularizer = None)) #keras.regularizers.L2(1e-2)
# 12*12*36

model.add(keras.layers.MaxPooling2D(pool_size=(3, 3))) 

# 4*4*36

model.add(keras.layers.Flatten())

# 576

model.add(keras.layers.Dense(60, activation="relu",
                             kernel_regularizer = None )) # keras.regularizers.L2(1e-2)
model.add(keras.layers.Dense(13, activation='softmax'))
model.summary()

#%% Compile the model



model.compile(loss="categorical_crossentropy", metrics=["accuracy", "f1_score"], 
              optimizer=keras.optimizers.SGD(learning_rate=0.005))

#%% Add class weights 

# loss[class] will be multiplied by weight[class] => try to make most weights of O(1)

class_weights = 3e4 / np.array(label_counts)
class_weights = class_weights / np.mean(class_weights[1:]) # make most weights O(1)

class_weights_dict = {i:weight for i,weight in enumerate(class_weights)}


#%% Fit the model

lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=3,
    min_delta=1e-4,
    cooldown=0,
    min_lr=1e-5
)

checkpoint_filepath = r'/home/anandrei/deep_learning_alpha/projekt/model_checkpoints/ckpt_model_no_dupl_run3.keras'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)




start = time.time()

# on average 1400 sec/epoch ~= 23m 20 sec/epoch

history = model.fit(train_data[0], epochs=30, verbose = True,
                    class_weight = class_weights_dict,
                    validation_data = train_data[1],
                    callbacks = [lr_reduction, model_checkpoint_callback])

print(f'Training time = {time.time() - start} sec') 
#%% Plot the learning curve

learn_curve = history.history

plt.figure(figsize=(15,7.5)) 

plt.subplot(1,2,1)
plt.plot(learn_curve['accuracy'], label ='Training')
plt.plot(learn_curve['val_accuracy'], label ='Validation')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(np.mean(learn_curve['f1_score'], axis = 1), label ='Training')
plt.plot(np.mean(learn_curve['val_f1_score'], axis = 1), label ='Validation')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')

plt.show()


#%%

file_path = r'/home/anandrei/deep_learning_alpha/projekt/model_checkpoints/chess_pieces.keras'

#model.save(file_path)             