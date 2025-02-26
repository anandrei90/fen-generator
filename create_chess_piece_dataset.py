#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:49:02 2025

@author: anandrei
"""

import numpy as np
from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, isdir
from skimage.util import view_as_blocks
import imagehash # https://github.com/JohannesBuchner/imagehash




def replace_digits_in_fen(fen):
    # takes a fen string and replaces each digit n with n consecutive e's
    # eg: 3 -> eee
    # outputs a 64 + 7 = 71 char string (each fen contains 7 dashes)
    for i in range(8):
        fen = fen.replace(str(i+1), 'e'*(i+1))
    
    return fen

# replace_digits_in_fen('1B1B2K1-1B6-5N2-6k1-8-8-8-4nq2')
      

    
def fen_to_one_dim_array(fen, separator = '-'):
    
    processed_fen = replace_digits_in_fen(fen)
    # get rid of separator such that an array of piece names can be created
    processed_fen = processed_fen.replace(separator, "")
    
    return np.array(list(processed_fen))


# fen_to_one_dim_array('1B1B2K1-1B6-5N2-6k1-8-8-8-4nq2')




def create_piece_dataset(
        load_path, 
        save_to_path,
        grayscale = False,
        hash_size = 16,
        remove_duplicates = True):
    # function that creates piece datasets (with an option to avoid duplicates)
    # the duplicatess will be detected using perceptual hash 
    
    # if last char in a path is a slash, remove it
    if load_path[-1] == r'/':
        load_path = load_path[:-1]
        
    if save_to_path[-1] == r'/':
        save_to_path = save_to_path[:-1]
        
    # remove '.jpeg' from file name to get the FENs
    all_fens = [f[:-5] for f in listdir(load_path) if 
                isfile(join(load_path, f))]
    
    # a set to store hashes; is used to detect (near) duplicates
    hashes = set()
    
    for i, fen in enumerate(all_fens):
        
        board_image = Image.open(join(load_path, fen + '.jpeg')) # open image
        fen_array = fen_to_one_dim_array(fen) # convert fen to 1d array
        
        # split the image in 64 squares
        if not(grayscale):
            data = np.asarray(board_image)
            squares = view_as_blocks(data,(50,50,3)).reshape(64, 50, 50, 3)
        else:
            data = np.asarray(board_image).convert('L')
            squares = view_as_blocks(data,(50,50,1)).reshape(64, 50, 50, 1)
        
        # save each square as a new image    
        for j in range(len(fen_array)):
            
            save_path = join(save_to_path, fen_array[j])
            
            # check if dir exists; if not, create it
            if not isdir(save_path):
                makedirs(save_path)
            
            piece_image = Image.fromarray(squares[j]) # convert array to image
            
            if remove_duplicates:
            
                #try:
                    #image_hash = str(imagehash.phash(piece_image, hash_size=hash_size))
                #except:
                        #image_hash = None
                
                # use phash (perceptual hash) to remove near duplicates
                image_hash = str(imagehash.phash(piece_image, hash_size=hash_size))
                
                # check if hash already exists
                if image_hash not in hashes:
                    hashes.add(image_hash)
                    piece_image.save(join(save_path, f'{i+1}_{j+1}.jpeg'))
                
            else:
                piece_image.save(join(save_path, f'{i+1}_{j+1}.jpeg'))
    
    print('Dataset created succesfully.')


#%% Create the no duplicate train piece dataset

'''#import time


load_from = r'/home/anandrei/deep_learning_alpha/projekt/data/train'
save_to = r'/home/anandrei/deep_learning_alpha/projekt/data/pieces_train_no_duplicates'


#start = time.time()

create_piece_dataset(
    load_path = load_from, 
    save_to_path = save_to)

#print(time.time() - start) # ~5600 sec for 80000 boards'''

#%% Create the test piece dataset

'''#import time

load_from = r'/home/anandrei/deep_learning_alpha/projekt/data/test'
save_to = r'/home/anandrei/deep_learning_alpha/projekt/data/pieces_test'



#start = time.time()

create_piece_dataset(
    load_path = load_from, 
    save_to_path = save_to,
    remove_duplicates = False)

#print(time.time() - start) # ~890 sec for 20000 boards'''