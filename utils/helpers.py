import os
from os.path import isfile, join

import numpy as np
import cv2 as cv
from skimage.util import view_as_blocks

from utils.constants import CLASS_NAMES


def get_fen_labels_from_dir(path):
    '''
    Retrieves the FEN strings out of the names of
    the image files from a given directory.
    '''
    # get rid of the .jpeg ending
    fen_labels = [f[:-5] for f in os.listdir(path) if isfile(join(path, f))]
    # sort the list for usage in image_dataset_from_directory
    return sorted(fen_labels)


def one_dim_array_to_fen(array, separator='-'):
    '''
    Converts an FEN array of length 64 to a FEN string.
    '''
    # array is a vector with 64 1-char long strings
    # organize the string along the 8 rows of the chess boards
    array = [''.join(array[i:i+8]) for i in range(0, 64, 8)]
    # reconstruct fen row by row
    fen = separator.join(array)
    # replace n * 'e' with n
    for j in range(8, 0, -1):  # go backwards, otherwise 'eeee' -> 1111
        fen = fen.replace('e'*j, str(j))
    return fen


def fens_from_chessboards(model, test_data):
    '''
    Predicts the FENS corresponding to each chessboard image from a dataset.
    Returns the accuracy and the predicted FENS.
    '''
    # test_data needs to be a generator (tf.Dataset object)

    # initialize accumulators
    correct_predictions = 0  # needed to calculate accuracy
    dataset_length = 0  # needed to calculate accuracy
    predicted_fens = np.array([])  # here i store all the predicted fens

    # iterate over the generator
    for batch, labels in test_data:  # labels come out as b(yte)-strings!!

        images = batch.numpy()  # (100, 400, 400, 3)
        labels = labels.numpy()  # (100,) => array of fens (byte-strings)
        # decode the strings to utf-8 (aka 'normal' strings)
        labels = np.array([label.decode('utf-8') for label in labels])
        # split the boards in squares
        squares = view_as_blocks(images, (1, 50, 50, 3))
        squares = squares.reshape(64 * images.shape[0], 50, 50, 3)
        # predict the pieces using the CNN
        predictions = model.predict(
            squares,
            verbose=0
            )  # 1H-encoded, (6400,13)
        predictions = np.argmax(predictions, axis=1)  # cardinal encoded
        # original class names, needed for reconstructing the fens
        predictions_piece = [CLASS_NAMES[i] for i in predictions]  # (6400,)

        predictions_piece = np.array(predictions_piece).reshape(-1, 64)

        prediction_fen = np.array([one_dim_array_to_fen(row)
                                   for row in predictions_piece])

        predicted_fens = np.concatenate((predicted_fens, prediction_fen))
        correct_predictions += np.sum(prediction_fen == labels)
        dataset_length += labels.size

    fen_accuracy = correct_predictions / dataset_length

    return fen_accuracy, predicted_fens


def find_chessboard_corners(bw_img):
    # TODO: make sure bw_img is black & white
    # i.e. pixel values either 0 or 255

    # detect the 6x6 internal chessboard and
    # find the coordinates of its corners
    found, points = cv.findChessboardCornersSB(
        image=bw_img,
        patternSize=(7, 7),
        flags=cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE
    )  # points[-1, 0, :] is bottom right, points[0, 0, :] is top left

    if not found:
        return None

    right, bottom = points[-1, 0, :]
    left, top = points[0, 0, :]
    # Side of the square
    square = 0.5 * (right-left + bottom-top)/6

    right = int(right+square)
    left = int(left-square)
    top = int(top-square)
    bottom = int(bottom+square)

    return left, right, top, bottom


def crop_chessboard(img):

    # convert to numpy array
    img = np.array(img)

    # convert image to grayscale
    gray = cv.cvtColor(src=img, code=cv.COLOR_RGB2GRAY)

    # hardcode thresholds to be tried
    thresholds = [127, 159, 191, 95, 223, 31]

    # threshold the image
    for threshold in thresholds:

        _, output = cv.threshold(
            src=gray,  # source image
            thresh=threshold,  # threshold value
            maxval=255,  # maximum value to use with binary thresholding types
            type=cv.THRESH_BINARY  # thresholding type
        )

        # return cropped image if chessboard successfully detected
        points = find_chessboard_corners(output)
        if points:
            left, right, top, bottom = points
            return img[top:bottom, left:right]

    return None
