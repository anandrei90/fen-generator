import os
from os.path import isfile, join

import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import Model
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


def preprocess_image(image):
    """
    Takes a cropped chessboard image and prepares it for ingestion
    in the ML model.
    """

    # convert to PIL.Image if image is an numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image = image.resize((400, 400))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # (400, 400, 3) -> (1, 400, 400, 3)

    return image


def classify_squares(image, model):
    """
    Takes a 400x400 pixel chessboard image, breaks it into its 64 squares,
    and the feeds each square to an ML model which identifies the piece
    (or absence thereof) in each square. The squares are analyzed from left
    to right starting from the uppermost rank of the chessboard. Returns a
    list containing the identified class for each square, ordered as described
    above.
    """

    if not isinstance(model, Model):
        raise TypeError("The model is not a keras model.")

    # initialize piece list
    pieces = []

    # break image into squares and classify each square
    for row in range(8):
        for col in range(8):
            # extract square
            # square.shape = (1, 50, 50, 3)
            square = image[:, row*50:(row+1)*50, col*50:(col+1)*50, :]

            # classify square using ML model
            # prediction.shape = (1, 13)
            prediction = model.predict(square, verbose=0)
            # get index of max prediction
            piece_index = np.argmax(prediction, axis=1)[0]
            piece = CLASS_NAMES[piece_index]
            # append piece to fen string
            pieces.append(piece)

    return pieces


def save_fig_in_buffer(figure, caption, buffer):

    # create the figure with matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(figure)
    # add fen string as title to the figure
    plt.title(f"FEN: {caption}", fontsize=16)
    plt.axis('off')
    # save the plot in the buffer as a png
    plt.savefig(buffer, format="png")
    plt.close()


def find_chessboard_corners(bw_img):
    """
    Takes a black & white (thresholded) image and detects if and where
    the chessboard is in the image. Returns the four integers (left,
    right, top, bottom) corresponding to the boundaries of the chessboard.
    Cannot detect more than one chessboard in a given image.

    Returns None if no chessboard is detected.
    """

    # detect the 6x6 internal chessboard and
    # find the coordinates of its corners
    found, points = cv.findChessboardCornersSB(
        image=bw_img,
        patternSize=(7, 7),
        flags=cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE
    )  # points[-1, 0, :] is bottom right, points[0, 0, :] is top left

    if not found:
        return None

    # extract coordinates of 6x6 chessboard corners
    right, bottom = points[-1, 0, :]
    left, top = points[0, 0, :]
    # find side of chessboard square
    square = 0.5 * (right-left + bottom-top)/6

    # infer coordinates of 8x8 chessboard corner
    # in openCV the origin of the x,y coordinates is top left
    # i.e. x-axis points rightwards, y-axis downwards
    right = int(right+square)
    left = int(left-square)
    top = int(top-square)
    bottom = int(bottom+square)

    return left, right, top, bottom


def crop_chessboard(img):
    """
    Takes an image/screenshot containing a chessboard and returns
    the cropped out chessboard (if found) from the image.
    Assumes there is only one chessboard in the image.

    Returns None if no chessboard is detected.
    """

    # convert to numpy array to make it openCV-ready
    img = np.array(img)

    # convert image to grayscale
    gray = cv.cvtColor(src=img, code=cv.COLOR_RGB2GRAY)

    # hardcode thresholds to be tried
    thresholds = [127, 159, 191, 95, 223, 31]

    # use different thresholds until chessboard is detected
    for threshold in thresholds:

        _, output = cv.threshold(
            src=gray,  # source image
            thresh=threshold,  # threshold value
            maxval=255,  # maximum value to use with binary thresholding types
            type=cv.THRESH_BINARY  # thresholding type
        )

        points = find_chessboard_corners(output)
        # return cropped image if chessboard successfully detected
        if points:
            left, right, top, bottom = points
            return img[top:bottom, left:right]  # openCV conventions

    return None
