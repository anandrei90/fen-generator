from os import listdir, makedirs
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
from skimage.util import view_as_blocks
import imagehash  # https://github.com/JohannesBuchner/imagehash


def replace_digits_in_fen(fen):
    """
    Takes a FEN string and replaces each digit n with n consecutive e's.
    Eg: 3 -> eee
    Outputs a 64 + 7 = 71 char string (each fen contains 7 dashes).
    """
    for i in range(8):
        fen = fen.replace(str(i+1), 'e'*(i+1))
    return fen


def fen_to_one_dim_array(fen, separator='-'):
    '''
    Transforms a FEN string into a list.
    Each piece and empty square is an element.
    '''
    processed_fen = replace_digits_in_fen(fen)
    # get rid of separator such that an array of piece names can be created
    processed_fen = processed_fen.replace(separator, "")
    return list(processed_fen)


def create_piece_dataset(
        load_path,
        save_to_path,
        grayscale=False,
        hash_size=16,
        remove_duplicates=True
        ):
    '''
    Function that creates piece datasets (with an option to avoid duplicates).
    The duplicates will be detected using perceptual hash.
    '''

    # remove '.jpeg' from file name to get the FENs
    all_fens = [f[:-5] for f in listdir(load_path) if
                isfile(join(load_path, f))]

    # a set to store hashes; is used to detect (near) duplicates
    hashes = set()

    for i, fen in enumerate(all_fens):
        board_image = Image.open(join(load_path, fen + '.jpeg'))  # open image
        fen_list = fen_to_one_dim_array(fen)  # convert fen to 1d array

        # split the image in 64 squares
        if not grayscale:
            data = np.asarray(board_image)
            squares = view_as_blocks(data, (50, 50, 3)).reshape(64, 50, 50, 3)
        else:
            data = np.asarray(board_image).convert('L')
            squares = view_as_blocks(data, (50, 50, 1)).reshape(64, 50, 50, 1)

        # save each square as a new image
        for j, piece in enumerate(fen_list):
            save_path = join(save_to_path, piece)

            # check if dir exists; if not, create it
            if not isdir(save_path):
                makedirs(save_path)
            piece_image = Image.fromarray(squares[j])  # convert array to image

            if remove_duplicates:
                # use phash (perceptual hash) to remove near duplicates
                try:
                    image_hash = str(imagehash.phash(piece_image, hash_size))
                except Exception as e:
                    image_hash = None
                    print(f'Hash error for {piece}, image {i+1}_{j+1}: {e}')

                # check if hash already exists
                if image_hash and image_hash not in hashes:
                    hashes.add(image_hash)
                    piece_image.save(join(save_path, f'{i+1}_{j+1}.jpeg'))
            else:
                piece_image.save(join(save_path, f'{i+1}_{j+1}.jpeg'))
    print('Dataset created succesfully.')


if __name__ == '__main__':

    from constants import (
        TRAIN_PATH_BOARDS,
        TEST_PATH_BOARDS,
        TRAIN_PATH_PIECES,
        TEST_PATH_PIECES
    )

    # Create training dataset without duplicates
    create_piece_dataset(
        load_path=TRAIN_PATH_BOARDS,
        save_to_path=TRAIN_PATH_PIECES
    )

    # Create test dataset (with duplicates)
    create_piece_dataset(
        load_path=TEST_PATH_BOARDS,
        save_to_path=TEST_PATH_PIECES,
        remove_duplicates=False
    )
    print('All datasets created successfully.')
