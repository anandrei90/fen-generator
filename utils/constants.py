import sys
from os.path import join

WORKING_DIR = sys.path[0]
TRAIN_PATH_BOARDS = join(WORKING_DIR, "data", "train")
TEST_PATH_BOARDS = join(WORKING_DIR, "data", "test")
TRAIN_PATH_PIECES = join(WORKING_DIR, "data", "pieces_train_no_duplicates")
TEST_PATH_PIECES = join(WORKING_DIR, "data", "pieces_test")
MODEL_PATH = join(WORKING_DIR, "model_checkpoints", "chess_pieces.keras")
BATCH_SIZE = 640
CLASS_NAMES = ['e', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
