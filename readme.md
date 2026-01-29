# Chess Position to FEN String

This repo contains scripts and notebooks used in training a CNN model that can inspect chessboard images and then reconstruct the FEN string (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) describing the chess position appearing in the considered image. Final project for my deep learning course. Data taken from https://www.kaggle.com/datasets/koryakinp/chess-positions.

The API contains two endpoints:
  - `infer_fen_from_chessboard`: behaves as intended only for chessboard images, such as the ones from https://www.kaggle.com/datasets/koryakinp/chess-positions;
  - `infer_fen_from_screenshot`: tries to find a chessboard in the inputed image/screenshot and infer its FEN. Can fail for some more unusual chessboard styles.

If you want to try it yourself:
  - clone the repo locally: `git clone https://github.com/anandrei90/fen-generator`
  - create a virtual env, activate it and install the necessary packages: `pip install -r requirements.txt`
  - from the root directory of the repo run `python3 -m api.api_server`
  - navigate to http://127.0.0.1:8000/docs
  - upload your chessboard image or screenshot using one of the two endpoints
