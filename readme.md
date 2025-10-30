# Chess Position to FEN String

This repo contains scripts and notebooks used in training a CNN model that can inspect chessboard images and then reconstruct the FEN string (https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) describing the chess position appearing in the considered image. Final project for my deep learning course.

If you want to try it yourself:
  - clone the repo locally: `git clone https://github.com/anandrei90/fen-generator`
  - create a virtual env, activate it and install the necessary packages: `pip install -r requirements.txt`
  - from the root directory of the repo run `python3 -m api.api_server`
  - navigate to http://127.0.0.1:8000/docs
  - upload your chessboard image using the infer_fen endpoint
