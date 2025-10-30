import uvicorn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from fastapi import FastAPI, Response, UploadFile, File
from contextlib import asynccontextmanager
import io
from PIL import Image
from constants import MODEL_PATH, CLASS_NAMES
from helpers import one_dim_array_to_fen


ml_models = {}  # does it have to be a dict?


# ensures ML model is loaded only once
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["chess_pieces"] = keras.models.load_model(MODEL_PATH)
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# make app
app = FastAPI(
    title="Infer FEN from Chessboard Image",
    description="API server to infer FEN from chessboard image using a CNN.",
    lifespan=lifespan
              )


# root endpoint
@app.get("/")
def home():
    """
    Serves as root for API.
    """
    return "Root of this API."


# endpoint for inferring FEN from chessboard image
@app.post("/infer_fen/")
async def upload_chessboard_image(file: UploadFile = File(...)):
    """
    Lets the user upload a chessboard image file to the API.
    The API breaks the chessboard image into squares and
    classifies each square using a pre-trained ML model.
    Finally, the API infers the FEN string using the classified squares,
    and returns the chessboard image with the FEN string overlayed.

    Parameters
    ----------
    file : UploadFile (FastAPI-form)
        The image file to be uploaded.

    Returns
    -------
    Response: image/png
        The chessboard image with the inferred FEN string overlayed.
    """

    # read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # save figure object for later
    figure = image

    # load image
    image = image.convert("RGB")
    image = image.resize((400, 400))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # (400, 400, 3) -> (1, 400, 400, 3)

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
            prediction = ml_models["chess_pieces"].predict(square, verbose=0)
            # get index of max prediction
            piece_index = np.argmax(prediction, axis=1)[0]
            piece = CLASS_NAMES[piece_index]
            # append piece to fen string
            pieces.append(piece)

    # construct fen string from pieces
    fen_string = one_dim_array_to_fen(pieces, separator='-')

    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()

    # create the figure with matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(figure)
    # add fen string as title to the figure
    plt.title(f"FEN: {fen_string}", fontsize=16)
    plt.axis('off')
    # save the plot in the buffer as a png
    plt.savefig(buffer, format="png")
    plt.close()

    # move the file pointer back to the start of the buffer so it can be read
    buffer.seek(0)

    # extract the binary image from the buffer
    binary_image = buffer.getvalue()

    # send the binary image as a png response to the client
    return Response(binary_image, media_type="image/png")


# my localhost adress
host = "127.0.0.1"

# run server
if __name__ == "__main__":
    # GUI at http://127.0.0.1:8000/docs
    uvicorn.run(app, host=host, port=8000, root_path="/")
