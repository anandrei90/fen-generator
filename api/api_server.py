import uvicorn
from tensorflow.keras.models import load_model
from fastapi import FastAPI, Response, UploadFile, File
from contextlib import asynccontextmanager
import io
from PIL import Image
from utils.constants import MODEL_PATH
from utils.helpers import (one_dim_array_to_fen, crop_chessboard,
                           preprocess_image, classify_squares,
                           save_fig_in_buffer)

# ml model dict for the async context manager
ml_models = {}  # does it have to be a dict?


# ensures ML model is loaded only once
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["chess_pieces"] = load_model(MODEL_PATH)
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
@app.post("/infer_fen_from_chessboard/")
async def upload_chessboard_image(file: UploadFile = File(...)):
    """
    Lets the user upload a chessboard image file to the endpoint.
    The endpoint breaks the chessboard image into squares and
    classifies each square using a pre-trained ML model.
    Finally, the endpoint infers the FEN string using the classified squares,
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

    # preprocess chessboard image
    image = preprocess_image(image)

    # classify squares from chessboard image
    pieces = classify_squares(image=image, model=ml_models["chess_pieces"])

    # construct fen string from pieces
    fen_string = one_dim_array_to_fen(pieces, separator='-')

    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()

    # save the figure in the buffer
    save_fig_in_buffer(figure=figure, caption=fen_string, buffer=buffer)

    # move the file pointer back to the start of the buffer so it can be read
    buffer.seek(0)

    # extract the binary image from the buffer
    binary_image = buffer.getvalue()

    # send the binary image as a png response to the client
    return Response(binary_image, media_type="image/png")


# endpoint for inferring FEN from screenshot
@app.post("/infer_fen_from_screenshot/")
async def upload_screenshot(file: UploadFile = File(...)):
    """
    Lets the user upload a screenshot containing exactly one chessboard image.
    The endpoint indentifies and crops out the chessboard, breaks it into
    squares and classifies each square using a pre-trained ML model. Finally,
    the endpoint infers the FEN string using the classified squares, and
    returns the chessboard image with the FEN string overlayed.

    Parameters
    ----------
    file : UploadFile (FastAPI-form)
        The image file to be uploaded.

    Returns
    -------
    Response: image/png
        The detected chessboard image with the inferred FEN string overlayed.
    """

    # read screenshot
    screenshot_data = await file.read()
    screenshot = Image.open(io.BytesIO(screenshot_data))

    # detect & crop out chessboard from screenshot
    image = crop_chessboard(screenshot)

    # return message if chessboard detection fails
    if image is None:
        return "Failed to detect chessboard in image."

    # save figure as PIL.Image object for later
    figure = Image.fromarray(image)

    # preprocess image
    image = preprocess_image(image)

    # classify squares from chessboard image
    pieces = classify_squares(image=image, model=ml_models["chess_pieces"])

    # construct fen string from pieces
    fen_string = one_dim_array_to_fen(pieces, separator='-')

    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()

    # save the figure in the buffer
    save_fig_in_buffer(figure=figure, caption=fen_string, buffer=buffer)

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
