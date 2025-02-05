from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
model = tf.keras.models.load_model("model.h5")

templates = Jinja2Templates(directory="templates")

def preprocess_image(image_data):
    """Convert base64 image to MNIST format (28x28 grayscale)."""
    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST size
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1,28, 28,1)  # Reshape for model
    return image

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="home.html"
    )

@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
    """Receive an image from canvas, preprocess, and predict."""
    data = await request.json()
    image_data = data["image"].split(",")[1]  # Remove base64 prefix
    processed_image = preprocess_image(image_data)

    # Predict using the model
    predictions = model.predict(processed_image)
    print("Predictions:", predictions)
    predicted_digit = int(np.argmax(predictions))  # Get highest probability digit
    print("Predicted digit:", predicted_digit)

    return JSONResponse({"digit": predicted_digit})

if __name__ == "__main__":
    app.run(debug=True)