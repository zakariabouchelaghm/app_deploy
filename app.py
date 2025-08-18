from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import tensorflow as tf 
import cv2
from io import BytesIO
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://zakariabouchelaghm.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("handwriting_model.h5")
dummy = np.zeros((1,28,28,1), dtype=np.float32)
model.predict(dummy)

@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")

    # Preprocessing (same as before)
    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img_input = np.array(img, dtype="float32")
     
    img_input = np.expand_dims(img, axis=-1)   # (28, 28, 1)
    img_input = np.expand_dims(img_input, axis=0)  # (1, 28, 28, 1)
    Image.fromarray((img_input[0,:,:,0]*255).astype(np.uint8)).save("debug_input.png")
    prediction = model.predict(img_input)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return {"predicted_class": predicted_class}

    
