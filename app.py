from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import tensorflow as tf 
import cv2
from io import BytesIO
from PIL import Image
import base64

app=FastAPI()

model = tf.keras.models.load_model("handwriting_model.h5")


@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")

    # Preprocessing (same as before)
    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img_input = np.array(input_data.data, dtype="float32")
     
    
    prediction = model.predict(img_input)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    return {"predicted_class": predicted_class}

    
