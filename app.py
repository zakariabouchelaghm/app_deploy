from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf 
from fastapi.middleware.cors import CORSMiddleware
import cv2
from io import BytesIO
from PIL import Image
import base64

app=FastAPI()

interpreter = tf.lite.Interpreter(model_path="handwriting_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.post("/predict")

def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")

    # Preprocessing (same as before)
    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img_input = np.array(input_data.data, dtype="float32")
     
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(prediction.argmax(axis=1)[0])
    return {"predicted_class": predicted_class}

    
