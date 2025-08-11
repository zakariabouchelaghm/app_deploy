from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf 

app=FastAPI()

interpreter = tf.lite.Interpreter(model_path="handwriting_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class ModelInput(BaseModel):
    data:list

@app.post("/predict")

def predict(input_data: ModelInput):
    img_input = np.array(input_data.data, dtype="float32")
     
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction.argmax(axis=1)

    