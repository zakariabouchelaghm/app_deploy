from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf 
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
interpreter = tf.lite.Interpreter(model_path="handwriting_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class ImageData(BaseModel):
    image: str

@app.post("/predict")

def predict(data: ImageData):
    header, encoded = data.image.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(img_bytes)).convert("L")

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

    
