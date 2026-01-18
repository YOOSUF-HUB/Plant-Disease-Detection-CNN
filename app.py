import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image, ImageOps

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'model.tflite'

# IMPORTANT: These must match the alphabetical order of the folders 
# you used for training (the "plant_data" folder names).
CLASS_NAMES = [
    'Pepper_Bacterial_Spot', 
    'Potato_Early_Blight', 
    'Tomato_Healthy',       # 'H' comes before 'L'
    'Tomato_Late_Blight'
]
# ---------------------

# 1. Load TFLite Model
# Note: If running on Mac/Linux without full TF, use: import tflite_runtime.interpreter as tf
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
except AttributeError:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    try:
        # 2. Preprocess Image
        image = Image.open(file)
        
        # Ensure RGB (removes Alpha channels if png)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize to 256x256 (Must match your training input)
        img = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
        
        # Convert to array & float32
        # Result is [0.0, 255.0]. We DO NOT divide by 255 here 
        # because your model has a Rescaling(1./255) layer inside.
        img_array = np.asarray(img).astype(np.float32)
        
        # Add batch dimension (1, 256, 256, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 3. Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Process Results
        predicted_index = np.argmax(output_data)
        predicted_class = CLASS_NAMES[predicted_index]
        
        # Get confidence as a percentage (0 to 100)
        confidence = float(np.max(output_data)) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)