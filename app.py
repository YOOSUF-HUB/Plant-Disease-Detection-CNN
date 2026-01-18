import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image, ImageOps

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'model.tflite'
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 
    'Potato___Early_blight', 
    'Tomato_healthy', 
    'Tomato_Late_blight'
]
# ---------------------

# 1. Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
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
        # Resize to 256x256 (Must match your training input)
        img = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
        
        # Convert to array & float32
        img_array = np.asarray(img).astype(np.float32)
        
        # Add batch dimension (1, 256, 256, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 3. Run Inference (TFLite style)
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        
        # Run the calculation
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Process Results
        predicted_index = np.argmax(output_data)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(output_data))

        # Adjust confidence logic if your model outputs logits (raw numbers) instead of probabilities
        # If confidence is like 14.5 or -3.2, we might need softmax here.
        # Assuming your model ends with Softmax, this is fine.

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)