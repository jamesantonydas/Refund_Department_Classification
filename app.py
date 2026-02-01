import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# model
MODEL_PATH = 'cnn_model.keras'
IMG_SIZE = (28, 28)
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")

load_model()

def preprocess_image(b64_string):
    """Decodes base64, resizes, converts to grayscale, and normalizes."""
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

# Health Check
@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200

# Batch Prediction
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    
    if not payload or 'batch' not in payload:
        print('payload missing')
        return jsonify({"error": "Missing 'batch' key"}), 400

    results = []
    batch_images = []
    item_ids = []

    try:
        for item in payload['batch']:
            b64_str = item.get('data') or item.get('data_b64')
            if b64_str:
                batch_images.append(preprocess_image(b64_str))
                item_ids.append(item.get('id', 'unknown'))

        if not batch_images:
            return jsonify({"error": "No valid image data found in batch"}), 400

        # Convert list to a single tensor
        input_tensor = np.stack(batch_images)
        
        # Inference
        predictions = model.predict(input_tensor, verbose=0)

        # Build response
        for i in range(len(item_ids)):
            results.append({
                "id": item_ids[i],
                "probabilities": predictions[i].tolist()
            })

        return jsonify({"results": results})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)