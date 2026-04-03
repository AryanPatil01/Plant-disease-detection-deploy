import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Initialize Flask App
# Serving static files from the root directory for Render deployment
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for the frontend interactions

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Define parameters
MODEL_PATH = 'efficientnetb3-Plant Village Disease-99.71.h5'
TARGET_SIZE = (224, 224)

# 38 Classes from Plant Village dataset
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load model lazily
model = None

def get_model():
    global model
    if model is None:
        print("Loading model via tf_keras...")
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model

# Format the label to look cleaner
def format_disease_name(name):
    parts = name.split('___')
    if len(parts) == 2:
        plant, disease = parts
        disease = disease.replace('_', ' ')
        return f"{plant}: {disease}"
    return name.replace('_', ' ')

# --- CLAUDE's OPENCV PREPROCESSING ---
def remove_background_approx(img_array):
    """Aggressive center-crop based on green color dominance to remove background."""
    img = img_array.astype(np.uint8)
    
    # Use green channel dominance to find leaf region
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Mask pixels where green is dominant
    green_mask = (g.astype(int) - r.astype(int) > -30) & \
                 (g.astype(int) - b.astype(int) > -30)
    
    rows = np.any(green_mask, axis=1)
    cols = np.any(green_mask, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add small padding
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(img.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(img.shape[1], cmax + pad)
        
        img = img[rmin:rmax, cmin:cmax]
    return img

def preprocess_real_world(file_stream):
    """Full preprocessing pipeline for messy real-world images."""
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Isolate the leaf (cut out background)
    img_cropped = remove_background_approx(img)
    
    # 2. Resize to model input (224x224)
    img_resized = cv2.resize(img_cropped, TARGET_SIZE)
    
    # 3. Enhance contrast (CLAHE) - fixes shadows and harsh sunlight
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    return img_enhanced

# --- TEST TIME AUGMENTATION (TTA) ---
def apply_tta(img_batch):
    """Generates augmented versions of the input image batch for ensemble voting."""
    augmented = [img_batch] # 1. Original
    augmented.append(np.flip(img_batch, axis=2)) # 2. Horizontal Flip
    augmented.append(np.flip(img_batch, axis=1)) # 3. Vertical Flip
    
    # 4. Zoom (Central 90%)
    zoom = tf.image.central_crop(img_batch[0], 0.9)
    zoom = tf.image.resize(zoom, TARGET_SIZE).numpy()
    augmented.append(np.expand_dims(zoom, axis=0))
    
    return augmented

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file provided'}), 400
        
    try:
        # Preprocess the messy real-world image
        processed_img_array = preprocess_real_world(file.stream)
        
        # Format for model (Batch size 1, float32, RAW PIXELS 0-255)
        img_batch = np.expand_dims(processed_img_array.astype(np.float32), axis=0)
        
        m = get_model()
        
        # Run Test-Time Augmentation
        print("Running TTA (Analyzing multiple angles)...")
        tta_versions = apply_tta(img_batch)
        all_preds = []
        
        for version in tta_versions:
            # Pass raw pixels because Rescaling is built into the model
            preds = m.predict(version, verbose=0)
            all_preds.append(preds[0]) 
            
        # Average the scores from all augmented versions
        avg_scores = np.mean(all_preds, axis=0)
        
        # --- HIERARCHICAL FILTERING ---
        target_plant = request.form.get('plant', '').strip().lower()
        
        if target_plant:
            print(f"Filtering for plant: {target_plant}")
            valid_indices = [i for i, name in enumerate(class_labels) if target_plant in name.lower()]
            
            if valid_indices:
                # Filter scores
                filtered_scores = [avg_scores[i] for i in valid_indices]
                total = sum(filtered_scores)
                
                # Re-normalize
                normalized = [s/total for s in filtered_scores] if total > 0 else filtered_scores
                
                # Find best in filtered set
                local_idx = np.argmax(normalized)
                predicted_idx = valid_indices[local_idx]
                confidence = float(normalized[local_idx])
            else:
                # Fallback if plant name doesn't match any classes
                predicted_idx = np.argmax(avg_scores)
                confidence = float(avg_scores[predicted_idx])
        else:
            # Global prediction (AI Analyzed only)
            predicted_idx = np.argmax(avg_scores)
            confidence = float(avg_scores[predicted_idx])
        
        # Return the label and confidence
        label = class_labels[predicted_idx]
        formatted_label = format_disease_name(label)
        
        return jsonify({
            'success': True,
            'prediction': formatted_label,
            'confidence': confidence,
            'raw_class': label
        })
        
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
