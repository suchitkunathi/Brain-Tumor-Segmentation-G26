from flask import Flask, request, render_template, send_file
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

app = Flask(__name__)

# Dice Coefficient as a custom metric
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true), axis=-1) + K.sum(K.abs(y_pred), axis=-1)
    return (2. * intersection + smooth) / (sum_ + smooth)

# Dice Loss as a custom loss function
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Load the trained model with the custom functions
model = tf.keras.models.load_model(
    "unet/UNET_modelfiles/model.keras",
    custom_objects={"dice_coef": dice_coef, "dice_loss": dice_loss}
)

# Ensure the `static` folder exists
if not os.path.exists('static'):
    os.makedirs('static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        # Save the uploaded file
        filepath = os.path.join('static', 'uploaded_image.png')
        file.save(filepath)
        
        # Preprocess the image
        image = Image.open(filepath)
        image = image.resize((256, 256))  # Resize to match model's input
        image = image.convert("RGB")  # Convert to RGB if needed
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Predict segmentation
        segmentation_result = model.predict(image)[0]
        
        # Post-process segmentation result
        segmentation_result = (segmentation_result > 0.5).astype(np.uint8) * 255  # Binary mask
        segmentation_image = Image.fromarray(segmentation_result[:, :, 0])  # Use the first channel
        
        # Save the segmentation result
        result_path = os.path.join('static', 'segmented_image.png')
        segmentation_image.save(result_path)
        
        return render_template('result.html', original_image='uploaded_image.png', segmented_image='segmented_image.png')


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)
