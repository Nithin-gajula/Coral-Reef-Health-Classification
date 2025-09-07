import os
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Define model path and class labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model/coral_ensemble_model.h5")
CLASS_LABELS = ["Bleached", "Healthy"]

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Set up image upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file is an allowed image format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for prediction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize image
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the class of an input image
@app.route('/predict',methods=['POST'])
def predict(image_path):
    """Predict the class of an input image."""
    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)

        if prediction.shape[1] == 1: 
            predicted_class = CLASS_LABELS[int(prediction[0][0] > 0.5)]
            confidence = float(prediction[0][0]) if predicted_class == "Healthy" else 1 - float(prediction[0][0])
        else:
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            confidence = np.max(prediction)

        print(f"🔹 Predicted Class: {predicted_class} with {confidence:.2f} confidence")
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Welcome page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Home page for image upload and classification
@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get file from form
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the uploaded image
            predicted_class, confidence = predict(file_path)

            if predicted_class is not None:
                return render_template('result.html', filename=filename, result=predicted_class, confidence=confidence)
            else:
                return "Error during prediction.", 500
    return render_template('index.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Retrieve data from the form
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # You can add code here to process or save the message
        print(f"Message from {name} ({email}): {message}")
        
        # Redirect to a thank-you page or back to contact
        return "Thank you for your message!"
    
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)

