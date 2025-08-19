from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from werkzeug.utils import secure_filename
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add error handlers
@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {e}")
    return jsonify(error=str(e)), 500

@app.errorhandler(404)
def not_found(e):
    app.logger.error(f"Not found: {e}")
    return jsonify(error=str(e)), 404

# Load the model - update path to your .h5 file
MODEL_PATH = 'cnn_emotion_model.h5'  # Updated model name

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    app.logger.error(f"Model file not found: {MODEL_PATH}")
    model = None
else:
    try:
        model = load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        model = None

# Define emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_face(face):
    try:
        # Resize face to match model input requirements (typically 48x48 for emotion models)
        face = cv2.resize(face, (48, 48))
        
        # Convert to grayscale if your model was trained on grayscale images
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        face = face / 255.0
        
        # Reshape for the model input (add channel dimension if grayscale)
        face = np.expand_dims(face, axis=-1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    except Exception as e:
        app.logger.error(f"Error in preprocess_face: {e}")
        return None

def process_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not read image"
        
        # Create a copy for displaying results
        result_image = image.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw number of faces detected
        cv2.putText(result_image, f"Faces detected: {len(faces)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Results dictionary
        results = {
            'faces_detected': len(faces),
            'face_data': []
        }
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_result = {
                'position': (x, y, w, h),
                'emotions': {}
            }
            
            if model is not None:
                # Preprocess face for emotion detection
                processed_face = preprocess_face(face_roi)
                
                if processed_face is not None:
                    # Make prediction
                    emotion_predictions = model.predict(processed_face)[0]
                    
                    # Get the emotion with highest probability
                    emotion_idx = np.argmax(emotion_predictions)
                    emotion_label = EMOTION_CLASSES[emotion_idx]
                    emotion_prob = float(emotion_predictions[emotion_idx])
                    
                    # Save all emotion probabilities
                    for i, emotion in enumerate(EMOTION_CLASSES):
                        face_result['emotions'][emotion] = float(emotion_predictions[i])
                    
                    # Prepare label with emotion and probability
                    label = f"{emotion_label}: {emotion_prob:.2f}"
                    
                    # Add text background for better visibility
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), (0, 255, 0), cv2.FILLED)
                    
                    # Add emotion label
                    cv2.putText(result_image, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            results['face_data'].append(face_result)
        
        # Generate unique filename for processed image
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Save the processed image
        cv2.imwrite(result_path, result_image)
        
        # Add the image paths to results - store only the filenames, not full paths
        results['original_image'] = os.path.basename(image_path)
        results['result_image'] = os.path.basename(result_path)
        
        return results, None
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            # Generate unique filename to prevent overwriting
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the uploaded file
            file.save(file_path)
            
            # Process the image
            results, error = process_image(file_path)
            
            if error:
                return render_template('index.html', error=error)
            
            # Return the results page with just the filenames
            return render_template('index.html', 
                                  result=results, 
                                  original_image=results['original_image'],
                                  result_image=results['result_image'])
        except Exception as e:
            app.logger.error(f"Error in upload_file: {e}")
            return render_template('index.html', error=str(e))
    
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/status')
def status():
    status_info = {
        'status': 'running',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    }
    return jsonify(status_info)

@app.route('/debug')
def debug():
    return jsonify({
        'routes': [str(rule) for rule in app.url_map.iter_rules()],
        'static_folder': app.static_folder,
        'template_folder': app.template_folder,
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

if __name__ == '__main__':
    app.run(debug=True)