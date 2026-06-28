# app.py - FIXED VERSION
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tb-pneumonia-ai-diagnostic-system-secret-key-2024'

# Fix database path - use absolute path
base_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(base_dir, 'instance', 'app.db')
os.makedirs(os.path.join(base_dir, 'instance'), exist_ok=True)  # Create instance folder
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload configuration
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

print("="*60)
print("TB & Pneumonia Diagnostic System")
print("="*60)
print(f"Database path: {db_path}")
print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")

# ================= DATABASE MODELS =================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_doctor = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(500))
    symptoms = db.Column(db.Text)
    image_prediction = db.Column(db.String(50))
    image_confidence = db.Column(db.Float)
    rf_prediction = db.Column(db.String(50))
    rf_confidence = db.Column(db.Float)
    ann_prediction = db.Column(db.String(50))
    ann_confidence = db.Column(db.Float)
    final_prediction = db.Column(db.String(50))
    final_confidence = db.Column(db.Float)
    fusion_method = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= AI MODELS LOADING =================
print("\n" + "="*60)
print("Loading AI Models...")
print("="*60)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. MOBILENET FOR X-RAY IMAGES
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load MobileNet model
image_model = None
try:
    model_path = os.path.join(base_dir, "mobilenet.pt")
    if os.path.exists(model_path):
        image_model = MobileNetModel(num_classes=3)
        image_model.load_state_dict(torch.load(model_path, map_location=device))
        image_model = image_model.to(device)
        image_model.eval()
        print(" MobileNet model loaded successfully")
    else:
        print(f"  MobileNet model file not found at: {model_path}")
        print("  Image prediction will not work without mobilenet.pt")
except Exception as e:
    print(f"  Error loading MobileNet: {e}")
    print("  Running without image model - only symptom analysis will work")

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. SYMPTOM MODELS
symptom_scaler = None
label_encoder = None
rf_model = None
ann_model = None

try:
    symptom_scaler = joblib.load(os.path.join(base_dir, 'symptom_scaler.pkl'))
    label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    rf_model = joblib.load(os.path.join(base_dir, 'random_forest_symptom_model.pkl'))
    ann_model = joblib.load(os.path.join(base_dir, 'ann_symptom_model.pkl'))
    print(" Symptom models loaded successfully")
except Exception as e:
    print(f"  Error loading symptom models: {e}")
    print("  Symptom analysis may not work correctly")

# Symptom definitions
SYMPTOMS = [
    'fever', 'cough', 'productive_cough', 'rusty_sputum',
    'chest_pain', 'shortness_of_breath', 'fatigue',
    'night_sweats', 'weight_loss', 'chills', 'loss_of_appetite',
    'headache', 'muscle_aches', 'hemoptysis', 'sweating'
]

CLASS_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']

# ================= HELPER FUNCTIONS =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    """Predict disease from X-ray image using MobileNet"""
    try:
        if image_model is None:
            return "Model not loaded", np.array([0.33, 0.33, 0.33]), np.array([0.33, 0.33, 0.33])
        
        image = Image.open(image_path).convert('RGB')
        image = image_transform(image).unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            output = image_model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        probs = probabilities.cpu().numpy()[0]
        image_probs_ordered = np.array([probs[1], probs[0], probs[2]])
        
        prediction_idx = predicted.item()
        if prediction_idx == 0:
            final_prediction = "Pneumonia"
        elif prediction_idx == 1:
            final_prediction = "Normal"
        else:
            final_prediction = "Tuberculosis"
        
        return final_prediction, image_probs_ordered, probs
        
    except Exception as e:
        print(f"Image prediction error: {e}")
        return "Error", np.array([0.33, 0.33, 0.33]), np.array([0.33, 0.33, 0.33])

def predict_symptoms(symptom_vector):
    """Predict disease from symptoms using both RF and ANN"""
    try:
        if rf_model is None or ann_model is None:
            # Return dummy predictions if models not loaded
            return {
                'rf': {'label': 'Normal', 'probs': np.array([0.8, 0.1, 0.1]), 'confidence': 0.8},
                'ann': {'label': 'Normal', 'probs': np.array([0.7, 0.2, 0.1]), 'confidence': 0.7}
            }
        
        symptom_array = np.array(symptom_vector).reshape(1, -1)
        
        # RF prediction
        rf_proba = rf_model.predict_proba(symptom_array)[0]
        rf_pred = rf_model.predict(symptom_array)[0]
        rf_pred_label = label_encoder.inverse_transform([rf_pred])[0]
        
        # ANN prediction
        symptom_scaled = symptom_scaler.transform(symptom_array)
        ann_proba = ann_model.predict_proba(symptom_scaled)[0]
        ann_pred = ann_model.predict(symptom_scaled)[0]
        ann_pred_label = label_encoder.inverse_transform([ann_pred])[0]
        
        return {
            'rf': {'label': rf_pred_label, 'probs': rf_proba, 'confidence': max(rf_proba)},
            'ann': {'label': ann_pred_label, 'probs': ann_proba, 'confidence': max(ann_proba)}
        }
        
    except Exception as e:
        print(f"Symptom prediction error: {e}")
        return None

def late_fusion(image_probs, symptom_rf_probs, symptom_ann_probs, method='weighted_average'):
    """Perform late fusion"""
    if method == 'weighted_average':
        weights = {'image': 0.6, 'symptoms_rf': 0.25, 'symptoms_ann': 0.15}
        fused_probs = (
            weights['image'] * image_probs +
            weights['symptoms_rf'] * symptom_rf_probs +
            weights['symptoms_ann'] * symptom_ann_probs
        )
    elif method == 'product':
        fused_probs = image_probs * symptom_rf_probs * symptom_ann_probs
        fused_probs = fused_probs / fused_probs.sum()
    elif method == 'max':
        fused_probs = np.maximum.reduce([image_probs, symptom_rf_probs, symptom_ann_probs])
        fused_probs = fused_probs / fused_probs.sum()
    elif method == 'majority_vote':
        predictions = [np.argmax(image_probs), np.argmax(symptom_rf_probs), np.argmax(symptom_ann_probs)]
        final_class = np.bincount(predictions).argmax()
        fused_probs = np.zeros(3)
        fused_probs[final_class] = 1.0
    else:
        fused_probs = image_probs
    
    final_idx = np.argmax(fused_probs)
    final_label = CLASS_NAMES[final_idx]
    final_confidence = fused_probs[final_idx]
    
    return final_label, final_confidence, fused_probs

def get_clinical_recommendation(diagnosis, confidence):
    """Generate clinical recommendations"""
    recommendations = {
        'Normal': [
            'No immediate treatment required',
            'Follow up if symptoms persist or worsen',
            'Maintain regular health checkups'
        ],
        'Pneumonia': [
            'Consult a physician immediately',
            'Antibiotic therapy may be required',
            'Get plenty of rest and stay hydrated',
            'Monitor oxygen saturation levels',
            'Chest X-ray follow-up in 4-6 weeks'
        ],
        'Tuberculosis': [
            'Urgent consultation with pulmonologist required',
            'Start anti-tubercular therapy (ATT) immediately',
            'Isolation precautions recommended for 2-3 weeks',
            'Contact tracing for close contacts',
            'Regular follow-up during 6-month treatment',
            'Sputum tests for confirmation'
        ]
    }
    
    confidence_level = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
    
    return {
        'diagnosis': diagnosis,
        'confidence': f"{confidence:.1%}",
        'confidence_level': confidence_level,
        'recommendations': recommendations.get(diagnosis, ['Consult a healthcare professional']),
        'urgency': 'High' if diagnosis in ['Pneumonia', 'Tuberculosis'] else 'Low'
    }

# ================= ROUTES =================
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        is_doctor = request.form.get('is_doctor') == 'on'
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password_hash=hashed_password, is_doctor=is_doctor)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check username and password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get user's recent predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc())\
        .limit(10)\
        .all()
    
    return render_template('dashboard.html', 
                         user=current_user, 
                         predictions=predictions,
                         symptoms=SYMPTOMS)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Check if image was uploaded
        if 'xray_image' not in request.files:
            flash('No X-ray image uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['xray_image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('File type not allowed. Please upload PNG, JPG, JPEG, or BMP.', 'danger')
            return redirect(request.url)
        
        # Save uploaded file
        filename = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get symptoms from form
        symptom_vector = []
        symptoms_dict = {}
        for symptom in SYMPTOMS:
            value = request.form.get(symptom, '0')
            symptom_vector.append(int(value))
            if int(value) == 1:
                symptoms_dict[symptom] = 1
        
        # Get fusion method
        fusion_method = request.form.get('fusion_method', 'weighted_average')
        
        # 1. Predict from X-ray image
        if image_model:
            image_prediction, image_probs_ordered, _ = predict_image(filepath)
            image_confidence = max(image_probs_ordered)
        else:
            # Use symptom-only prediction if image model not available
            image_prediction = "Model not loaded"
            image_probs_ordered = np.array([0.33, 0.33, 0.33])
            image_confidence = 0.33
        
        # 2. Predict from symptoms
        symptom_predictions = predict_symptoms(symptom_vector)
        
        if symptom_predictions is None:
            flash('Error in symptom prediction. Please try again.', 'danger')
            return redirect(request.url)
        
        # 3. Perform late fusion
        final_label, final_confidence, fused_probs = late_fusion(
            image_probs_ordered,
            symptom_predictions['rf']['probs'],
            symptom_predictions['ann']['probs'],
            method=fusion_method
        )
        
        # 4. Generate clinical recommendations
        clinical_info = get_clinical_recommendation(final_label, final_confidence)
        
        # 5. Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=filepath,
            symptoms=str(symptoms_dict),
            image_prediction=image_prediction,
            image_confidence=float(image_confidence),
            rf_prediction=symptom_predictions['rf']['label'],
            rf_confidence=float(symptom_predictions['rf']['confidence']),
            ann_prediction=symptom_predictions['ann']['label'],
            ann_confidence=float(symptom_predictions['ann']['confidence']),
            final_prediction=final_label,
            final_confidence=float(final_confidence),
            fusion_method=fusion_method
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Store results in session for display
        session['last_prediction'] = {
            'image_path': f'static/uploads/{filename}',
            'image_prediction': image_prediction,
            'image_confidence': f"{image_confidence:.1%}",
            'rf_prediction': symptom_predictions['rf']['label'],
            'rf_confidence': f"{symptom_predictions['rf']['confidence']:.1%}",
            'ann_prediction': symptom_predictions['ann']['label'],
            'ann_confidence': f"{symptom_predictions['ann']['confidence']:.1%}",
            'final_prediction': final_label,
            'final_confidence': f"{final_confidence:.1%}",
            'fusion_method': fusion_method.replace('_', ' ').title(),
            'symptoms_present': [s.replace('_', ' ').title() for s, v in symptoms_dict.items() if v == 1],
            'clinical_info': clinical_info,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        flash('Diagnosis completed successfully!', 'success')
        return redirect(url_for('result'))
    
    return render_template('predict.html', symptoms=SYMPTOMS)

@app.route('/result')
@login_required
def result():
    """Display prediction result"""
    if 'last_prediction' not in session:
        flash('No recent prediction found. Please make a new diagnosis.', 'info')
        return redirect(url_for('predict'))
    
    result_data = session['last_prediction']
    return render_template('result.html', result=result_data)

@app.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc())\
        .all()
    
    # Pre-process symptoms data for the template
    processed_predictions = []
    for pred in predictions:
        # Convert symptoms string to dictionary
        symptoms_dict = {}
        if pred.symptoms and pred.symptoms != '{}':
            try:
                # Remove curly braces and quotes, then split
                symptoms_str = pred.symptoms.strip('{}').replace("'", "")
                if symptoms_str:
                    for item in symptoms_str.split(', '):
                        if ':' in item:
                            key, value = item.split(':')
                            symptoms_dict[key.strip()] = int(value.strip())
            except:
                symptoms_dict = {}
        
        processed_predictions.append({
            'id': pred.id,
            'created_at': pred.created_at,
            'image_path': pred.image_path,
            'symptoms_dict': symptoms_dict,
            'symptoms_count': len(symptoms_dict),
            'image_prediction': pred.image_prediction,
            'image_confidence': pred.image_confidence,
            'rf_prediction': pred.rf_prediction,
            'rf_confidence': pred.rf_confidence,
            'ann_prediction': pred.ann_prediction,
            'ann_confidence': pred.ann_confidence,
            'final_prediction': pred.final_prediction,
            'final_confidence': pred.final_confidence,
            'fusion_method': pred.fusion_method
        })
    
    return render_template('history.html', predictions=processed_predictions)

@app.route('/api/check_models')
def check_models():
    """API endpoint to check model status"""
    status = {
        'image_model': image_model is not None,
        'symptom_models': rf_model is not None and ann_model is not None,
        'device': str(device),
        'symptoms_count': len(SYMPTOMS),
        'classes': CLASS_NAMES
    }
    return jsonify(status)

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# ================= CREATE DEFAULT TEMPLATES =================
def create_default_templates():
    """Create basic HTML templates if they don't exist"""
    templates_dir = os.path.join(base_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create a simple 404.html
    with open(os.path.join(templates_dir, '404.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head><title>404 - Page Not Found</title></head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>The page you are looking for does not exist.</p>
    <a href="/">Return to Home</a>
</body>
</html>''')
    
    # Create a simple 500.html
    with open(os.path.join(templates_dir, '500.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head><title>500 - Internal Server Error</title></head>
<body>
    <h1>500 - Internal Server Error</h1>
    <p>Something went wrong on our server.</p>
    <a href="/">Return to Home</a>
</body>
</html>''')

# ================= INITIALIZE APPLICATION =================
with app.app_context():
    try:
        db.create_all()
        print(" Database initialized successfully")
    except Exception as e:
        print(f"  Database initialization error: {e}")
        print("  Creating database in current directory...")
        # Try alternative database path
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
        db.create_all()
        print(" Database created in current directory")

# Create default templates
create_default_templates()

# ================= RUN APPLICATION =================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting TB & Pneumonia Diagnostic System")
    print("="*60)
    print(f" Access the application at: http://localhost:5000")
    print(f" Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f" Uploads: {app.config['UPLOAD_FOLDER']}")
    
    if image_model:
        print(" Image model: LOADED")
    else:
        print("  Image model: NOT LOADED (mobilenet.pt required)")
    
    if rf_model and ann_model:
        print(" Symptom models: LOADED")
    else:
        print("  Symptom models: NOT FULLY LOADED")
    
    print("\n  Medical Disclaimer:")
    print("This tool is for educational/research purposes only.")
    print("Always consult healthcare professionals for medical diagnosis.")
    print("="*60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative port...")
        app.run(debug=True, host='0.0.0.0', port=5001)