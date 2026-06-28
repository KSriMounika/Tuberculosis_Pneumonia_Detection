# models.py
from app import db
from datetime import datetime
from flask_login import UserMixin

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
    symptoms = db.Column(db.Text)  # JSON string of symptoms
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