# test_late_fusion.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LATE FUSION VALIDATION TEST")
print("="*60)

# ================= CONFIGURATION =================
# Test image paths (update these to match your actual test images)
TEST_IMAGES = {
    "Pneumonia": r"DATA\test\Pneumonia\BACTERIA-1351146-0002.jpeg",  # Update path
    "Normal": r"DATA\test\Normal\CHNCXR_0320_0.png",  # Update path
    "Tuberculosis": r"DATA\test\Tuberculosis\CHNCXR_0365_1.png"  # Update path
}

# Test symptom vectors for each disease
TEST_SYMPTOMS = {
    "Pneumonia": [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],  # fever, cough, productive_cough, etc.
    "Normal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Only headache maybe
    "Tuberculosis": [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]  # cough, night_sweats, weight_loss, etc.
}

# Symptom names (must match training)
SYMPTOMS = [
    'fever', 'cough', 'productive_cough', 'rusty_sputum',
    'chest_pain', 'shortness_of_breath', 'fatigue',
    'night_sweats', 'weight_loss', 'chills', 'loss_of_appetite',
    'headache', 'muscle_aches', 'hemoptysis', 'sweating'
]

CLASS_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']

# ================= 1. LOAD MODELS =================
print("\n1. LOADING MODELS...")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1.1 Load MobileNet for X-ray images
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

try:
    image_model = MobileNetModel(num_classes=3)
    image_model.load_state_dict(torch.load("mobilenet.pt", map_location=device))
    image_model = image_model.to(device)
    image_model.eval()
    print("✅ MobileNet model loaded successfully")
except Exception as e:
    print(f"❌ Error loading MobileNet: {e}")
    exit()

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1.2 Load symptom models
try:
    symptom_scaler = joblib.load('symptom_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    rf_model = joblib.load('random_forest_symptom_model.pkl')
    ann_model = joblib.load('ann_symptom_model.pkl')
    print("✅ Symptom models loaded successfully")
except Exception as e:
    print(f"❌ Error loading symptom models: {e}")
    exit()

# ================= 2. PREDICTION FUNCTIONS =================
def predict_image(image_path):
    """Predict disease from X-ray image using MobileNet"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image_transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        
        with torch.no_grad():
            output = image_model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        # Get all probabilities
        probs = probabilities.cpu().numpy()[0]
        
        # Convert from your label mapping {0: "Pneumonia", 1: "Normal", 2: "Tuberculosis"}
        # to standard order: [Normal, Pneumonia, Tuberculosis]
        image_probs_ordered = np.array([probs[1], probs[0], probs[2]])
        
        prediction_idx = predicted.item()
        # Convert prediction index to standard order
        if prediction_idx == 0:  # Pneumonia in your mapping
            final_prediction = "Pneumonia"
        elif prediction_idx == 1:  # Normal in your mapping
            final_prediction = "Normal"
        else:  # Tuberculosis in your mapping
            final_prediction = "Tuberculosis"
        
        return final_prediction, image_probs_ordered, probs
        
    except Exception as e:
        print(f"Image prediction error for {image_path}: {e}")
        return "Error", np.array([0.33, 0.33, 0.33]), np.array([0.33, 0.33, 0.33])

def predict_symptoms(symptom_vector, verbose=False):
    """Predict disease from symptoms using both RF and ANN"""
    try:
        # Convert to numpy array
        symptom_array = np.array(symptom_vector).reshape(1, -1)
        
        # RF prediction
        rf_proba = rf_model.predict_proba(symptom_array)[0]
        rf_pred = rf_model.predict(symptom_array)[0]
        rf_pred_label = label_encoder.inverse_transform([rf_pred])[0]
        
        # ANN prediction (needs scaling)
        symptom_scaled = symptom_scaler.transform(symptom_array)
        ann_proba = ann_model.predict_proba(symptom_scaled)[0]
        ann_pred = ann_model.predict(symptom_scaled)[0]
        ann_pred_label = label_encoder.inverse_transform([ann_pred])[0]
        
        if verbose:
            print(f"  RF: {rf_pred_label} (Probs: {rf_proba.round(3)})")
            print(f"  ANN: {ann_pred_label} (Probs: {ann_proba.round(3)})")
        
        return {
            'rf': {'label': rf_pred_label, 'probs': rf_proba},
            'ann': {'label': ann_pred_label, 'probs': ann_proba}
        }
        
    except Exception as e:
        print(f"Symptom prediction error: {e}")
        return None

def late_fusion(image_probs, symptom_rf_probs, symptom_ann_probs, method='weighted_average'):
    """
    Perform late fusion using different methods
    
    Args:
        image_probs: [Normal, Pneumonia, Tuberculosis]
        symptom_probs: [Normal, Pneumonia, Tuberculosis] (same order)
        method: 'weighted_average', 'product', 'max', 'majority_vote'
    """
    if method == 'weighted_average':
        # Weights for fusion
        weights = {
            'image': 0.6,      # X-ray usually more reliable
            'symptoms_rf': 0.25,
            'symptoms_ann': 0.15
        }
        
        # Calculate weighted average
        fused_probs = (
            weights['image'] * image_probs +
            weights['symptoms_rf'] * symptom_rf_probs +
            weights['symptoms_ann'] * symptom_ann_probs
        )
        
    elif method == 'product':
        # Product of probabilities (assumes independence)
        fused_probs = image_probs * symptom_rf_probs * symptom_ann_probs
        fused_probs = fused_probs / fused_probs.sum()  # Normalize
        
    elif method == 'max':
        # Take maximum probability for each class
        fused_probs = np.maximum.reduce([image_probs, symptom_rf_probs, symptom_ann_probs])
        fused_probs = fused_probs / fused_probs.sum()  # Normalize
        
    elif method == 'majority_vote':
        # Convert probabilities to predictions and vote
        predictions = [
            np.argmax(image_probs),
            np.argmax(symptom_rf_probs),
            np.argmax(symptom_ann_probs)
        ]
        final_class = np.bincount(predictions).argmax()
        fused_probs = np.zeros(3)
        fused_probs[final_class] = 1.0
        
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    # Get final prediction
    final_idx = np.argmax(fused_probs)
    final_label = CLASS_NAMES[final_idx]
    final_confidence = fused_probs[final_idx]
    
    return final_label, final_confidence, fused_probs

def get_symptom_names(symptom_vector):
    """Get names of present symptoms"""
    present = [SYMPTOMS[i] for i, val in enumerate(symptom_vector) if val == 1]
    return present

# ================= 3. TEST FUSION WITH SAMPLE CASES =================
print("\n" + "="*60)
print("3. TESTING LATE FUSION")
print("="*60)

test_results = []

for disease_name, image_path in TEST_IMAGES.items():
    print(f"\n🧪 Testing {disease_name} case:")
    print(f"  Image: {image_path}")
    print(f"  Expected diagnosis: {disease_name}")
    
    # Get symptom vector for this disease
    symptom_vector = TEST_SYMPTOMS[disease_name]
    present_symptoms = get_symptom_names(symptom_vector)
    print(f"  Symptoms: {', '.join(present_symptoms[:5])}{'...' if len(present_symptoms) > 5 else ''}")
    
    # 1. Predict from image
    print("  📷 Image prediction:")
    image_pred, image_probs_ordered, image_probs_raw = predict_image(image_path)
    print(f"    Prediction: {image_pred}")
    print(f"    Probabilities: Normal={image_probs_ordered[0]:.3f}, "
          f"Pneumonia={image_probs_ordered[1]:.3f}, "
          f"TB={image_probs_ordered[2]:.3f}")
    
    # 2. Predict from symptoms
    print("  📝 Symptom prediction:")
    symptom_preds = predict_symptoms(symptom_vector, verbose=True)
    
    if symptom_preds:
        # 3. Perform late fusion with different methods
        print("  🔄 Late Fusion Results:")
        
        fusion_methods = ['weighted_average', 'product', 'max', 'majority_vote']
        fusion_results = {}
        
        for method in fusion_methods:
            final_label, final_confidence, fused_probs = late_fusion(
                image_probs_ordered,
                symptom_preds['rf']['probs'],
                symptom_preds['ann']['probs'],
                method=method
            )
            
            fusion_results[method] = {
                'label': final_label,
                'confidence': final_confidence,
                'probs': fused_probs,
                'correct': final_label == disease_name
            }
            
            print(f"    {method.replace('_', ' ').title():20} → {final_label} "
                  f"(Confidence: {final_confidence:.1%}, "
                  f"Correct: {'✓' if final_label == disease_name else '✗'})")
        
        # Store results
        test_results.append({
            'disease': disease_name,
            'image_pred': image_pred,
            'image_correct': image_pred == disease_name,
            'rf_pred': symptom_preds['rf']['label'],
            'rf_correct': symptom_preds['rf']['label'] == disease_name,
            'ann_pred': symptom_preds['ann']['label'],
            'ann_correct': symptom_preds['ann']['label'] == disease_name,
            'fusion_results': fusion_results,
            'symptoms_present': len(present_symptoms)
        })
    
    print("  " + "-"*40)

# ================= 4. PERFORMANCE ANALYSIS =================
print("\n" + "="*60)
print("4. PERFORMANCE ANALYSIS")
print("="*60)

if test_results:
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(test_results)
    
    # Calculate accuracy for each model
    print("\n📊 Accuracy by Model:")
    print(f"{'Model':20} {'Correct':10} {'Total':10} {'Accuracy':10}")
    print("-" * 50)
    
    models_to_check = [
        ('Image Model (MobileNet)', 'image_correct'),
        ('Symptom Model (RF)', 'rf_correct'),
        ('Symptom Model (ANN)', 'ann_correct')
    ]
    
    for model_name, col_name in models_to_check:
        correct = results_df[col_name].sum()
        total = len(results_df)
        accuracy = correct / total
        print(f"{model_name:20} {correct:10} {total:10} {accuracy:10.1%}")
    
    # Fusion method comparison
    print("\n🎯 Fusion Method Performance:")
    print(f"{'Method':20} {'Correct':10} {'Avg Confidence':15}")
    print("-" * 50)
    
    fusion_summary = {}
    for method in fusion_methods:
        correct_count = 0
        total_confidence = 0
        
        for result in test_results:
            fusion_result = result['fusion_results'][method]
            if fusion_result['correct']:
                correct_count += 1
            total_confidence += fusion_result['confidence']
        
        avg_confidence = total_confidence / len(test_results) if test_results else 0
        fusion_summary[method] = {
            'correct': correct_count,
            'accuracy': correct_count / len(test_results),
            'avg_confidence': avg_confidence
        }
        
        print(f"{method.replace('_', ' ').title():20} "
              f"{correct_count:10} / {len(test_results)} "
              f"{avg_confidence:15.1%}")
    
    # Find best fusion method
    best_method = max(fusion_summary.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best fusion method: {best_method[0].replace('_', ' ').title()} "
          f"(Accuracy: {best_method[1]['accuracy']:.1%})")
    
    # Case-by-case analysis
    print("\n🔍 Case-by-Case Details:")
    for result in test_results:
        print(f"\n{result['disease']}:")
        print(f"  Image: {result['image_pred']} ({'✓' if result['image_correct'] else '✗'})")
        print(f"  RF: {result['rf_pred']} ({'✓' if result['rf_correct'] else '✗'})")
        print(f"  ANN: {result['ann_pred']} ({'✓' if result['ann_correct'] else '✗'})")
        
        # Show fusion results for this case
        for method, fusion_result in result['fusion_results'].items():
            correct_symbol = '✓' if fusion_result['correct'] else '✗'
            print(f"  {method}: {fusion_result['label']} "
                  f"(conf: {fusion_result['confidence']:.1%}) {correct_symbol}")

# ================= 5. ADVANCED FUSION TEST =================
print("\n" + "="*60)
print("5. ADVANCED FUSION TEST: AMBIGUOUS CASES")
print("="*60)

# Test ambiguous cases where models might disagree
print("\n🧬 Testing ambiguous cases with conflicting predictions:")

# Case 1: TB image but Pneumonia symptoms
print("\nCase 1: TB X-ray + Pneumonia symptoms")
tb_image_path = TEST_IMAGES["Tuberculosis"]
pneumonia_symptoms = TEST_SYMPTOMS["Pneumonia"]

print("  Expected: Models should detect conflict")
print("  Ideal: Fusion should flag low confidence or suggest review")

image_pred, image_probs, _ = predict_image(tb_image_path)
print(f"  Image predicts: {image_pred} (probs: {image_probs.round(3)})")

symptom_preds = predict_symptoms(pneumonia_symptoms)
if symptom_preds:
    print(f"  Symptoms predict: RF={symptom_preds['rf']['label']}, "
          f"ANN={symptom_preds['ann']['label']}")
    
    # Try fusion
    final_label, final_confidence, fused_probs = late_fusion(
        image_probs,
        symptom_preds['rf']['probs'],
        symptom_preds['ann']['probs']
    )
    
    print(f"  Fusion result: {final_label} (confidence: {final_confidence:.1%})")
    
    # Check confidence level
    if final_confidence < 0.6:
        print("  ⚠️  LOW CONFIDENCE: This case should be flagged for clinician review")
    else:
        print("  ✅ Moderate/High confidence: Fusion resolved the conflict")

# Case 2: All models agree
print("\nCase 2: Consistent predictions (Pneumonia image + Pneumonia symptoms)")
pneumonia_image_path = TEST_IMAGES["Pneumonia"]

image_pred, image_probs, _ = predict_image(pneumonia_image_path)
symptom_preds = predict_symptoms(pneumonia_symptoms)

if symptom_preds:
    final_label, final_confidence, fused_probs = late_fusion(
        image_probs,
        symptom_preds['rf']['probs'],
        symptom_preds['ann']['probs']
    )
    
    print(f"  All models agree on: {final_label}")
    print(f"  Fusion confidence: {final_confidence:.1%}")
    
    if final_confidence > 0.9:
        print("  ✅ HIGH CONFIDENCE: Reliable automated diagnosis")

# ================= 6. VISUALIZE FUSION PROCESS =================
print("\n" + "="*60)
print("6. FUSION PROCESS VISUALIZATION")
print("="*60)

# Show detailed probability flow for one case
print("\n📈 Probability Flow for Pneumonia Case:")

test_case = "Pneumonia"
image_path = TEST_IMAGES[test_case]
symptom_vector = TEST_SYMPTOMS[test_case]

image_pred, image_probs, _ = predict_image(image_path)
symptom_preds = predict_symptoms(symptom_vector)

if symptom_preds:
    print(f"\nInput Probabilities:")
    print(f"{'Model':20} {'Normal':10} {'Pneumonia':10} {'Tuberculosis':10}")
    print("-" * 60)
    print(f"{'Image (MobileNet)':20} {image_probs[0]:10.3f} {image_probs[1]:10.3f} {image_probs[2]:10.3f}")
    print(f"{'Symptoms (RF)':20} {symptom_preds['rf']['probs'][0]:10.3f} "
          f"{symptom_preds['rf']['probs'][1]:10.3f} {symptom_preds['rf']['probs'][2]:10.3f}")
    print(f"{'Symptoms (ANN)':20} {symptom_preds['ann']['probs'][0]:10.3f} "
          f"{symptom_preds['ann']['probs'][1]:10.3f} {symptom_preds['ann']['probs'][2]:10.3f}")
    
    # Show weighted fusion calculation
    print(f"\nWeighted Average Calculation (0.6*Image + 0.25*RF + 0.15*ANN):")
    
    weights = {'image': 0.6, 'symptoms_rf': 0.25, 'symptoms_ann': 0.15}
    
    for i, disease in enumerate(CLASS_NAMES):
        weighted_sum = (
            weights['image'] * image_probs[i] +
            weights['symptoms_rf'] * symptom_preds['rf']['probs'][i] +
            weights['symptoms_ann'] * symptom_preds['ann']['probs'][i]
        )
        
        print(f"  {disease:12}: 0.6*{image_probs[i]:.3f} + "
              f"0.25*{symptom_preds['rf']['probs'][i]:.3f} + "
              f"0.15*{symptom_preds['ann']['probs'][i]:.3f} = {weighted_sum:.3f}")
    
    # Get final result
    final_label, final_confidence, fused_probs = late_fusion(
        image_probs, symptom_preds['rf']['probs'], symptom_preds['ann']['probs']
    )
    
    print(f"\nFinal Fused Probabilities: {fused_probs.round(3)}")
    print(f"Diagnosis: {final_label} (Confidence: {final_confidence:.1%})")

# ================= 7. SAVE TEST RESULTS =================
print("\n" + "="*60)
print("7. SAVING TEST RESULTS")
print("="*60)

# Save detailed results to CSV
if test_results:
    # Create a flat structure for CSV
    csv_data = []
    for result in test_results:
        row = {
            'disease': result['disease'],
            'image_prediction': result['image_pred'],
            'image_correct': result['image_correct'],
            'rf_prediction': result['rf_pred'],
            'rf_correct': result['rf_correct'],
            'ann_prediction': result['ann_pred'],
            'ann_correct': result['ann_correct'],
            'symptoms_count': result['symptoms_present']
        }
        
        # Add fusion results
        for method, fusion_result in result['fusion_results'].items():
            row[f'fusion_{method}_pred'] = fusion_result['label']
            row[f'fusion_{method}_conf'] = fusion_result['confidence']
            row[f'fusion_{method}_correct'] = fusion_result['correct']
        
        csv_data.append(row)
    
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv('late_fusion_test_results.csv', index=False)
    print("✅ Results saved to 'late_fusion_test_results.csv'")
    
    # Print summary statistics
    print("\n📋 Test Summary:")
    print(f"Total test cases: {len(test_results)}")
    
    # Calculate overall accuracy
    overall_correct = sum([1 for r in test_results 
                          if r['fusion_results']['weighted_average']['correct']])
    overall_accuracy = overall_correct / len(test_results)
    
    print(f"Overall fusion accuracy: {overall_correct}/{len(test_results)} = {overall_accuracy:.1%}")

print("\n" + "="*60)
print("LATE FUSION TEST COMPLETE ✓")
print("="*60)
print("\n✅ Next steps:")
print("1. Check the CSV file for detailed results")
print("2. Adjust fusion weights if needed")
print("3. Test with more diverse cases")
print("4. Proceed to build the Flask web application")