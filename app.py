import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

print(f"Scikit-learn: {pickle.__version__}")
print(f"Pandas: {pd.__version__}")


# Load the model dictionary
with open('knn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)  # Load the dictionary

# Extract components from the dictionary
knn_model = model_data['model']
all_symptoms_loaded = model_data['symptoms']
class_names = model_data['classes']

# Prepare input correctly
user_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches']
input_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms_loaded]

# Create DataFrame with proper column order
user_input = pd.DataFrame([input_vector], columns=all_symptoms_loaded)

# Make prediction
prediction = knn_model.predict(user_input)
print(f"Predicted disease: {prediction[0]}")

@app.route('/')
def home():
    return f'User '
    
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all valid symptoms"""
    return jsonify({
        'symptoms': [s.replace('_', ' ') for s in all_symptoms_loaded],
        'count': len(all_symptoms_loaded)
    })


@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Get symptoms from request body
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
            
        user_symptoms = data['symptoms']
        
        # Validate symptoms and create input vector
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in user_symptoms:
            # Convert to underscore format
            formatted = symptom.strip().lower().replace(' ', '_')
            if formatted in all_symptoms_loaded:
                valid_symptoms.append(formatted)
            else:
                invalid_symptoms.append(symptom)
        
        if invalid_symptoms:
            return jsonify({
                'error': 'Invalid symptoms detected',
                'invalid_symptoms': invalid_symptoms,
                'valid_symptoms': all_symptoms_loaded  # or provide suggestions
            }), 400
        
        # Create input vector
        input_vector = [1 if symptom in valid_symptoms else 0 for symptom in all_symptoms_loaded]
        
        # Make prediction
        user_input = pd.DataFrame([input_vector], columns=all_symptoms_loaded)
        prediction = knn_model.predict(user_input)
        probabilities = knn_model.predict_proba(user_input)[0]
        
        # Get top predictions with probabilities
        top_predictions = sorted(
            [(prob, cls) for cls, prob in zip(class_names, probabilities)],
            reverse=True
        )[:3]  # Get top 3 predictions
        
        # Format response
        response = {
            'prediction': prediction[0],
            'confidence': f"{max(probabilities)*100:.2f}%",
            'top_predictions': [
                {'disease': cls, 'confidence': f"{prob*100:.2f}%"}
                for prob, cls in top_predictions
            ],
            'warning': 'This is not medical advice. Consult a healthcare professional.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)