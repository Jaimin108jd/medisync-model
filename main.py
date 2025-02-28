import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from joblib import dump, load

# Load training data
training_data = pd.read_csv('Training.csv')

# Prepare training data
symptoms_train = training_data.drop(['prognosis', 'Unnamed: 133'], axis=1)
prognoses_train = training_data['prognosis']

# Get the list of all symptoms (column names)
all_symptoms = symptoms_train.columns.tolist()

# Find optimal K for KNN model
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 30)}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(symptoms_train, prognoses_train)
best_k = grid_search.best_params_['n_neighbors']
print(f"Best number of neighbors (k): {best_k}")

# Train KNN model with optimal K
knn_optimized = KNeighborsClassifier(n_neighbors=best_k)
knn_optimized.fit(symptoms_train, prognoses_train)

# After training the model
model_data = {
    'model': knn_optimized,
    'symptoms': all_symptoms,
    'classes': knn_optimized.classes_
}

# Save with compression
dump(model_data, 'disease_model.joblib', compress=3)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully as disease_predictor.pkl")

# Load and evaluate on testing data
testing_data = pd.read_csv('Testing.csv')
symptoms_test = testing_data.drop(['prognosis'], axis=1)
prognoses_test = testing_data['prognosis']

# Make predictions
prognoses_predicted = knn_optimized.predict(symptoms_test)
accuracy = accuracy_score(prognoses_test, prognoses_predicted)
conf_matrix = confusion_matrix(prognoses_test, prognoses_predicted)
class_report = classification_report(prognoses_test, prognoses_predicted)

# Print evaluation results
print(f"Accuracy with optimized k: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

def predict_disease():
    """
    Function to take user input for symptoms and predict disease
    """
    print("\n===== Disease Prediction System =====")
    print("Please answer with 'yes' or 'no' for the following symptoms:")
    
    # Create an empty dictionary to store symptom inputs
    user_symptoms = {}
    
    # Initialize all symptoms to 0 (absent)
    for symptom in all_symptoms:
        user_symptoms[symptom] = 0
    
    # Ask user about their symptoms
    print("\nEnter 'done' anytime to finish input and see prediction")
    while True:
        print("\nSymptoms you can choose from:")
        # Display symptoms in multiple columns for better readability
        symptom_list = [s.replace('_', ' ') for s in all_symptoms]
        col_width = max(len(s) for s in symptom_list) + 2
        num_cols = 3
        for i in range(0, len(symptom_list), num_cols):
            row = symptom_list[i:i+num_cols]
            print(''.join(s.ljust(col_width) for s in row))
        
        # Get symptom input from user
        symptom_input = input("\nEnter a symptom you're experiencing (or 'done' to finish): ").strip().lower()
        
        if symptom_input == 'done':
            break
        
        # Convert user input to match column name format
        formatted_input = symptom_input.replace(' ', '_')
        
        # Find closest matching symptom if exact match not found
        matching_symptom = None
        if formatted_input in all_symptoms:
            matching_symptom = formatted_input
        else:
            # Simple fuzzy matching by finding symptoms that contain the input
            possible_matches = [s for s in all_symptoms if formatted_input in s.lower()]
            if possible_matches:
                print(f"\nDid you mean one of these?")
                for i, match in enumerate(possible_matches, 1):
                    print(f"{i}. {match.replace('_', ' ')}")
                choice = input("Enter number or 'no': ")
                if choice.isdigit() and 1 <= int(choice) <= len(possible_matches):
                    matching_symptom = possible_matches[int(choice)-1]
        
        if matching_symptom:
            # Set the symptom value to 1 (present)
            user_symptoms[matching_symptom] = 1
            print(f"Recorded: {matching_symptom.replace('_', ' ')}")
        else:
            print("Symptom not recognized. Please try again.")
    
    # Create a DataFrame with user symptoms
    user_df = pd.DataFrame([user_symptoms])
    
    # Make prediction
    if sum(user_symptoms.values()) > 0:  # Check if at least one symptom is selected
        prediction = knn_optimized.predict(user_df)[0]
        probabilities = knn_optimized.predict_proba(user_df)
        prob_sorted_indices = probabilities[0].argsort()[::-1]  # Sort in descending order
        
        # Get the top 3 predictions with probabilities
        top_diseases = []
        classes = knn_optimized.classes_
        for i in prob_sorted_indices[:3]:
            if probabilities[0][i] > 0.05:  # Only include diseases with >5% probability
                top_diseases.append((classes[i], probabilities[0][i]*100))
        
        print("\n===== Prediction Results =====")
        print(f"Most likely diagnosis: {prediction}")
        
        print("\nTop predictions and confidence levels:")
        for disease, probability in top_diseases:
            print(f"- {disease}: {probability:.1f}%")
        
        print("\nNote: This is not a medical diagnosis. Please consult a healthcare professional.")
    else:
        print("\nNo symptoms selected. Cannot make a prediction.")

# Add option to make multiple predictions
while True:
    predict_disease()
    continue_prediction = input("\nWould you like to make another prediction? (yes/no): ").lower()
    if continue_prediction != 'yes':
        print("Thank you for using the Disease Prediction System!")
        break
