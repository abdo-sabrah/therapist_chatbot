#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, request, jsonify
import json
import random
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# In[5]:




app = Flask(__name__)
@app.route('/')
def home():
    return 'Welcome to my Flask app!'

# Load the JSON dataset
with open('unpacked.json', 'r') as f:
    dataset = json.load(f)

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to predict the intent of the user input
def predict_intent(user_input):
    # Use the loaded model to predict the intent
    predicted_intent = model.predict([user_input])[0]
    return predicted_intent

# Function to select a random response from the dataset based on the intent
def get_random_response(intent):
    # Filter the dataset for the intent and select a random response
    intent_responses = [item['Response'] for item in dataset if item['Knowledge'] == intent]
    if intent_responses:
        return random.choice(intent_responses)
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

# Route for handling chatbot requests
@app.route('/chatbot', methods=['POST'])
def chatbot():
    if request.json is None:
        
        return jsonify({'error': 'No JSON data received'}), 400
    # Get the user input from the request
    user_input = request.json['user_input']
    
    # Predict the intent of the user input
    predicted_intent = predict_intent(user_input)
    
    # Get a random response based on the predicted intent
    random_response = get_random_response(predicted_intent)
    
    # Return the response as JSON
    return jsonify({'bot_response': random_response})

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=os.getenv('PORT', 8000))


# In[ ]:




