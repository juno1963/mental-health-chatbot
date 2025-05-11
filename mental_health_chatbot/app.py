from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# File path to store user credentials
USERS_FILE = 'users.json'

# Load or initialize users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Load the model
model_name = "finetuned_chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Routes
@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/signup', methods=['POST'])
def signup():
    users = load_users()
    username = request.form['username']
    password = request.form['password']
    if username in users:
        return render_template('landing.html', error='Username already exists.', show_form=True)
    users[username] = password
    save_users(users)
    session['username'] = username
    return redirect(url_for('chat'))

@app.route('/login', methods=['POST'])
def login():
    users = load_users()
    username = request.form['username']
    password = request.form['password']
    if users.get(username) == password:
        session['username'] = username
        return redirect(url_for('chat'))
    return render_template('landing.html', error='Invalid credentials.', show_form=True)

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect('/')
    return render_template('chat.html', username=session['username'])

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message', '')
    try:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean bot label if present
        response = full_output.replace(user_input, '').strip()
        for prefix in ['Bot:', 'Assistant:', 'AI:', 'Response:', 'Chatbot:']:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return jsonify({'response': response})
    except Exception as e:
        print("Error generating response:", e)
        return jsonify({'response': "Oops, something went wrong. Please try again."})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

#if __name__ == '__main__':
   # app.run(debug=True)
if __name__ == "__main__":
    app.run()

