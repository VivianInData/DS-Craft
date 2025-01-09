from flask import Flask, request, jsonify, render_template_string
from model import RecommenderModel
import numpy as np
import json
import pandas as pd
import os
import gdown

app = Flask(__name__)
recommender = RecommenderModel()

app.config.update(
    SERVER_NAME=None,  # Clear any existing server name
    APPLICATION_ROOT='/'
)


# Google Drive file ID and output file
file_id = "1A2B3C4D5E6F7G8H9"
output_file = "X_item_features.npy"

# Check if the file exists, if not download it
if not os.path.exists(output_file):
    print(f"{output_file} not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
else:
    print(f"{output_file} already exists.")

# Load the .npy file after ensuring it exists
try:
    X_item_features = np.load(output_file)
    print(f"{output_file} loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load {output_file}: {e}")

# Load the necessary data
try:
    with open('item_ids.json', 'r') as f:
        item_ids = json.load(f)
    print("item_ids.json loaded successfully.")
except FileNotFoundError:
    raise RuntimeError("item_ids.json not found in the directory.")

try:
    train_inter = pd.read_csv('train_inter.csv')  # Adjust path if needed
    print("train_inter.csv loaded successfully.")
except FileNotFoundError:
    raise RuntimeError("train_inter.csv not found in the directory.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Product Recommender System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        input, button { padding: 10px; margin: 10px 0; }
        .recommendations { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Product Recommender System</h1>
        
        <form method="POST" action="/web_recommend">
            <div>
                <label for="user_id">Enter User ID:</label><br>
                <input type="text" id="user_id" name="user_id" required 
                       placeholder="e.g., AFUSEI15AFBC5VIFXHC3Z7UYAGKQ">
            </div>
            <button type="submit">Get Recommendations</button>
        </form>

        {% if interacted_items %}
        <div class="recommendations">
            <h2>User History for: {{ user_id }}</h2>
            <h3>Previously interacted items:</h3>
            <ul>
            {% for item in interacted_items %}
                <li>{{ item }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if recommendations %}
        <div class="recommendations">
            <h3>Recommended items:</h3>
            <ul>
            {% for item in recommendations %}
                <li>{{ item }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if error %}
        <div class="error" style="color: red;">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/web_recommend', methods=['POST'])
def web_recommend():
    user_id = request.form.get('user_id')
    
    try:
        # Get user's interacted items from train_inter
        user_items = train_inter[train_inter['user_id'] == user_id]['item_id'].tolist()
        
        if not user_items:
            return render_template_string(
                HTML_TEMPLATE, 
                error=f"No interaction history found for user: {user_id}"
            )
        
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            user_items=user_items
        )
        
        return render_template_string(
            HTML_TEMPLATE, 
            user_id=user_id,
            interacted_items=user_items,
            recommendations=recommendations
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            error=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    # Prepare the recommender system
    print("Preparing recommender system...")
    recommender.prepare_data(X_item_features, item_ids)
    recommender.train()
    print("Recommender system ready. Starting the Flask server...")
    
    print(f"Attempting to bind to 10.128.0.2:5001...")  # Add debug print
    app.run(
        host='10.128.0.2',  # Your specific VM IP
        port=5001,
        debug=True,
        use_reloader=True,
        threaded=True
    )
