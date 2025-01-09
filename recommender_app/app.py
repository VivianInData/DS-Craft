from flask import Flask, request, jsonify, render_template_string
from model import RecommenderModel
import numpy as np
import json
import pandas as pd

app = Flask(__name__)
recommender = RecommenderModel()

# Load all the necessary data
X_item_features = np.load('X_item_features.npy')
with open('item_ids.json', 'r') as f:
    item_ids = json.load(f)
    
# Load the training interactions data
train_inter = pd.read_csv('train_inter.csv')  # Adjust path if needed

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

# Keep your existing endpoints...

if __name__ == '__main__':
    recommender.prepare_data(X_item_features, item_ids)
    recommender.train()
    app.run(debug=True, port=5001)  # Change to a different port number