from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import random
import gc


gc.collect()
# Load saved models
collaborative_model = joblib.load('collaborative_model.pkl')  # Loaded as a pandas DataFrame
content_model = joblib.load('content_similarity_model.pkl')   # Assuming a sparse matrix or ndarray

#Extract list of users
list_users_result = collaborative_model.index.tolist()  # Adjust as per your model's structure

# Extract product IDs
# content_model is a sparse matrix, so we use the `.tocoo()` method
coo_matrix = content_model.tocoo()  # Convert to COO format to access row and column indices
#product_ids = list(set(coo_matrix.row))  # Get unique product IDs from row indices
 # Get unique product IDs from row indices and Convert to Python `int` and get unique product IDs
product_ids = list(set(map(int, coo_matrix.row))) 

app = Flask(__name__)

@app.route('/generate-random-users', methods=['GET'])
def generate_random_users():
    random_users = random.sample(list_users_result, 5)    
    return jsonify({'users': random_users})

@app.route('/generate-random-products', methods=['GET'])
def generate_random_products():
    # Generate 5 random product IDs each time
    products = random.sample(product_ids, 5)
    # Convert int32 to Python's native int type
    #products = [int(product) for product in products]
    return jsonify({'products:': products})

def get_recommendations(method, user_id=None, item_id=None):
    if method == 'collaborative':
        if user_id is None:
            return {"error": "user_id is required for collaborative recommendations"}
        
        try:
            # Fetch user ratings from collaborative model
            user_ratings = collaborative_model.loc[user_id]
            recommendations = user_ratings.sort_values(ascending=False).head(10).index.tolist()
        except KeyError:
            return {"error": f"User ID '{user_id}' not found in the collaborative model"}

    elif method == 'content':
        if item_id is None:
            return {"error": "item_id is required for content-based recommendations"}
        
        try:
            # Fetch item similarities from content model
            item_similarities = content_model[item_id].toarray().flatten()
            recommendations = np.argsort(item_similarities)[-10:][::-1].tolist()
        except IndexError:
            return {"error": f"Item ID {item_id} not found in the content model"}
    else:
        return {"error": "Invalid method"}
    
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')  # User ID is a string
    item_id = request.args.get('item_id', type=int)
    method = request.args.get('method', 'collaborative')  # Default to collaborative

    recommendations = get_recommendations(method, user_id, item_id)
    
    if isinstance(recommendations, dict) and "error" in recommendations:
        return jsonify(recommendations), 400

    return jsonify({"recommendations": recommendations})
    
          
if __name__ == '__main__':
    import os
    os.environ["FLASK_DEBUG"] = "development"
    app.run(debug=True, use_reloader=False)
    
    '''
     if __name__ == '__main__':
    import os
    os.environ["FLASK_ENV"] = "development"
    app.run(debug=True, use_reloader=False)
    '''
    