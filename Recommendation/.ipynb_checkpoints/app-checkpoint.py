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

#Load dataset. It will help us find which product is recomended using their product id
metadata = pd.read_json('meta_All_Beauty.jsonl', lines=True)

product_id_to_asin = dict(zip(metadata.index, metadata['parent_asin']))

# There's a mapping between product IDs in the sparse matrix and ASINs (ASINs is unique product id in dataset)
# So we need to create a mapping from the sparse matrix indices to ASINs
# Here, product IDs correspond to the index of the `metadata` DataFrame
product_id_to_asin = dict(zip(metadata.index, metadata['parent_asin']))


# Extract list of users
list_users_result = collaborative_model.index.tolist()  # Adjust as per your model's structure

# Extract product IDs
coo_matrix = content_model.tocoo()  # Convert to COO format to access row and column indices
product_ids = list(set(map(int, coo_matrix.row)))  # Get unique product IDs from row indices
#print("Product IDs:", product_ids)  # Debug print statement

app = Flask(__name__)

@app.route('/generate-random-users', methods=['GET'])
def generate_random_users():
    random_users = random.sample(list_users_result, 5)
    return jsonify({'users': random_users})

@app.route('/generate-random-products', methods=['GET'])
def generate_random_products():
    # Generate 5 random product IDs each time
    products = random.sample(product_ids, 5)
    #print("Random Products:", products)  # Debug print statement
    return jsonify({'products': products})

def find_titles_from_ids(random_ids, id_to_asin_map, metadata):
    if len(random_ids) > 5:
        random_ids = random_ids[:5]

    titles = []
    for product_id in random_ids:
        asin = id_to_asin_map.get(product_id)
        if asin:
            title = metadata.loc[metadata['parent_asin'] == asin, 'title'].values
            titles.append(title[0] if len(title) > 0 else "Title not found")
        else:
            titles.append("ASIN not found for product ID")
    return titles

'''
@app.route('/get-titles-from-ids', methods=['POST'])
def get_titles_from_ids():
    data = request.json
    random_ids = data.get('product_ids', [])
    titles = find_titles_from_ids(random_ids, product_id_to_asin, metadata)
    return jsonify({'titles': titles})

'''

'''

@app.route('/get-titles-from-ids', methods=['POST'])
def get_titles_from_ids():
    try:
        # Log the incoming request data
        print("Request data received:", request.data)
        data = request.json
        print("Parsed JSON data:", data)
        
        # Extract product IDs
        random_ids = data.get('product_ids', [])
        print("Product IDs received:", random_ids)
        
        # Find titles from IDs
        titles = find_titles_from_ids(random_ids, product_id_to_asin, metadata)
        print("Titles found:", titles)
        
        # Return the result
        response = jsonify({'titles': titles})
        #print("Response data:", response.get_json())
        return response
    except Exception as e:
        # Log any errors
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    
    '''
    
    
@app.route('/get-titles-from-ids', methods=['POST'])
def get_titles_from_ids():
    try:
        # Log the incoming request data
        print("Request data received:", request.data)
        data = request.json
        print("Parsed JSON data:", data)
        
        # Extract product IDs
        random_ids = data.get('product_ids', [])
        print("Product IDs received:", random_ids)
        
        # Initialize an empty list to store results
        results = []
        
        for product_id in random_ids:
            title = find_titles_from_ids([product_id], product_id_to_asin, metadata)
            # Create a dictionary for each product_id and its corresponding title
            results.append({'product_id': product_id, 'title': title[0]})
        
        # Return the result
        response = jsonify({'results': results})
        print("Response data:", response.get_json())
        return response
    except Exception as e:
        # Log any errors
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500




# Function to find the product title using parent_asin
def find_title_by_parent_asin(parent_asin, metadata):
    result = metadata.loc[metadata['parent_asin'] == parent_asin, 'title']
    if not result.empty:
        return result.iloc[0]
    return "Title not found for the given parent_asin."

# Function to process a list of poduct IDs and return titles
def process_user_ids(product_ids, metadata):
    if len(product_ids) > 5:
        print("Error: The list should contain a maximum of 5 user IDs.")
        return []
    
    titles = []
    for product_id in product_ids:
        title_by_parent_asin = find_title_by_parent_asin(product_id, metadata)
        titles.append(title_by_parent_asin)
    
    return titles
'''
@app.route('/get-titles-from-asin', methods=['POST'])
def get_titles_from_asin():
    data = request.json
    parent_asins = data.get('parent_asins', [])
    if len(parent_asins) > 5:
        parent_asins = parent_asins[:5]
    titles = process_user_ids(parent_asins, metadata)
    return jsonify({'titles': titles})
'''

@app.route('/get-titles-from-asin', methods=['POST'])
def get_titles_from_asin():
    data = request.json
    parent_asins = data.get('parent_asins', [])
    if len(parent_asins) > 5:
        parent_asins = parent_asins[:5]
    
    # Initialize an empty list to store results
    results = []
    
    for parent_asin in parent_asins:
        title = find_title_by_parent_asin(parent_asin, metadata)
        # Create a dictionary for each parent_asin and its corresponding title
        results.append({'parent_asin': parent_asin, 'title': title})
    
    return jsonify({'results': results})


'''
@app.route('/get-titles-from-asin', methods=['GET'])
def get_titles_from_asin():
    data = request.json
    parent_asins = data.get('parent_asins', [])
    #parent_asins = ['B007IAE5WY', 'B00EEN2HCS', 'B07C533XCW', 'B00R1TAN7I', 'B08L5KN7X4']
    if len(parent_asins) > 5:
        parent_asins = parent_asins[:5]
    titles = process_user_ids(parent_asins, metadata)
    return jsonify({'titles': titles})
'''

def get_recommendations(method, user_id=None, item_id=None):
    if method == 'collaborative':
        if user_id is None:
            return {"error": "user_id is required for collaborative recommendations"}
        
        try:
            # Fetch user ratings from collaborative model
            user_ratings = collaborative_model.loc[user_id]
            recommendations = user_ratings.sort_values(ascending=False).head(5).index.tolist()
        except KeyError:
            return {"error": f"User ID '{user_id}' not found in the collaborative model"}

    elif method == 'content':
        if item_id is None:
            return {"error": "item_id is required for content-based recommendations"}
        
        try:
            # Fetch item similarities from content model
            item_similarities = content_model[item_id].toarray().flatten()
            recommendations = np.argsort(item_similarities)[-5:][::-1].tolist()
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
