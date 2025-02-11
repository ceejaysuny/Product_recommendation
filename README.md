# Recommendation System for All_Beauty Products

This project builds a recommendation engine for products in the "All_Beauty" category using user behavior and product metadata.

## Objective
- Build a recommendation engine for products in the "All_Beauty" category.

## Deliverable
- A system recommending products based on user behavior and product metadata.

## Dataset
- Download the dataset from the following links:
  - [All_Beauty Reviews](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz)
  - [All_Beauty Metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz)

## Installation
Install the required libraries using the following command:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tensorflow flask
```

## Usage
1. **Data Preprocessing**:
   - Load the dataset.
   - Drop unnecessary columns.
   - Handle missing values.

2. **Model Training**:
   - Create a user-item matrix.
   - Apply Collaborative Filtering using Singular Value Decomposition (SVD).
   - Apply Content-Based Filtering using TF-IDF and cosine similarity.

3. **Evaluation**:
   - Evaluate the model using Precision@K, Recall@K, and NDCG@K.

4. **Model Deployment**:
   - Save the models using joblib.
   - Create a Flask API to serve recommendations.

## Running the Jupyter Notebook
Open the `Recommender_System.ipynb` file and run the cells sequentially to preprocess data, train models, and evaluate them.

## Running the Flask Application
1. Place the `index.html` file inside the `templates` folder.
2. Run the Flask app using the following command:
   ```bash
   python app.py
   ```
3. Access the app in your browser at `http://127.0.0.1:5000`.

## Testing the Recommendation Model
You can test the recommendation model using the provided Jupyter notebook cells or by interacting with the Flask web interface.

## Example Usage
- Generate random users and products.
- Get recommendations for a specific user or product.
- Fetch product titles using product IDs or parent ASINs.

## Contributing
Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License.
