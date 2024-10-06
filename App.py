from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model when the app starts
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('FrontendFile.html')  

def index():
    # Load and prepare the data
    df=pd.read_csv('top_10_revenue.csv')
    top_products = df.groupby('StockCode')['TotalRevenue'].sum().reset_index()
    top_products = top_products.sort_values(by='TotalRevenue', ascending=False).head(10)

    # Convert to a list of dictionaries for easier use in the template
    top_products_list = top_products.to_dict(orient='records')

    return render_template('FrontendFile.html', products=top_products_list)
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert the data into a numpy array for prediction
    # Ensure that the data received is in the correct format for the model
    input_features = np.array(data['features'])

    # Use the model to make a prediction
    prediction = model.predict(input_features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
