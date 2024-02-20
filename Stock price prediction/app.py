from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    stock_code = request.args.get('stockCode')

    # Download stock data
    data = yf.download(stock_code, "2010-01-01", "2023-01-16", auto_adjust=True)

    # Extract features
    X = data.drop("Close", axis=1)
    
    # Make a simple linear regression prediction
    lr = LinearRegression()
    lr.fit(X, data["Close"])
    predicted_price = lr.predict(X.tail(1))

    return jsonify({'predictedPrice': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)
