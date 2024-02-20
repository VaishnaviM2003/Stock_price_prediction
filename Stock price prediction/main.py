import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_scores = r2_score(y_test, y_pred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2_score: ", r2_scores)

def main():
    stocks = input("Enter the code of the stock: ")
    data = yf.download(stocks, "2010-01-01", "2023-01-16", auto_adjust=True)

    # Data Exploration
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.describe())

    # Data Visualization
    data.Close.plot(figsize=(10, 7), color='r')
    plt.ylabel("{} Prices".format(stocks))
    plt.title("{} Price Series".format(stocks))
    plt.show()

    sns.histplot(data["Close"])
    plt.show()
    sns.histplot(data["Open"])
    sns.histplot(data["High"])

    # Feature Engineering
    X = data.drop("Close", axis=1)
    y = data["Close"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    print("Linear Regression:")
    calculate_metrics(y_test, pred_lr)

    # Lasso Regression
    lasso = Lasso().fit(X_train, y_train)
    pred_lasso = lasso.predict(X_test)
    print("Lasso Regression:")
    calculate_metrics(y_test, pred_lasso)

    # Ridge Regression
    ridge = Ridge().fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    print("Ridge Regression:")
    calculate_metrics(y_test, pred_ridge)

    # Support Vector Regression with Grid Search
    svr = SVR()
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    print("Support Vector Regression with Grid Search:")
    calculate_metrics(y_test, grid.predict(X_test))

if __name__ == "__main__":
    main()
