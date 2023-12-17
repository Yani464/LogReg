import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score,accuracy_score
import streamlit as st


# train[['CCAvg','Income']] = scaler.fit_transform(train[['CCAvg','Income']])



st.write ("""
Приложение для вычисления логарифмической регрессии
          """)

st.sidebar.header('Добавление Datset')

file = st.sidebar.file_uploader('Твой Dataset',type=['csv'])



test = pd.read_csv(file)

scaler = StandardScaler()
test[['CCAvg','Income']] = scaler.fit_transform(test[['CCAvg','Income']])

class LogisticRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None 
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features)
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.theta)
            y_pred = self.sigmoid(linear_model)
            
            gradient = np.dot(X.T, (y_pred - y)) / num_samples
            
            self.theta -= self.learning_rate * gradient
    
    def predict(self, X):
        linear_model = np.dot(X, self.theta)
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class
    
model = LogisticRegressionGradientDescent()

model.fit(test[['CCAvg','Income']],test['Personal.Loan'])

accuracy = accuracy_score(test['Personal.Loan'], model.predict(test[['CCAvg','Income']]))
st.write({'Точность нашей модели' : accuracy.round(2)})