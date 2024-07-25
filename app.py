from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# Load the trained model, scaler, and PCA
rfc_model = joblib.load('rfc_model.pkl')
xgbc_model = joblib.load('xgbc_model.pkl')


# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data3 = pd.read_csv('data2.csv')
    new_data = request.json  # Get the data from the request
    new_data = pd.read_json(new_data, typ='series')
    data3.loc[len(data3)] = new_data
    data3.drop(columns=['_id','Unnamed: 84','f22'], inplace=True)
    for col in data3.columns:
        data3.fillna({col:data3[col].mode()[0]}, inplace=True)
    le = LabelEncoder()
    for i in data3.columns:
        if data3[i].dtype == 'object':
            data3[i] = le.fit_transform(data3[i])
    group1 = ['f5', 'f6', 'f7', 'f8', 'f9','f10','f11']
    group2 = ['f17', 'f18', 'f19', 'f20']
    group3 = ['f27', 'f28', 'f29', 'f30', 'f31','f32','f33']
    group4 = ['f38', 'f39', 'f40', 'f41', 'f42','f43','f44']
    group5 = ['f49', 'f50', 'f51', 'f52', 'f53','f54','f55']
    group6 = ['f61', 'f62', 'f63', 'f64', 'f65','f66','f67']
    group7 = ['f74', 'f75', 'f76', 'f77', 'f78','f79','f80']

    groups = [group1, group2, group3, group4, group5, group6, group7]
    # convert new_data from json to dictionary
    def apply_pca_to_group(data, group, n_components=1):
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(data[group])
        pca_column = f'PCA_{group[0]}'
        data[pca_column] = pca_components
        data.drop(columns=group,inplace=True)
        return data

    # Apply PCA to each group and update the dataframe
    for group in groups:
        data3_pca = apply_pca_to_group(data3, group, n_components=1)
    data_test = data3_pca.iloc[-1].drop('label').values.reshape(1, -1)
    prediction_rfc = rfc_model.predict(data_test)  # Make prediction using Random Forest
    prediction_xgbc = xgbc_model.predict(data_test)  # Make prediction using XGBoost
    return jsonify({'Random forest Prediction': prediction_rfc.tolist(), 'XGBoost Prediction': prediction_xgbc.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
