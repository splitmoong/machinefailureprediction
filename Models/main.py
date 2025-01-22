from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

model = None
data = None


@app.route('/upload', methods=['POST'])
def upload():
    global data
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        data = pd.read_csv(file)
        return jsonify({'message': 'File uploaded successfully', 'columns': list(data.columns)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    global model, data

    if data is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    try:
        drop_columns = ['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        X = data.drop(columns=drop_columns)
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

        #train
        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return jsonify({'message': 'Model trained successfully', 'accuracy': accuracy, 'f1_score': f1})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({'error': 'Model not trained'}), 400

    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
