from flask import Flask, request, jsonify
import pandas as pd
import pickle
import models

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
        return jsonify({'error': 'No Dataset uploaded'}), 400

    try:
        model, metrics = models.random_forest(data)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return jsonify({'message': 'Model trained successfully', 'accuracy': metrics.get('accuracy'),
                        'f1_score': metrics.get('f1')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Model not trained or file not found'}), 400

    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = int(prediction)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
