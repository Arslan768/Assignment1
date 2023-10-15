from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained Iris model
model = joblib.load('Model/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_width']
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        return jsonify({' PREDICTION : ': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
