import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('Model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        my_prediction = model.predict(input_data)

        return render_template('result.html', prediction=my_prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
