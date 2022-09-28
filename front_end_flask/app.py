import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_xgb.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    print("int_feat = ", int_features)
    final_features = np.array([int_features])
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        status = "LOW"
    elif output == 1:
        status = "MEDIUM"
    else:
        status = "HIGH"

    return render_template(
        'index.html',
        prediction_text='Wine Quality should be {}'.format(status))
    # ada {{prediction_text di index.html}}


@app.route('/predict_api', methods=['GET', 'POST'])
def predict_api():
    try:
        int_features = [float(x) for x in request.form.values()]
        print("int_feat = ", int_features)
        final_features = np.array([int_features])
        prediction = model.predict(final_features)

        output = prediction[0]

        if output == 0:
            status = "LOW"
        elif output == 1:
            status = "MEDIUM"
        else:
            status = "HIGH"

        # return render_template('index.html', prediction_text='Wine Quality should be {}'.format(status))
        return jsonify({
            "Message": "Quality of wine LOW/MEDIUM/HIGH",
            "Status": 200,
            "Data": status
        })

    except Exception as exception_message:
        print("INI = ", type(str(exception_message)))
        return jsonify({
            "Message": "Error",
            "Status": 400,
            "Data": str(exception_message)
        })


if __name__ == "__main__":
    app.run(debug=True)