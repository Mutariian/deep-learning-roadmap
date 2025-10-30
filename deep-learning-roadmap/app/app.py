from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/simple_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'])
    preds = model.predict(data).tolist()
    return jsonify({'predictions': preds})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
