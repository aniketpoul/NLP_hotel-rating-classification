# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:47:19 2020

@author: hp
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import flask

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)