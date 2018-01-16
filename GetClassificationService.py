from flask import Flask
import pandas as pd
import pickle
import json
import requests

app = Flask(__name__)


@app.route("/classify", methods=['GET'])
def classify():
    clf = pickle.loads(requests.get('http://127.0.0.1:5000/get_clf').content)
    data = pd.read_csv("sample_data.csv", header=None)
    data = data.iloc[:, :-1]
    pred = clf.predict(data)
    return "\n\n".join(pred)


@app.route("/post_new")
def post_new_data():
    """A function to check the functionality of the append_db function from the ModelTrainService.py"""
    with open('sample_data.json') as f:
        data = json.load(f)
    request = requests.post('http://127.0.0.1:5000/append_db', json=data)
    return request.content


if __name__ == '__main__':
    app.debug = True
    app.run(port=5001)
