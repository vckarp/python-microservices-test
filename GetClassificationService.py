from flask import Flask
import pandas as pd
import pickle

import requests

app = Flask("GetClassification")


@app.route("/classify", methods=['GET'])
def classify():
    clf = pickle.loads(requests.get('http://127.0.0.1:5000/get_clf').content)
    data = pd.read_csv("sample_data.csv", header=None)
    pred = clf.predict(data)
    return "\n\n".join(pred)


if __name__ == '__main__':
    app.debug = True
    app.run(port=5001)
