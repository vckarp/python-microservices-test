from flask import Flask, request, make_response
import json
import pandas as pd
import pickle
from sklearn import svm
import os


def string_to_int(data):
    """A function to transform factorial data into python readable format"""

    def get_dict(column):
        d1 = {val: num + 1 for num, val in enumerate(sorted(set(column)))}
        return d1

    for val in data.keys():
        d = get_dict(data[val])
        for i, num in enumerate(data[val]):
            data[val][i] = d[num]
    return data


app = Flask("train_model")


@app.route("/train", methods=['GET'])
def train_model():
    data = pd.read_csv('carData.txt', header=None)
    target = data.iloc[:, -1]
    data = data.iloc[:, :-1]
    data = string_to_int(data)
    clf2 = svm.SVC()
    # print(clf2.fit(X=data.values, y=target.values).score(data, target))
    clf2.fit(X=data.values, y=target.values)
    with open('model.pickle', 'wb') as f:
        pickle.dump(clf2, f)
    return str(clf2.score(data, target))


# @app.route("/train", methods=["POST"])
# def append_and_retrain():
#     with open("")


@app.route("/get_clf", methods=['GET'])
def provide_clf():
    if os.path.isfile('model.pickle'):
        with open('model.pickle', 'rb') as f:
            response = pickle.load(f)
        return pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return "No classifier found. Perhaps it should be recreated."


if __name__ == '__main__':
    app.debug = True
    app.run()
