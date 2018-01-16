from flask import Flask, request
import pandas as pd
import pickle
from sklearn import svm
import os
from flask_sqlalchemy import SQLAlchemy
from config import Config


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


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feature1 = db.Column(db.Integer)
    feature2 = db.Column(db.Integer)
    feature3 = db.Column(db.Integer)
    feature4 = db.Column(db.Integer)
    feature5 = db.Column(db.Integer)
    feature6 = db.Column(db.Integer)
    target = db.Column(db.String(20))

    def __repr__(self):
        return 'Row_No: {}  Features: {} {} {} {} {} {} Target: {} ||||||| '.format(self.id, self.feature1, self.feature2, self.feature3,
                                                                        self.feature4, self.feature5, self.feature6, self.target)


@app.route("/train", methods=['GET'])
def train_model():
    data = pd.read_sql_query('select * from Dataset', Dataset.query.session.bind)
    target = data.iloc[:, -1]
    data = data.iloc[:, 1:-1]
    clf2 = svm.SVC()
    # print(clf2.fit(X=data.values, y=target.values).score(data, target))
    clf2.fit(X=data.values, y=target.values)
    with open('model.pickle', 'wb') as f:
        pickle.dump(clf2, f)
    return str(clf2.score(data, target))


@app.route("/append_db", methods=["POST"])
def append_db():
    try:
        new_data = request.get_json()
    except Exception:
        return 'Error'
    if isinstance(new_data, list):
        for row in new_data:
            keys = list(row.keys())
            db.session.add(Dataset(feature1=row[keys[0]], feature2=row[keys[1]], feature3=row[keys[2]], feature4=row[keys[3]],
                                   feature5=row[keys[4]], feature6=row[keys[5]], target=row[keys[6]]))
        db.session.commit()
    if isinstance(new_data, dict):
        keys = list(new_data.keys())
        db.session.add(Dataset(feature1=new_data[keys[0]], feature2=new_data[keys[1]], feature3=new_data[keys[2]],
                               feature4=new_data[keys[3]], feature5=new_data[keys[4]], feature6=new_data[keys[5]],
                               target=new_data[keys[6]]))
        db.session.commit()
    return "Rows added to the table, check localhost:5000/show_data to see if the new rows are appended to the db"


@app.route("/get_clf", methods=['GET'])
def provide_clf():
    if os.path.isfile('model.pickle'):
        with open('model.pickle', 'rb') as f:
            response = pickle.load(f)
        return pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return "No classifier found. Perhaps it should be recreated."


@app.route("/export_data")
def export_data():
    data = pd.read_csv('carData.txt', header=None)
    target = data.iloc[:, -1]
    data = data.iloc[:, :-1]
    data = string_to_int(data)
    data = pd.concat([data.reset_index(), target], axis=1)
    for row in data.values:
        db.session.add(Dataset(feature1=row[1], feature2=row[2], feature3=row[3], feature4=row[4],
                               feature5=row[5], feature6=row[6], target=row[7]))
    db.session.commit()


@app.route("/show_data")
def show_data():
    data = Dataset.query.all()
    data = [row.__str__() for row in data]
    data = "".join(data)
    return data


if __name__ == '__main__':
    app.debug = True
    app.run()
