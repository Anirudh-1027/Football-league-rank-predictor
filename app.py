from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('data.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data1 = request.form['gf']
        data2 = request.form['ga']
        data = np.array(int(data1)-int(data2)).reshape(-1, 1)
        pred = model.predict(data)
        import math
        if(pred<=1):
            pred=1
        else:
            pred=math.ceil(pred)
        return render_template('predict.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















