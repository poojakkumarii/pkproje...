import pickle 
from flask import Flask, render_template, request


app= Flask(__name__)
loadedModel= pickle.load(open('Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    SepalLengthCm = request.form['SepalLengthCm']
    SepalWidthCm = request.form['SepalWidthCm']
    PetalLengthCm = request.form['PetalLengthCm']
    PetalWidthCm = request.form['PetalWidthCm']

    prediction = loadedModel.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])[0]

    if prediction == 0:
        prediction = "Setosa"
    elif prediction == 1:
        prediction = "Versicolor"
    else:
        prediction = "Virginica"

    return render_template('iris.html',output=prediction)

if __name__=='__main__':
    app.run(debug=True)
