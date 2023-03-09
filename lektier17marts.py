from flask import Flask, request, render_template
from tensorflow import keras
import numpy as np

app = Flask(__name__)
model = keras.models.load_model('my_model_NAND.h5')

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def my_form_post():
    global input1, input2
    text1 = request.form['input1']
    text2 = request.form['input2']
    input1 = request.form['input1']
    input2 = request.form['input2']
    result = int(np.round(get_model_result(input1, input2), 0))
    loss = model.evaluate(x, y)
    accuracy = 100 - (loss * 100)
    return render_template('result.html', result=result, accuracy=accuracy,text1=text1,text2=text2)

def get_model_result(input1, input2):
    input1 = float(input1)
    input2 = float(input2)
    return model.predict([[input1, input2]])

if __name__=='__main__':
    app.run(debug=True)
