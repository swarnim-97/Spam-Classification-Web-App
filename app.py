import requests
import tensorflow as tf
from flask import Flask, request, redirect, url_for, render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from preprocess import Preprocess
#Load tokenizer
tokenizer = Preprocess.tokenizer_load()
loaded_model = Preprocess.loadModel()
loaded_model._make_predict_function()
graph = tf.get_default_graph()

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

def text_preprocess(text):
    pre = Preprocess(text)
    text = pre.clean_text(text)
    text = pre.clean_contractions(text,pre.contraction_mapping)
    text = pre.correct_spelling(text,pre.mispell_dict)
    text = pre.clean_special_chars(text,pre.punct,pre.punct_mapping)
    return text

def text_tokenizer(text,tokenizer):
    length = 40
    text_pad = tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(text_pad,maxlen = length)
    return text_pad

def text_class(text_pad,loaded_model):
    # global loaded_model
    global graph
    with graph.as_default():
        pre = loaded_model.predict(text_pad)
    # pre = loaded_model.predict(text_pad)
    label = (pre > 0.5).astype(int)
    return label[0][0]

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user = request.form['nm']
        text = text_preprocess(user)
        padded_text = text_tokenizer(text,tokenizer)
        label = text_class(padded_text,loaded_model)
        return render_template('predict.html', text=user, classes=label)
    else:
        user = request.args.get('nm')
        # return redirect(url_for('check', w = user))

if __name__ == "__main__":
    app.run(debug = True)
