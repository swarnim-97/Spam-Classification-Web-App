import requests
from flask import Flask, request, redirect, url_for, render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import model_from_json
import json
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# from keras.preprocessing.text import tokenizer_from_json

from preprocess import Preprocess

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

def text_preprocess(text):
    pre = Preprocess(text)
    print(pre)
    text = pre.clean_text(text)
    text = pre.clean_contractions(text,pre.contraction_mapping)
    text = pre.correct_spelling(text,pre.mispell_dict)
    text = pre.clean_special_chars(text,pre.punct,pre.punct_mapping)
    return text

def text_tokenizer(text):
    length = 40
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    text_pad = tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(text_pad,maxlen = length)
    return text_pad

def text_class(text_pad):
    with open('model_in_json.json','r') as f:
        model_json = json.load(f)

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model.h5')
    pre = loaded_model.predict(text_pad)
    label = (pre > 0.5).astype(int)
    return label[0][0]

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user = request.form['nm']
        text = text_preprocess(user)
        padded_text = text_tokenizer(text)
        label = text_class(padded_text)
        print(label)
        return render_template('predict.html', text=user, classes=label)
    else:
        user = request.args.get('nm')
        # return redirect(url_for('check', w = user))

if __name__ == "__main__":
    app.run(debug = True)
