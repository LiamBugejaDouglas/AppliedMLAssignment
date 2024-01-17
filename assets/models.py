import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
import json

rnn_model = load_model('models/rnn_model.h5')
lstm_model = load_model('models/lstm_model.h5')
transformer_model = load_model('models/transformer_model.h5')

with open('tokenizers/rnn_tokenizer.json') as file:
    tokenizer_data = json.load(file)
    rnn_tokenizer = tokenizer_from_json(tokenizer_data)

with open('tokenizers/lstm_tokenizer.json') as file:
    tokenizer_data = json.load(file)
    lstm_tokenizer = tokenizer_from_json(tokenizer_data)

with open('tokenizers/transformer_tokenizer.json') as file:
    tokenizer_data = json.load(file)
    transformer_tokenizer = tokenizer_from_json(tokenizer_data)


def preprocess_and_predict(news_text, model, tokenizer, max_length=100):
    sequence = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded)
    return "Fake News" if prediction[0] > 0.5 else "Real News"


def startRNN(news_text):
    return preprocess_and_predict(news_text, rnn_model, rnn_tokenizer)


def startLSTM(news_text):
    # return preprocess_and_predict(news_text, lstm_model, lstm_tokenizer)
    return 'null'


def startTransformer(news_text):
    # return preprocess_and_predict(news_text, transformer_model, transformer_tokenizer)
    return 'null'
