from transformers import BertForSequenceClassification, BertConfig
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from transformers import BertTokenizer
import json
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_keras_model(filename):
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at {path}")
    return load_model(path)


try:
    rnn_model = load_keras_model('AddRNN.h5')
    lstm_model = load_keras_model('AddLSTM.h5')
except FileNotFoundError as e:
    print(e)
    exit(1)


def load_tokenizer(path):
    with open(path) as file:
        tokenizer_data = json.load(file)
        return tokenizer_from_json(tokenizer_data)


path = os.path.join(dir_path, '../tokenizers/rnn_tokenizer.json')
rnn_tokenizer = load_tokenizer(path)
path = os.path.join(dir_path, '../tokenizers/lstm_tokenizer.json')
lstm_tokenizer = load_tokenizer(path)


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'best_model.pth')

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')))


def preprocess_and_predict_bert(news_text, model, tokenizer):
    max_length = 20

    tokenized_data = tokenizer.batch_encode_plus(
        news_text, max_length=max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')

    input_ids = tokenized_data['input_ids'].to(model.device)
    attention_mask = tokenized_data['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    prediction = torch.sigmoid(outputs.logits)
    return "Fake News" if prediction[0, 0] > 0.5 else "Real News"


def preprocess_and_predict_keras(news_text, model, tokenizer, max_length=150):
    sequence = tokenizer.texts_to_sequences([news_text])
    print("Sequence:", sequence)
    padded = pad_sequences(sequence, maxlen=max_length)
    print("Padded sequence:", padded)
    prediction = model.predict(padded)
    print("Raw prediction:", prediction)
    return "Fake News" if prediction[0] > 0.5 else "Real News"


def startRNN(news_text):
    return preprocess_and_predict_keras(news_text, rnn_model, rnn_tokenizer)


def startLSTM(news_text):
    return preprocess_and_predict_keras(news_text, lstm_model, lstm_tokenizer)


def startBERT(news_text):
    return preprocess_and_predict_bert(news_text, bert_model, bert_tokenizer)
