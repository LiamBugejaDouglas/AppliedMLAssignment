from transformers import BertForSequenceClassification, BertConfig, AutoModel, BertTokenizer
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
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
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')))

# Load the best model statemodel.load_state_dict(torch.load('best_model.pth'))
bert_model.eval()

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)


def preprocess_and_predict_bert(news_text, bert_model, bert_tokenizer, device):
    max_length = 20

    # Tokenize the input text
    tokenized_text = bert_tokenizer.encode_plus(
        news_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move data to the device (GPU or CPU)
    input_ids = tokenized_text['input_ids'].to(device)
    attention_mask = tokenized_text['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(
        logits, dim=1).squeeze().cpu().numpy()

    # Return both the predicted label and the raw probabilities
    predicted_label = "Real News" if torch.argmax(
        logits, dim=1).item() == 0 else "Fake News"
    # Assuming index 1 is for "Fake News"
    raw_prediction_value = probabilities[1]
    return predicted_label, raw_prediction_value


def preprocess_and_predict_keras(news_text, model, tokenizer, max_length=150):
    sequence = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded)
    raw_prediction_probability = prediction[0][0]
    predicted_label = "Fake News" if raw_prediction_probability > 0.5 else "Real News"

    return predicted_label, raw_prediction_probability


def startRNN(news_text):
    return preprocess_and_predict_keras(news_text, rnn_model, rnn_tokenizer)


def startLSTM(news_text):
    return preprocess_and_predict_keras(news_text, lstm_model, lstm_tokenizer)


def startBERT(news_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    return preprocess_and_predict_bert(news_text, bert_model, bert_tokenizer, device)
