import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow import keras
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

warnings.filterwarnings('ignore')


def load_and_label_data(file_path, label_value, drop_columns=None):
    df = pd.read_csv(file_path)
    df['label'] = label_value
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df


def preprocess_text(text, stopwords_set, stemmer):
    text = text.lower()
    text = re.sub('\d+', '', text)  # Remove numbers
    words = text.split()
    words = [
        word for word in words if word not in stopwords_set and word not in string.punctuation]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


# Load and preprocess datasets
df1 = load_and_label_data("Dataset/Fake.csv", 1, ['subject', 'date'])
print("df1 head:", df1.head())
df2 = load_and_label_data("Dataset/True.csv", 0, ['subject', 'date'])
print("df2 head:", df2.head())
df3 = load_and_label_data("Dataset/news.csv", None)
print("df3 head:", df3.head())
df3['label'] = df3['label'].replace(
    {'FAKE': 1, 'REAL': 0}).drop(columns=df3.columns[0])
df4 = load_and_label_data("Dataset/WELFake_Dataset.csv", None)
df4['label'] = df4['label'].replace({1: 0, 0: 1}).drop(columns=df4.columns[0])

# Combine and clean data
df = pd.concat([df1, df2, df3, df4], ignore_index=True).dropna().drop_duplicates(
    subset=['title'], keep='first').sample(frac=1, random_state=42)
df['final'] = df['title'] + ' ' + df['text']  # Combine title and text

# Preprocess text data
stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()
df['final'] = df['final'].apply(
    lambda x: preprocess_text(x, stopwords_set, stemmer))

# Tokenization and padding
vocab_size = 10000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(df['final'])
sequences = tokenizer.texts_to_sequences(df['final'])
print("Size of sequences:", len(sequences))
padded = pad_sequences(sequences, maxlen=max_length,
                       padding='post', truncating='post')
print("Size of padded data:", padded.shape)

# Splitting data
labels = df['label'].values
print("Labels:", labels[:10])  # Print first 10 labels
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2)
print("Training and testing data sizes:", X_train.shape, X_test.shape)

# Model building
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('rnn_model.h5')
