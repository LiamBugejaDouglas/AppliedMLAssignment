import gradio as gr
from assets.models.models import startRNN, startLSTM, startBERT


def predict_fake_news(news_text):
    rnn_prediction = startRNN(news_text)
    lstm_prediction = startLSTM(news_text)
    bert_prediction = startBERT(news_text)

    return rnn_prediction, lstm_prediction, bert_prediction


demo = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.Textbox(label="News Title", lines=2),
    ],
    outputs=[
        gr.Textbox(label="RNN Model Prediction", lines=1),
        gr.Textbox(label="LSTM Model Prediction", lines=1),
        gr.Textbox(label="BERT Model Prediction", lines=1),
    ],
    title="Fake News Predictor",
    description="Enter a piece of news and see predictions from RNN, LSTM, and BERT models."
)

demo.launch(share=True)
