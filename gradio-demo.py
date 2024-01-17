import gradio as gr
from assets.models import startRNN, startLSTM, startTransformer


def predict_fake_news(news_text):
    rnn_prediction = startRNN(news_text)
    lstm_prediction = startLSTM(news_text)
    transformer_prediction = startTransformer(news_text)

    return rnn_prediction, lstm_prediction, transformer_prediction


demo = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.Textbox(label="News Title", lines=2),
    ],
    outputs=[
        gr.Textbox(label="RNN Model Prediction", lines=1),
        gr.Textbox(label="LSTM Model Prediction", lines=1),
        gr.Textbox(label="Transformer Model Prediction", lines=1),
    ],
    title="Fake News Predictor",
    description="Enter a piece of news and see predictions from RNN, LSTM, and Transformer models."
)

demo.launch(share=True)
