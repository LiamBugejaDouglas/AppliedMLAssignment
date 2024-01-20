import gradio as gr
from assets.models.models import startRNN, startLSTM, startBERT


def predict_fake_news(news_title, article_text):
    combined_text = news_title + " " + article_text

    rnn_prediction, rnn_raw = startRNN(combined_text)
    lstm_prediction, lstm_raw = startLSTM(combined_text)
    bert_prediction, bert_raw = startBERT(combined_text)

    rnn_result = f"Prediction: {rnn_prediction}, Score: {rnn_raw:.4f}"
    lstm_result = f"Prediction: {lstm_prediction}, Score: {lstm_raw:.4f}"
    bert_result = f"Prediction: {bert_prediction}, Score: {bert_raw:.4f}"

    return rnn_result, lstm_result, bert_result


demo = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.Textbox(label="News Title", lines=2,
                   placeholder="Enter News Title Here"),
        gr.Textbox(label="Article Text", lines=10,
                   placeholder="Enter Full Article Text Here")
    ],
    outputs=[
        gr.Textbox(label="RNN Model Prediction & Score"),
        gr.Textbox(label="LSTM Model Prediction & Score"),
        gr.Textbox(label="BERT Model Prediction & Score")
    ],
    title="Fake News Predictor",
    description="Enter the title and the full text of the news article to see predictions from RNN, LSTM, and BERT models."
)

demo.launch(share=True)
