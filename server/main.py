from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

import sys

sys.path.append("..")
from src.DataAnalyzer import DataAnalyzer

"""
Initialize server
"""
app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="This is an airline sentiment analysis api for analyzing airline tweets",
    version="0.0.1",
    contact={
        "name": "Lawrence Li",
        "url": "https://github.com/lawrencexli",
        "email": "lawrencexli2@gmail.com",
    },
    docs_url="/docs"
)

data_analyzer = DataAnalyzer()
data_analyzer.load_model()


class Message(BaseModel):
    tweet_msg: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Airline Sentiment Analysis API</title>
        </head>
        <body>
            <h1>Welcome to my airline sentiment analysis API</h1>
            <h1>In this demo I will use Postman to make API calls</h1>
        </body>
    </html>
    """


@app.post("/analysis/")
def analyze_text(msg: Message):
    sentiment_result = data_analyzer.predict(msg.tweet_msg)

    return {
        "msgReceived": msg.tweet_msg,
        "result": sentiment_result,
    }
