# Airline Sentiment Classification with API

Developed by Lawrence Xiaohua Li

Email: lawrence.lixiaohua@gmail.com

## Video demo

Check out the demo.mp4 in the main repository or click the link below: 

https://www.youtube.com/watch?v=Ryflhwq2QzY

## Library Used

`nltk`, `pickle`, `fastapi`, `pydantic`

## Run API

Go to `server` folder and run command `uvicorn main:app`
Server default starts on `127.0.0.1:8000` or `localhost:8000` on `GET`

For `POST`, go to `127.0.0.1:8000/analysis` or `localhost:8000/analysis`

## Run train or predict test

Go to `src` folder and run `train.py` or `predict_test.py`

## Swagger docs

Go to `127.0.0.1:8000/docs` or `localhost:8000/docs`
