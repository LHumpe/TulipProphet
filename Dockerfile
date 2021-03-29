FROM python:3.8.6

WORKDIR /app

ENV FLASK_APP=serving/app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY ./tulipprophet /app
# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["flask", "run"]