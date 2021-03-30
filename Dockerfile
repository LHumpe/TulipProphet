FROM python:3.8.6

ENV FLASK_APP=./code/model_api.py
ENV FLASK_RUN_HOST=0.0.0.0

ENV MODEL_DIR=trained_models/crypto_model

# Install Python dependencies
COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["flask", "run"]