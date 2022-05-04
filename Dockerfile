#FROM python:3.7-alpine
FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY . /app

RUN pip install -r requirements.txt
RUN python -m spacy download en
#EXPOSE process.env.PORT || 8000
EXPOSE 8501
# ENTRYPOINT ["streamlit", "run"]
# CMD ["app.py"]

# CMD streamlit run app.py
CMD streamlit run --server.port $PORT app.py