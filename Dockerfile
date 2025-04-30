FROM python:3.10

WORKDIR /code

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords wordnet

EXPOSE 7860

CMD ["python", "app.py"]
