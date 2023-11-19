FROM python:3.10

COPY app.py /app/
COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
