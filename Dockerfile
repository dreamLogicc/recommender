FROM python

ADD ./main.py .
ADD ./recommender.py .
ADD ./data_for_db.csv .
ADD model.keras .

COPY ./req.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install protobuf==3.20.*
RUN pip install --no-cache-dir -r req.txt

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
