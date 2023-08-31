FROM python

ADD ./main.py .
ADD ./recommender.py .
ADD ./data_with_array_emb.csv .
ADD ./clf.keras .

COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install protobuf==3.20.*
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
