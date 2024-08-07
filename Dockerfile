FROM python
RUN apt update
RUN pip install --upgrade pip
RUN python3 --version
RUN mkdir /app
WORKDIR /app
ENV ENV=local
ENV MLFLOW_SERVER=http://127.0.0.1:5000
ENV MLFLOW_REGISTRY_NAME=purchase_predict
COPY requirements.txt /app/
COPY app.py /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["flask", "run", "--host", "0.0.0.0"]
#CMD ["python3", "/app/app.py"]
# FIXME environment variables can go into a list_env