FROM python:3.9-slim
WORKDIR /app
RUN apt update && \
    apt install -y --no-install-recommends \
    && pip install --upgrade pip \
    && pip install mlflow==2.15.0 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
EXPOSE 5000
#CMD ["mlflow", "server", "--host", "0.0.0.0"]
