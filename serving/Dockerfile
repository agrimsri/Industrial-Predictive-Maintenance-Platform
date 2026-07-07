FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

COPY serving /app/serving
COPY ml/models/registry /app/ml/models/registry

ENV PYTHONPATH=/app/serving
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
