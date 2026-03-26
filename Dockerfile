FROM python:3.11-slim

WORKDIR /app

# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код
COPY . .

# важно для логов
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py", "--config", "config.yaml"]