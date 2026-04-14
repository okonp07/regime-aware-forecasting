FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY start.sh .

RUN mkdir -p data outputs && chmod +x start.sh

ENV PORT=8501

EXPOSE ${PORT}

CMD ["./start.sh"]
