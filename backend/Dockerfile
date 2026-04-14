FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

RUN mkdir -p data outputs

ENV PORT=8000

EXPOSE ${PORT}

CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}
