FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

ENV PORT=8501
ENV BACKEND_URL=http://localhost:8000

EXPOSE ${PORT}

CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
