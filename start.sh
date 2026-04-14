#!/bin/bash
# Start backend on internal port, then frontend on the exposed port
echo "Starting backend on port 8000..."
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend ready."
        break
    fi
    sleep 1
done

echo "Starting frontend on port ${PORT:-8501}..."
export BACKEND_URL=http://localhost:8000
cd frontend
exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
