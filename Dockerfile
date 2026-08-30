FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/metadata.json ./data/metadata.json
COPY data/processed/ ./data/processed/
COPY data/chroma_db/ ./data/chroma_db/

WORKDIR /app/src

EXPOSE 8501

CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]