FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the RAG index at image build time so the app starts ready-to-serve
RUN python -m app.rag.build_index

# Hugging Face Spaces expects the app on port 7860; most other platforms
# inject $PORT. Default to 7860, but honor $PORT if the platform sets it.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
