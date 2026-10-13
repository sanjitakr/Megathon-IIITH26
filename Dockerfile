FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"
ENV PYTHONPATH=/app
# Build data and models at image build time (about a minute) so the container starts ready.
RUN make all PY=python N=30000
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501"]
