FROM python:3.11-slim

# System deps for OpenCV headless and general runtime
RUN apt-get update && apt-get install -y --no-install-recommends     libglib2.0-0 libsm6 libxext6 libxrender1 libgl1     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Python deps (PyTorch/Ultralytics may be large)
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]
