FROM python:3.11-slim

# opencv-python needs libGL/libglib at runtime even with no display attached.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY football_detection/ football_detection/
COPY scripts/ scripts/
COPY data/ data/
COPY config.yaml config.football_yolo.yaml ./

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
