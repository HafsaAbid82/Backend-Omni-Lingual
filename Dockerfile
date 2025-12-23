FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    build-essential cmake g++ wget libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch + torchaudio explicitly
RUN pip install --no-cache-dir torch==2.2.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "7860"]
