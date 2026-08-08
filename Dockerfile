FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgbm1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=10000"]
