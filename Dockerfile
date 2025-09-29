FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies including wget for connectivity check
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

ENV OLLAMA_MODEL=llama3

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]