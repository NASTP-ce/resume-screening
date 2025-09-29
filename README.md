# Resume Screening Assistant

AI-powered resume screening using LangGraph multi-agent system and Ollama LLMs.

---

## Prerequisites

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Ollama** - [Install Ollama](https://ollama.com/download)

---

## Installation

### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows/Mac:** Download from [ollama.com](https://ollama.com/download)

### 2. Start Ollama and Download Model

```bash
# Start Ollama service
ollama serve &

# Verify it's running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Download a model
ollama pull llama2:7b
# Or other models: llama3.2, mistral, qwen2.5:7b
```

---

## Running the Application

### Option 1: Run Locally (Without Docker)

**Setup:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Run:**
```bash
# Default model
streamlit run app.py

# Specific model
OLLAMA_MODEL="llama2:7b" streamlit run app.py
OLLAMA_MODEL="mistral" streamlit run app.py
```

Access at: **http://localhost:8501**

---

### Option 2: Run with Docker (Recommended)

**Build:**
```bash
docker-compose up --build
```

**Or using docker run:**

Linux:
```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -p 5000:5000 \
  -e OLLAMA_MODEL="llama2:7b" \
  resume-screening
```

Windows:
```bash
docker run --rm ^
  --add-host=host.docker.internal:host-gateway ^
  -p 5000:5000 ^
  -e OLLAMA_MODEL="llama2:7b" ^
  resume-screening
```

Access at: **http://localhost:5000**

---

## Configuration

### Change Model

**docker-compose.yml:**
```yaml
environment:
  - OLLAMA_MODEL=mistral  # Change model here
```

**Runtime:**
```bash
OLLAMA_MODEL="llama3.2" docker-compose up
```

### Manual Ollama URL (if needed)

**Linux:**
```bash
docker run -p 5000:5000 \
  -e OLLAMA_BASE_URL=http://172.17.0.1:11434 \
  -e OLLAMA_MODEL="llama2:7b" \
  resume-screening
```

**Windows:**
```bash
docker run -p 5000:5000 ^
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ^
  -e OLLAMA_MODEL="llama2:7b" ^
  resume-screening
```

---

## Usage

1. **Upload resumes** (PDF files or folder path)
2. **Provide job description** (upload .txt or paste)
3. **Set filters** (minimum score, top N resumes)
4. **Click "Match Resume(s)"**
5. **View results** with scores and analysis

---

## Troubleshooting

**Ollama connection failed:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# View logs
docker-compose logs resume-screening
```

**GPU error:** Edit `docker-compose.yml` and remove the `deploy` section.

**Port conflict:** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8501:5000"
```

**Model not found:** Pull the model first:
```bash
ollama pull llama2:7b
```

---

## Important Notes

- Ollama runs on **host machine**, not inside Docker
- The app connects to Ollama at `http://localhost:11434` (auto-detected)
- First Docker build takes 15-20 minutes (downloads ML libraries)
- Subsequent builds are cached and much faster

---

## Stop Application

**Docker:**
```bash
docker-compose down
```

**Local:**
```bash
# Press Ctrl+C in terminal
deactivate  # Exit virtual environment
```