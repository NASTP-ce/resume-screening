#!/bin/sh

# Auto-detect the appropriate Ollama host
if [ -z "$OLLAMA_BASE_URL" ]; then
    # Try host.docker.internal first (Windows/Mac Docker Desktop and recent Linux)
    if wget --spider -q --timeout=2 http://host.docker.internal:11434/api/tags 2>/dev/null; then
        export OLLAMA_BASE_URL="http://host.docker.internal:11434"
        echo "Detected Ollama via host.docker.internal"
    # Try Docker bridge gateway (Linux fallback)
    elif wget --spider -q --timeout=2 http://172.17.0.1:11434/api/tags 2>/dev/null; then
        export OLLAMA_BASE_URL="http://172.17.0.1:11434"
        echo "Detected Ollama via Docker bridge (172.17.0.1)"
    else
        export OLLAMA_BASE_URL="http://localhost:11434"
        echo "Using default localhost (Ollama not detected)"
    fi
else
    echo "Using OLLAMA_BASE_URL from environment: $OLLAMA_BASE_URL"
fi

echo "Starting application with Ollama at: $OLLAMA_BASE_URL"

# Start the application
exec streamlit run app.py --server.port=5000 --server.address=0.0.0.0