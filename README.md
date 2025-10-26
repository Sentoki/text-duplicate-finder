# text-duplicate-finder

HTTP service for text vectorization and duplicate news detection using the BAAI/bge-large-en-v1.5 embeddings model.

## Description

The service provides a REST API for converting texts into vector representations (embeddings) and computing similarity between them. Its primary purpose is detecting duplicate news articles by comparing their vector representations.

It uses the **BAAI/bge-large-en-v1.5** model — one of the best encoders for English texts at the time of project creation.

## Features

- Vectorization of individual texts
- Batch vectorization of multiple texts in a single request
- Computation of cosine similarity between vectors
- No authentication required (designed for internal network use)

## API Endpoints

### POST /embed

Vectorize a single text.

**Request:**
```json
{
  "text": "Breaking news: Scientists discover new planet"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "dimension": 1024
}
```

### POST /embed/batch

Batch vectorization of multiple texts.

**Request:**
```json
{
  "texts": [
    "First news article text",
    "Second news article text",
    "Third news article text"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.234, -0.567, 0.890, ...],
    [0.345, -0.678, 0.901, ...]
  ],
  "dimension": 1024,
  "count": 3
}
```

### POST /similarity

Compute cosine similarity between two vectors.

**Request:**
```json
{
  "vector1": [0.123, -0.456, 0.789, ...],
  "vector2": [0.234, -0.567, 0.890, ...]
}
```

**Response:**
```json
{
  "similarity": 0.87,
  "is_duplicate": true,
  "threshold": 0.85
}
```

The similarity value ranges from [-1, 1], where:
- 1 = identical texts
- 0 = no similarity
- -1 = opposite meanings

For news duplicate detection, a similarity threshold of ≥ 0.85 is recommended.

## Usage

### News Duplicate Detection

Typical workflow:

1. Get embedding for a new article via `/embed`
2. Compare with embeddings of existing articles via `/similarity`
3. If similarity ≥ 0.85 — it's a duplicate, ignore it

```python
import requests

API_URL = "http://localhost:8000"

# New article
new_article = "Breaking: Scientists discover water on Mars"
response = requests.post(f"{API_URL}/embed", json={"text": new_article})
new_embedding = response.json()["embedding"]

# Compare with existing articles
existing_embedding = [...]  # from database
response = requests.post(
    f"{API_URL}/similarity",
    json={"vector1": new_embedding, "vector2": existing_embedding}
)

if response.json()["is_duplicate"]:
    print("Duplicate detected, ignoring")
else:
    print("New unique article, saving")
```

## Installation and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn src.main:app --reload
```

On first run, the BAAI/bge-large-en-v1.5 model will be downloaded automatically (~1.3 GB).

## Requirements

- Python 3.12+
- CUDA (optional, for GPU acceleration)
- ~2 GB RAM for the model
- ~1.3 GB free space for model cache

## Technologies

- **FastAPI** — web framework
- **sentence-transformers** — working with transformer models
- **PyTorch** — neural network backend
- **BAAI/bge-large-en-v1.5** — encoder model

## Production Deployment

For deployment on Ubuntu 24.04 LTS servers, you need to configure automatic service startup via systemd, so it automatically starts on system reboot and restarts on failures.

**Detailed instructions:** [docs/deployment.md](docs/deployment.md)

Quick version:

```bash
# Create systemd service file
sudo nano /etc/systemd/system/text-duplicate-finder.service

# Enable autostart
sudo systemctl enable text-duplicate-finder
sudo systemctl start text-duplicate-finder
```

After configuration, the service will automatically start on every server reboot.
