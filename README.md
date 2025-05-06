# Cancer Classification and Disease Extraction API

This repository contains a FastAPI application for cancer classification and disease extraction from medical abstracts.

## Features

- Cancer classification using BERT model
- Disease entity extraction using PubMedBERT
- Cancer-specific disease identification
- RESTful API with multiple endpoints

## Model Training with Kaggle

For this project, Kaggle was chosen as the platform for training our cancer classification models for several key reasons:

1. **GPU Resources**: Kaggle provides free access to NVIDIA Tesla P100 GPUs, which significantly accelerated the BERT model training process.

2. **Pre-built Environments**: Kaggle notebooks come with pre-installed machine learning libraries like PyTorch and Transformers, eliminating environment setup time.

3. **Large Dataset Handling**: The platform efficiently handles the large biomedical datasets required for training cancer classification models.

4. **Collaborative Features**: Kaggle provides version control and the ability to fork and share notebooks, enabling better team collaboration.

5. **Reproducibility**: Training on Kaggle ensures consistent environment specifications for reproducible results across different machines.

The trained BERT model was then exported from Kaggle and integrated into this API for cancer classification.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (if running without Docker)

## Getting Started

### Docker Deployment

1. Clone this repository
2. Ensure you have Docker and Docker Compose installed
3. Build and start the service:

```bash
docker-compose up -d
```

4. The API will be available at http://localhost:8000
5. Access the API documentation at http://localhost:8000/docs

### Running Without Docker

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install torch transformers datasets fastapi uvicorn pandas tqdm
```

2. Start the API server:

```bash
uvicorn disease_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `/analyze` - Analyze a document for cancer classification and disease extraction
- `/extract_diseases` - Extract diseases from a document
- `/classify` - Classify a document as Cancer/Non-Cancer
- `/health` - Check API health status

## Example Usage

### Analyzing a Medical Abstract

```python
import requests
import json

url = "http://localhost:8000/analyze"

payload = {
    "id": "123456",
    "title": "Advances in Breast Cancer Treatment",
    "abstract": "Recent studies have shown promising results in the treatment of breast cancer patients using novel therapies..."
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))
```

## Running the Disease Extraction Script

The repository includes a standalone script for extracting cancer-related diseases from datasets:

```bash
# Inside Docker container
docker exec -it cancer_classification_api python disease_extraction

# Or locally
python disease_extraction
```

## Development

To modify the disease extraction or API functionality, edit the following files:

- `disease_api.py` - Main FastAPI application
- `disease_extraction` - Script for processing datasets and extracting diseases

## License

[MIT License](LICENSE) 