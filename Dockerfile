FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add additional dependencies needed for NER and BERT
RUN pip install --no-cache-dir torch transformers datasets fastapi uvicorn pandas tqdm

# Pre-download the ClinicalBERT model to cache it in the image
RUN python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); print('ClinicalBERT model downloaded successfully')"

# Copy application code and models
COPY . .

# Create necessary directories
RUN mkdir -p Dataset/data model/BERT

# Make the disease_extraction script executable
RUN chmod +x disease_extraction

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "disease_api:app", "--host", "0.0.0.0", "--port", "8000"] 