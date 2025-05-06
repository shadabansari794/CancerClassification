from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from transformers import pipeline
import uvicorn
import time
import os
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="Cancer Classification and Disease Extraction API",
    description="API for classifying medical abstracts and extracting disease mentions using ClinicalBERT",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models
MODEL_DIR = os.path.join(os.getcwd(), "model/BERT")
NER_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Load the BERT classification model for Cancer/Non-Cancer prediction
print("Loading BERT classification model from model directory...")
try:
    # Check if model files exist
    if os.path.exists(MODEL_DIR):
        # Load the tokenizer and model
        bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("BERT classification model loaded successfully")
    else:
        print(f"Model directory {MODEL_DIR} not found, falling back to ClinicalBERT")
        # Fallback to ClinicalBERT
        bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
        print("ClinicalBERT model loaded for classification")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    raise RuntimeError(f"Failed to load BERT model: {e}")

# Load NER pipeline with ClinicalBERT
print("Loading ClinicalBERT for entity recognition...")
try:
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    ner_model = AutoModel.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, tokenizer=ner_tokenizer, aggregation_strategy="simple")
    print("ClinicalBERT NER model loaded successfully")
    
    # Debug: Print available entity labels to ensure we're using the right ones
    try:
        all_labels = list(ner_pipeline.model.config.id2label.values())
        print(f"Available entity labels: {all_labels}")
    except:
        print("Could not retrieve entity labels from model")
except Exception as e:
    print(f"Error loading ClinicalBERT NER model: {e}")
    print("Falling back to biomedical NER model...")
    try:
        NER_MODEL_NAME = "d4data/biomedical-ner-all"
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
        ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
        print("Fallback NER model loaded successfully")
    except Exception as e2:
        print(f"Error loading fallback NER model: {e2}")
        raise RuntimeError(f"Failed to load NER model: {e2}")

# Define cancer-related labels and keywords for NER detection
# Using lowercase for case-insensitive comparison
disease_labels_lower = {"disease", "cancer", "tumor", "carcinoma", "problem", "diagnosis", "finding", "disorder", "syndrome", "neoplasm"}
print(f"Filtering for entities with labels (case-insensitive): {disease_labels_lower}")

# Cancer-specific keywords for detection
cancer_keywords = [
    "cancer", "tumor", "carcinoma", "leukemia", "lymphoma", 
    "sarcoma", "melanoma", "neoplasm", "malignancy", "metastasis",
    "oncology", "adenocarcinoma", "glioma", "myeloma", "blastoma",
    "malignant", "metastatic", "neoplastic", "cancerous"
]

# Define label mapping for classification
id2label = {0: "Non-Cancer", 1: "Cancer"}
label2id = {"Non-Cancer": 0, "Cancer": 1}

# Request and response models
class DocumentInput(BaseModel):
    id: str
    title: str = ""
    abstract: str

class DiseaseEntity(BaseModel):
    disease: str
    score: float

class AnalysisResponse(BaseModel):
    document_id: str
    predicted_labels: List[str]
    confidence_scores: Dict[str, float]
    extracted_diseases: List[DiseaseEntity]
    processing_time_ms: float

def predict_category(text: str):
    """Predict if the abstract is about cancer or not using the BERT model"""
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted class and confidence scores
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    class_probabilities = {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return {
        "predicted_labels": [id2label[predicted_class]],
        "confidence_scores": class_probabilities
    }

def extract_diseases_from_text(text: str):
    """Extract disease mentions from text using the ClinicalBERT NER model"""
    if not text.strip():
        return []
    
    # Process the text in chunks if it's very long
    # ClinicalBERT context length limit is 512
    if len(text) > 512:
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        all_entities = []
        for chunk in chunks:
            all_entities.extend(ner_pipeline(chunk))
    else:
        all_entities = ner_pipeline(text)
    
    # Debug: print first few entities to understand structure
    if len(all_entities) > 0:
        print(f"First entity example: {all_entities[0]}")
    
    # Filter for disease entities with broader matching criteria
    disease_entities = {}
    for entity in all_entities:
        if isinstance(entity, dict):
            # Case insensitive matching and broader label criteria
            entity_group = entity.get("entity_group", "").lower() if "entity_group" in entity else \
                          entity.get("entity", "").lower() if "entity" in entity else ""
            entity_word = entity.get("word", "") if "word" in entity else \
                         entity.get("text", "") if "text" in entity else ""
            entity_score = entity.get("score", 0.0) if "score" in entity else 0.75
            
            if entity_group and any(label in entity_group for label in disease_labels_lower):
                if entity_word not in disease_entities or entity_score > disease_entities[entity_word]:
                    disease_entities[entity_word] = entity_score
            # Direct keyword matching for critical terms
            elif entity_word and any(keyword in entity_word.lower() for keyword in cancer_keywords):
                if entity_word not in disease_entities or entity_score > disease_entities[entity_word]:
                    disease_entities[entity_word] = entity_score
    
    # If we still didn't find diseases, try a simple text-based approach
    if not disease_entities and text:
        # Simple text search for disease terms
        words = text.lower().split()
        for i, word in enumerate(words):
            if any(keyword in word for keyword in cancer_keywords):
                # Try to extract a reasonable phrase (up to 3 words)
                start = max(0, i-1)
                end = min(len(words), i+2)
                phrase = " ".join(words[start:end])
                disease_entities[phrase] = 0.75  # Assign a moderate confidence score
    
    # Format the response
    result = [
        DiseaseEntity(disease=disease, score=score) 
        for disease, score in disease_entities.items()
    ]
    
    # Sort by score (highest first)
    result.sort(key=lambda x: x.score, reverse=True)
    
    return result

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_document(document: DocumentInput):
    """
    Analyze a medical abstract: classify it as Cancer/Non-Cancer and extract diseases
    
    Parameters:
    - id: Document identifier
    - title: Document title (optional)
    - abstract: Text to analyze
    
    Returns:
    - document_id: The original document ID
    - predicted_labels: List of predicted categories (Cancer or Non-Cancer)
    - confidence_scores: Confidence scores for each category
    - extracted_diseases: List of detected diseases with confidence scores
    - processing_time_ms: Time taken to process the request in milliseconds
    """
    
    start_time = time.time()
    
    try:
        # Use the abstract text for prediction
        abstract = document.abstract
        
        # Get category prediction
        classification_result = predict_category(abstract)
        
        # Extract diseases
        extracted_diseases = extract_diseases_from_text(abstract)
        
        # If classification predicts "Cancer" but no diseases found, try to extract based on keywords
        if classification_result["predicted_labels"][0] == "Cancer" and not extracted_diseases:
            cancer_terms = cancer_keywords
            lower_abstract = abstract.lower()
            
            # Find mentions of cancer terms
            for term in cancer_terms:
                if term in lower_abstract:
                    index = lower_abstract.find(term)
                    # Get surrounding context (20 chars before and after)
                    start = max(0, index - 20)
                    end = min(len(lower_abstract), index + len(term) + 20)
                    context = abstract[start:end]
                    
                    # Add as a disease entity
                    disease_type = context
                    extracted_diseases.append(DiseaseEntity(disease=disease_type, score=0.8))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return AnalysisResponse(
        document_id=document.id,
        predicted_labels=classification_result["predicted_labels"],
        confidence_scores=classification_result["confidence_scores"],
        extracted_diseases=extracted_diseases,
        processing_time_ms=processing_time
    )

# Keep the original endpoints for backward compatibility
@app.post("/extract_diseases", response_model=Dict[str, Any])
def extract_diseases(document: DocumentInput):
    """Extract disease mentions from a medical abstract"""
    result = extract_diseases_from_text(document.abstract)
    return {
        "document_id": document.id,
        "extracted_diseases": result,
        "processing_time_ms": 0
    }

@app.post("/classify", response_model=Dict[str, Any])
def classify_document(document: DocumentInput):
    """Classify a medical abstract as Cancer or Non-Cancer"""
    return predict_category(document.abstract)

@app.get("/health")
def health_check():
    """Check if the API is up and running"""
    return {
        "status": "healthy", 
        "models": {
            "classification": "BERT model for cancer classification",
            "ner": NER_MODEL_NAME
        }
    }

if __name__ == "__main__":
    uvicorn.run("disease_api:app", host="0.0.0.0", port=8000, reload=True) 