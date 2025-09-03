from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/intent_model")

print("Loading Intent model from:", MODEL_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True,max_length=128)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    print("✅ Intent model loaded successfully!")
except Exception as e:
    print("❌ Failed to load Intent model:", e)
    tokenizer = None
    model = None


intent_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

def predict(text):
    return intent_pipeline(text)[0]['label']


