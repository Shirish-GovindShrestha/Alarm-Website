# models/ner_model.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
import string
import re

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/ner_model")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, max_length=128)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()  # evaluation mode
    print("✅ NER model loaded successfully!")
except Exception as e:
    print("❌ Failed to load NER model:", e)
    tokenizer = None
    model = None

# Mapping from id to tag
id2tag = model.config.id2label if model else {}


def clean_token(token):
    """Clean token by removing special characters but preserving meaningful punctuation"""
    # Remove leading/trailing punctuation but keep internal ones (like O'Connor, U.S.A)
    # Keep standalone ':' so we can merge times like 8 : 00 later
    if token == ':':
        return ':'
    token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
    return token


def merge_subwords_improved(tokens, tags, word_ids=None):
    """Improved subword merging with better handling of BIO tags"""
    if not tokens or not tags:
        return [], []

    merged_tokens = []
    merged_tags = []
    current_word = ""
    current_tag = "O"

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # Skip special tokens
        if token in tokenizer.all_special_tokens:
            continue

        # Clean token
        clean_tok = clean_token(token)
        if not clean_tok:  # Skip if token becomes empty after cleaning
            continue

        # Merge colon and AM/PM into current time token when appropriate
        if clean_tok == ':' and current_word and re.fullmatch(r"\d{1,2}", current_word):
            current_word += ':'
            # Prefer to keep time tag if present
            if current_tag.startswith('B-time'):
                current_tag = 'I-time'
            elif current_tag == 'O' and tag.endswith('time'):
                current_tag = tag.replace('B-', 'I-')
            continue

        if clean_tok.lower() in ('am', 'pm') and current_word and re.fullmatch(r"\d{1,2}(:\d{2})?", current_word):
            current_word += ' ' + clean_tok.upper()
            if current_tag.startswith('B-time'):
                current_tag = 'I-time'
            elif tag.endswith('time'):
                current_tag = tag.replace('B-', 'I-')
            continue

        # Handle subword tokens (starting with ##)
        if token.startswith("##"):
            # Merge with previous token
            current_word += clean_tok
            # Keep the tag from the first subword, but convert B- to I- for consistency
            if current_tag.startswith("B-"):
                current_tag = current_tag.replace("B-", "I-", 1)
        else:
            # Save previous merged word if exists
            if current_word:
                merged_tokens.append(current_word)
                merged_tags.append(current_tag)

            # Start new word
            current_word = clean_tok
            current_tag = tag

    # Don't forget the last word
    if current_word:
        merged_tokens.append(current_word)
        merged_tags.append(current_tag)

    return merged_tokens, merged_tags


def fix_bio_sequence(tags):
    """Fix BIO tag sequence inconsistencies"""
    if not tags:
        return tags

    fixed_tags = []
    prev_entity_type = None

    for tag in tags:
        if tag == "O":
            fixed_tags.append(tag)
            prev_entity_type = None
        elif tag.startswith("B-"):
            entity_type = tag[2:]
            fixed_tags.append(tag)
            prev_entity_type = entity_type
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            # If I- tag appears without preceding B- of same type, convert to B-
            if prev_entity_type != entity_type:
                fixed_tags.append(f"B-{entity_type}")
            else:
                fixed_tags.append(tag)
            prev_entity_type = entity_type
        else:
            # Handle any other tags
            fixed_tags.append(tag)
            prev_entity_type = None

    return fixed_tags


def extract_entities(tokens, tags):
    """Extract named entities from tokens and BIO tags"""
    entities = []
    current_entity = {"text": "", "type": "", "start": -1, "end": -1}

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # Save previous entity if exists
            if current_entity["text"]:
                current_entity["end"] = i - 1
                entities.append(current_entity.copy())

            # Start new entity
            entity_type = tag[2:]
            current_entity = {
                "text": token,
                "type": entity_type,
                "start": i,
                "end": i
            }

        elif tag.startswith("I-") and current_entity["text"]:
            # Continue current entity
            entity_type = tag[2:]
            if entity_type == current_entity["type"]:
                current_entity["text"] += " " + token
                current_entity["end"] = i
            else:
                # Type mismatch - save current and start new
                entities.append(current_entity.copy())
                current_entity = {
                    "text": token,
                    "type": entity_type,
                    "start": i,
                    "end": i
                }

        elif tag == "O":
            # End current entity if exists
            if current_entity["text"]:
                entities.append(current_entity.copy())
                current_entity = {"text": "", "type": "", "start": -1, "end": -1}

    # Don't forget last entity
    if current_entity["text"]:
        entities.append(current_entity.copy())

    return entities


def post_process_entities(entities):
    """Post-process entities to clean up common issues"""
    processed_entities = []

    for entity in entities:
        text = entity["text"].strip()

        # Skip empty or very short entities
        if len(text) < 2:
            continue

        # Clean up entity text
        # Remove leading/trailing punctuation
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text).strip()

        # Skip if entity becomes empty
        if not text:
            continue

        # Update entity with cleaned text
        entity["text"] = text
        processed_entities.append(entity)

    return processed_entities


def predict(text):
    """Return list of (token, BIO tag) tuples for a given text with improved processing"""
    if not model or not tokenizer:
        return []

    # Encode input with attention to word boundaries
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    inputs = {k: v.to(model.device) for k, v in encodings.items() if k != 'offset_mapping'}

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits

        # Use confidence-based prediction instead of just argmax
        probabilities = torch.softmax(logits, dim=2)
        pred_ids = torch.argmax(probabilities, dim=2)[0].cpu().tolist()
        confidences = torch.max(probabilities, dim=2)[0][0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])
    raw_tags = [id2tag.get(pred_id, 'O') for pred_id in pred_ids]

    # Apply confidence threshold (optional - you can tune this)
    confidence_threshold = 0.5
    filtered_tags = []
    for tag, conf in zip(raw_tags, confidences):
        if conf < confidence_threshold and tag != 'O':
            filtered_tags.append('O')  # Low confidence entity -> O
        else:
            filtered_tags.append(tag)

    # Merge subwords with improved handling
    merged_tokens, merged_tags = merge_subwords_improved(tokens, filtered_tags)

    # Fix BIO sequence
    fixed_tags = fix_bio_sequence(merged_tags)

    return list(zip(merged_tokens, fixed_tags))


def get_entities(text):
    """Get structured entity information from text"""
    if not model or not tokenizer:
        return []

    # Get token-tag pairs
    token_tag_pairs = predict(text)

    if not token_tag_pairs:
        return []

    tokens, tags = zip(*token_tag_pairs)

    # Extract entities
    entities = extract_entities(tokens, tags)

    # Post-process entities
    entities = post_process_entities(entities)

    return entities


def get_entity_summary(text):
    """Get a summary of entities found in the text"""
    entities = get_entities(text)

    if not entities:
        return {"total": 0, "by_type": {}, "entities": []}

    # Count by type
    type_counts = {}
    for entity in entities:
        entity_type = entity["type"]
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

    return {
        "total": len(entities),
        "by_type": type_counts,
        "entities": entities
    }


def __init__():
    user_input = input("Enter text for NER (or 'exit' to quit): ") 
    result = get_entity_summary(user_input)
    print("Entity Summary:", result)