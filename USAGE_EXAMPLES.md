# Usage Examples - Happiness Prediction System

This document provides practical examples of how to use the Happiness Prediction System API functions and components.

## Table of Contents
- [Basic Function Usage](#basic-function-usage)
- [Data Preprocessing Examples](#data-preprocessing-examples)
- [Model Training Examples](#model-training-examples)
- [Prediction Examples](#prediction-examples)
- [Integration Examples](#integration-examples)
- [Batch Processing Examples](#batch-processing-examples)
- [Error Handling Examples](#error-handling-examples)

## Basic Function Usage

### Text Cleaning with `cleanData()`

```python
from main import cleanData

# Example 1: Basic text cleaning
text = "Hello World! This is a great product."
cleaned = cleanData(text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
# Output: "HELLO WORLD GREAT PRODUCT hello world great product"

# Example 2: Handling special characters
text = "User@example.com says: 'Amazing service!' (5 stars)"
cleaned = cleanData(text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
# Output: "USER EXAMPL STAR user exampl say amaz servic star"

# Example 3: Processing social media text
text = "OMG! This is soooo good!!! #happy #love"
cleaned = cleanData(text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
# Output: "OMG HAPPI LOVE omg sooo good happi love"
```

### Sentiment Analysis with `sentenceSentiment()`

```python
from main import sentenceSentiment

# Example 1: Positive sentiment
text = "I love this product. It's absolutely amazing!"
sentiment = sentenceSentiment(text)
print(f"Text: {text}")
print(f"Sentiment: Polarity={sentiment[0]:.2f}, Subjectivity={sentiment[1]:.2f}")
# Output: Polarity=1.25, Subjectivity=1.8

# Example 2: Negative sentiment
text = "I hate this service. It's terrible and disappointing."
sentiment = sentenceSentiment(text)
print(f"Text: {text}")
print(f"Sentiment: Polarity={sentiment[0]:.2f}, Subjectivity={sentiment[1]:.2f}")
# Output: Polarity=-1.6, Subjectivity=1.8

# Example 3: Mixed sentiment
text = "The product is good but the service is bad. Overall okay."
sentiment = sentenceSentiment(text)
print(f"Text: {text}")
print(f"Sentiment: Polarity={sentiment[0]:.2f}, Subjectivity={sentiment[1]:.2f}")
# Output: Polarity=0.1, Subjectivity=1.4
```

### Label Conversion with `to_labels()`

```python
from main import to_labels

# Example 1: High confidence predictions
predictions = [0.85, 0.92, 0.78]
labels = [to_labels(pred) for pred in predictions]
print("High confidence predictions:")
for pred, label in zip(predictions, labels):
    print(f"  Probability: {pred:.2f} -> Label: {label}")
# Output: happy, happy, happy

# Example 2: Low confidence predictions
predictions = [0.15, 0.28, 0.43]
labels = [to_labels(pred) for pred in predictions]
print("Low confidence predictions:")
for pred, label in zip(predictions, labels):
    print(f"  Probability: {pred:.2f} -> Label: {label}")
# Output: not_happy, not_happy, not_happy

# Example 3: Edge cases
predictions = [0.5, 0.50001, 0.49999]
labels = [to_labels(pred) for pred in predictions]
print("Edge case predictions:")
for pred, label in zip(predictions, labels):
    print(f"  Probability: {pred:.5f} -> Label: {label}")
# Output: not_happy, happy, not_happy
```

## Data Preprocessing Examples

### Complete Text Preprocessing Pipeline

```python
import pandas as pd
from main import cleanData, sentenceSentiment

# Example dataset
data = {
    'User_ID': [1, 2, 3, 4, 5],
    'Description': [
        "I love this product! It's amazing.",
        "Terrible service. Very disappointed.",
        "Good quality, fast delivery.",
        "Not bad, but could be better.",
        "Excellent customer support!"
    ],
    'Browser_Used': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Edge'],
    'Device_Used': ['Desktop', 'Mobile', 'Desktop', 'Tablet', 'Mobile']
}

df = pd.DataFrame(data)

# Apply text cleaning
print("Original vs Cleaned Text:")
df['Description_Clean'] = df['Description'].apply(cleanData)
for idx, row in df.iterrows():
    print(f"Original: {row['Description']}")
    print(f"Cleaned:  {row['Description_Clean']}")
    print()

# Extract sentiment features
print("Sentiment Analysis:")
sentiment_features = df['Description'].apply(sentenceSentiment)
df['Polarity'] = [s[0] for s in sentiment_features]
df['Subjectivity'] = [s[1] for s in sentiment_features]

for idx, row in df.iterrows():
    print(f"Text: {row['Description']}")
    print(f"Sentiment: Polarity={row['Polarity']:.2f}, Subjectivity={row['Subjectivity']:.2f}")
    print()
```

### Categorical Feature Encoding

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Browser_Used': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Edge'],
    'Device_Used': ['Desktop', 'Mobile', 'Desktop', 'Tablet', 'Mobile']
})

# Label encoding
print("Original Data:")
print(data)

print("\nAfter Label Encoding:")
cols = ['Browser_Used', 'Device_Used']
for col in cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    print(f"{col} mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(data)

# One-hot encoding
print("\nAfter One-Hot Encoding:")
data_encoded = pd.get_dummies(data, columns=['Browser_Used', 'Device_Used'])
print(data_encoded)
```

## Model Training Examples

### Custom Training Loop

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout

# Example: Custom model training with different configurations
def create_model(vocab_size=20000, embedding_dim=100, sequence_length=1000):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=sequence_length),
        Conv1D(filters=embedding_dim, kernel_size=5, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(0.9),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage
model = create_model()
print("Model Architecture:")
print(model.summary())
```

### Progressive Batch Training

```python
def progressive_training(model, X_train, Y_train, X_val, Y_val):
    """
    Example implementation of progressive batch training
    """
    batch_sizes = [128, 512, 2048]
    
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        
        # Train for one epoch
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=1,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate performance
        val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Early stopping condition
        if val_accuracy > 0.95:
            print("Early stopping: High accuracy achieved")
            break
    
    return model

# Example usage (with dummy data)
# model = progressive_training(model, X_train, Y_train, X_val, Y_val)
```

## Prediction Examples

### Single Prediction

```python
from main import cleanData, sentenceSentiment, to_labels
import numpy as np

def predict_happiness(text, model, tokenizer, max_length=1000):
    """
    Complete prediction pipeline for a single text
    """
    # Step 1: Clean the text
    cleaned_text = cleanData(text)
    
    # Step 2: Extract sentiment features
    sentiment_features = sentenceSentiment(text)
    
    # Step 3: Tokenize and pad
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=max_length)
    
    # Step 4: Make prediction
    prediction = model.predict(padded)[0][0]
    
    # Step 5: Convert to label
    label = to_labels(prediction)
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'sentiment_polarity': sentiment_features[0],
        'sentiment_subjectivity': sentiment_features[1],
        'prediction_probability': prediction,
        'predicted_label': label
    }

# Example usage
text = "I absolutely love this product! It's fantastic and exceeded my expectations."
# result = predict_happiness(text, trained_model, tokenizer)
# print(result)
```

### Batch Prediction

```python
def predict_batch(texts, model, tokenizer, max_length=1000):
    """
    Batch prediction for multiple texts
    """
    results = []
    
    for text in texts:
        try:
            result = predict_happiness(text, model, tokenizer, max_length)
            results.append(result)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... Error: {e}")
            results.append({
                'original_text': text,
                'error': str(e)
            })
    
    return results

# Example usage
texts = [
    "I love this product!",
    "This is terrible quality.",
    "Not bad, could be better.",
    "Excellent service and fast delivery!",
    "Disappointed with the experience."
]

# results = predict_batch(texts, trained_model, tokenizer)
# for result in results:
#     if 'error' not in result:
#         print(f"Text: {result['original_text']}")
#         print(f"Prediction: {result['predicted_label']} ({result['prediction_probability']:.3f})")
#         print()
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from main import cleanData, sentenceSentiment, to_labels

app = Flask(__name__)

# Load model and tokenizer (global variables)
# model = load_model('model.h5')
# tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Process text
        cleaned = cleanData(text)
        sentiment = sentenceSentiment(text)
        
        # Make prediction (placeholder)
        # prediction = predict_happiness(text, model, tokenizer)
        
        return jsonify({
            'original_text': text,
            'cleaned_text': cleaned,
            'sentiment_polarity': sentiment[0],
            'sentiment_subjectivity': sentiment[1],
            # 'prediction': prediction['predicted_label'],
            # 'confidence': prediction['prediction_probability']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
```

### Command Line Interface

```python
import argparse
from main import cleanData, sentenceSentiment, to_labels

def main():
    parser = argparse.ArgumentParser(description='Happiness Prediction CLI')
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--clean-only', action='store_true', help='Only clean text')
    parser.add_argument('--sentiment-only', action='store_true', help='Only extract sentiment')
    
    args = parser.parse_args()
    
    if args.clean_only:
        cleaned = cleanData(args.text)
        print(f"Cleaned text: {cleaned}")
    
    elif args.sentiment_only:
        sentiment = sentenceSentiment(args.text)
        print(f"Sentiment - Polarity: {sentiment[0]:.3f}, Subjectivity: {sentiment[1]:.3f}")
    
    else:
        # Full pipeline
        cleaned = cleanData(args.text)
        sentiment = sentenceSentiment(args.text)
        
        print(f"Original: {args.text}")
        print(f"Cleaned: {cleaned}")
        print(f"Sentiment: Polarity={sentiment[0]:.3f}, Subjectivity={sentiment[1]:.3f}")
        
        # Note: Add model prediction here
        # prediction = predict_happiness(args.text, model, tokenizer)
        # print(f"Prediction: {prediction['predicted_label']}")

if __name__ == '__main__':
    main()
```

## Batch Processing Examples

### Processing CSV Files

```python
import pandas as pd
from main import cleanData, sentenceSentiment, to_labels

def process_csv(input_file, output_file):
    """
    Process a CSV file with text data
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Apply preprocessing
    print("Cleaning text data...")
    df['Description_Clean'] = df['Description'].apply(cleanData)
    
    print("Extracting sentiment features...")
    sentiment_features = df['Description'].apply(sentenceSentiment)
    df['Polarity'] = [s[0] for s in sentiment_features]
    df['Subjectivity'] = [s[1] for s in sentiment_features]
    
    # Add predictions (placeholder)
    print("Making predictions...")
    # df['Prediction_Prob'] = df['Description'].apply(lambda x: predict_happiness(x, model, tokenizer)['prediction_probability'])
    # df['Predicted_Label'] = df['Prediction_Prob'].apply(to_labels)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df

# Example usage
# processed_df = process_csv('input_data.csv', 'processed_data.csv')
```

### Streaming Data Processing

```python
import json
from main import cleanData, sentenceSentiment, to_labels

def process_stream(input_stream, output_stream):
    """
    Process streaming data (e.g., from Kafka, Redis, etc.)
    """
    for line in input_stream:
        try:
            # Parse JSON data
            data = json.loads(line)
            text = data.get('text', '')
            
            # Process text
            cleaned = cleanData(text)
            sentiment = sentenceSentiment(text)
            
            # Add results to data
            data['cleaned_text'] = cleaned
            data['sentiment_polarity'] = sentiment[0]
            data['sentiment_subjectivity'] = sentiment[1]
            
            # Make prediction (placeholder)
            # prediction = predict_happiness(text, model, tokenizer)
            # data['prediction'] = prediction['predicted_label']
            # data['confidence'] = prediction['prediction_probability']
            
            # Write to output stream
            output_stream.write(json.dumps(data) + '\n')
            
        except Exception as e:
            print(f"Error processing line: {line[:50]}... Error: {e}")

# Example usage
# with open('input_stream.jsonl', 'r') as input_file:
#     with open('output_stream.jsonl', 'w') as output_file:
#         process_stream(input_file, output_file)
```

## Error Handling Examples

### Robust Text Processing

```python
from main import cleanData, sentenceSentiment, to_labels
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_cleanData(text):
    """
    Wrapper for cleanData with error handling
    """
    try:
        if pd.isna(text) or text is None:
            return ""
        return cleanData(str(text))
    except Exception as e:
        logger.error(f"Error cleaning text: {text[:50]}... Error: {e}")
        return ""

def safe_sentenceSentiment(text):
    """
    Wrapper for sentenceSentiment with error handling
    """
    try:
        if pd.isna(text) or text is None or text == "":
            return [0.0, 0.0]
        return sentenceSentiment(str(text))
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {text[:50]}... Error: {e}")
        return [0.0, 0.0]

def safe_predict(text, model, tokenizer):
    """
    Safe prediction with comprehensive error handling
    """
    try:
        result = predict_happiness(text, model, tokenizer)
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {text[:50]}... Error: {e}")
        return {
            'original_text': text,
            'error': str(e),
            'predicted_label': 'not_happy',  # Default prediction
            'prediction_probability': 0.0
        }

# Example usage
texts = [
    "I love this product!",
    None,  # Null value
    "",    # Empty string
    "Normal text",
    123    # Non-string input
]

for text in texts:
    cleaned = safe_cleanData(text)
    sentiment = safe_sentenceSentiment(text)
    print(f"Input: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Sentiment: {sentiment}")
    print()
```

### Model Loading with Fallback

```python
import os
from keras.models import load_model
import pickle

def load_trained_model(model_path='model.h5', tokenizer_path='tokenizer.pkl'):
    """
    Load trained model and tokenizer with error handling
    """
    model = None
    tokenizer = None
    
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
            return None, None
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    try:
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from {tokenizer_path}")
        else:
            print(f"Tokenizer file not found: {tokenizer_path}")
    
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = None
    
    return model, tokenizer

# Example usage
model, tokenizer = load_trained_model()
if model is not None and tokenizer is not None:
    print("Model and tokenizer loaded successfully")
    # Use for predictions
else:
    print("Failed to load model or tokenizer")
```

These examples demonstrate various ways to use the Happiness Prediction System API in different scenarios, from simple function calls to complete integration examples with error handling and batch processing capabilities.