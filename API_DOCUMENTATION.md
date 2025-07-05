# API Documentation - Happiness Prediction System

## Table of Contents
- [Overview](#overview)
- [Public Functions](#public-functions)
- [Model Configuration](#model-configuration)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Error Handling](#error-handling)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## Overview

This document provides detailed technical documentation for the Happiness Prediction System API. The system implements a deep learning pipeline for binary text classification using Convolutional Neural Networks (CNN) and pre-trained word embeddings.

## Public Functions

### `cleanData(text: str) -> str`

**Purpose**: Preprocesses raw text data for model input

**Algorithm**:
1. Convert input to string format
2. Remove all non-alphanumeric characters except spaces
3. Replace newlines with spaces
4. Filter out English stopwords
5. Extract uppercase letters for acronym preservation
6. Apply Porter stemming to remaining words
7. Combine processed acronyms with stemmed words
8. Remove single-character words

**Parameters**:
- `text` (str): Raw text input to be processed

**Returns**:
- `str`: Cleaned and preprocessed text

**Complexity**: O(n) where n is the length of input text

**Dependencies**:
- `nltk.corpus.stopwords`
- `nltk.stem.PorterStemmer`
- `re` module

**Error Handling**:
- Handles None input by converting to string
- Gracefully processes empty strings
- Maintains consistency with malformed text

**Example Usage**:
```python
# Basic usage
cleaned = cleanData("Hello World! This is a test.")
print(cleaned)  # Output: "HELLO WORLD hello world test"

# With special characters
cleaned = cleanData("User@domain.com said: 'Great product!'")
print(cleaned)  # Output: "USER DOMAIN user domain said great product"
```

### `sentenceSentiment(text: str) -> List[float]`

**Purpose**: Extracts sentiment features from text using TextBlob

**Algorithm**:
1. Split text into sentences using period delimiter
2. Create TextBlob object for each sentence
3. Extract polarity and subjectivity scores
4. Aggregate scores across all sentences
5. Return cumulative scores

**Parameters**:
- `text` (str): Input text for sentiment analysis

**Returns**:
- `List[float]`: [cumulative_polarity, cumulative_subjectivity]
  - `cumulative_polarity`: Sum of polarity scores (-∞ to +∞)
  - `cumulative_subjectivity`: Sum of subjectivity scores (0 to +∞)

**Complexity**: O(n) where n is the number of sentences

**Dependencies**:
- `textblob.TextBlob`

**Error Handling**:
- Handles empty strings by returning [0.0, 0.0]
- Processes text without periods as single sentence
- Manages TextBlob initialization errors

**Example Usage**:
```python
# Positive sentiment
sentiment = sentenceSentiment("I love this product. It's amazing!")
print(sentiment)  # Output: [1.25, 1.8] (example values)

# Negative sentiment
sentiment = sentenceSentiment("I hate this. It's terrible.")
print(sentiment)  # Output: [-1.6, 1.8] (example values)

# Neutral sentiment
sentiment = sentenceSentiment("This is a factual statement.")
print(sentiment)  # Output: [0.0, 0.0]
```

### `to_labels(x: float) -> str`

**Purpose**: Converts prediction probability to categorical label

**Algorithm**:
1. Compare input probability to threshold (0.5)
2. Return appropriate string label

**Parameters**:
- `x` (float): Prediction probability between 0 and 1

**Returns**:
- `str`: Classification label ("happy" or "not_happy")

**Complexity**: O(1)

**Error Handling**:
- Handles edge case where x = 0.5 (returns "not_happy")
- Processes values outside [0,1] range based on threshold

**Example Usage**:
```python
# High confidence positive
label = to_labels(0.85)
print(label)  # Output: "happy"

# High confidence negative
label = to_labels(0.15)
print(label)  # Output: "not_happy"

# Edge case
label = to_labels(0.5)
print(label)  # Output: "not_happy"
```

## Model Configuration

### Constants
```python
MAX_SEQUENCE_LENGTH = 1000    # Maximum number of tokens per sequence
MAX_NB_WORDS = 20000         # Maximum vocabulary size
EMBEDDING_DIM = 100          # Word embedding dimension
BATCH_SIZES = [128, 512, 2048]  # Progressive batch training
VALIDATION_SPLIT = 0.95      # Train/validation split ratio
```

### Hyperparameters
```python
OPTIMIZER = "adam"
LOSS_FUNCTION = "binary_crossentropy"
METRICS = ["accuracy"]
DROPOUT_RATE = 0.9
HIDDEN_UNITS = 512
CONV_FILTERS = 100
FILTER_SIZE = 5
POOL_SIZE = 2
```

## Data Pipeline

### Input Data Requirements

**Training Data Structure**:
```
train.csv:
├── User_ID (int): Unique identifier
├── Description (str): Text to analyze
├── Browser_Used (str): Browser type
├── Device_Used (str): Device type
└── Is_Response (str): Target label ("happy"/"not_happy")
```

**Test Data Structure**:
```
test.csv:
├── User_ID (int): Unique identifier
├── Description (str): Text to analyze
├── Browser_Used (str): Browser type
└── Device_Used (str): Device type
```

### Data Preprocessing Pipeline

1. **Text Cleaning**:
   - Apply `cleanData()` function to all descriptions
   - Remove punctuation and special characters
   - Filter stopwords and apply stemming

2. **Sentiment Feature Extraction**:
   - Apply `sentenceSentiment()` to extract polarity and subjectivity
   - Add as additional features to dataset

3. **Categorical Encoding**:
   - Label encode Browser_Used and Device_Used
   - Apply one-hot encoding for categorical variables
   - Remove rare categories (< 1% frequency)

4. **Text Tokenization**:
   - Create vocabulary from training text
   - Convert text to integer sequences
   - Pad sequences to maximum length

5. **Feature Scaling**:
   - Multiply categorical features by 100 for scaling
   - Combine with tokenized text features

## Feature Engineering

### Text Features
- **Tokenized sequences**: Integer-encoded text (max 1000 tokens)
- **Polarity score**: Sentiment polarity from TextBlob
- **Subjectivity score**: Sentiment subjectivity from TextBlob

### Categorical Features
- **Browser encoding**: One-hot encoded browser types
- **Device encoding**: One-hot encoded device types

### Feature Combination
```python
# Final feature vector structure
[scaled_categorical_features, tokenized_text_features]
```

## Model Architecture

### Layer Structure
```python
model = Sequential([
    Embedding(input_dim=MAX_NB_WORDS, 
              output_dim=EMBEDDING_DIM, 
              input_length=SEQUENCE_LENGTH),
    Conv1D(filters=EMBEDDING_DIM, 
           kernel_size=5, 
           padding='same', 
           activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dropout(0.9),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Model Summary
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 1000, 100)        2000000   
conv1d (Conv1D)             (None, 1000, 100)        50100     
max_pooling1d (MaxPooling1D) (None, 500, 100)         0         
flatten (Flatten)           (None, 50000)            0         
dropout (Dropout)           (None, 50000)            0         
dense (Dense)               (None, 512)              25600512  
dense_1 (Dense)             (None, 1)                513       
=================================================================
Total params: 27,651,125
Trainable params: 27,651,125
Non-trainable params: 0
```

## Training Process

### Progressive Batch Training
```python
BATCH_SIZES = [128, 512, 2048]
for batch_size in BATCH_SIZES:
    model.fit(X_train, Y_train, 
              validation_data=(X_dev, Y_dev), 
              epochs=1, 
              batch_size=batch_size)
```

### Training Configuration
- **Epochs per batch size**: 1
- **Validation split**: 5% of training data
- **Early stopping**: Based on validation accuracy
- **Model checkpointing**: Save best model as 'model.h5'

## Error Handling

### Common Error Scenarios

1. **Missing Data Files**:
   ```python
   try:
       train = pd.read_csv('train.csv')
       test = pd.read_csv('test.csv')
   except FileNotFoundError:
       print("Error: Required CSV files not found")
   ```

2. **Missing GloVe Embeddings**:
   ```python
   try:
       f = open('glove.6B.100d.txt', encoding="utf8")
   except FileNotFoundError:
       print("Error: GloVe embeddings file not found")
   ```

3. **Memory Issues**:
   - Monitor RAM usage during embedding loading
   - Consider using smaller embedding dimensions
   - Implement batch processing for large datasets

4. **NLTK Data Missing**:
   ```python
   import nltk
   try:
       stops = set(stopwords.words("english"))
   except LookupError:
       nltk.download('stopwords')
   ```

## Performance Metrics

### Model Evaluation
- **Primary Metric**: Binary accuracy
- **Loss Function**: Binary crossentropy
- **Validation Method**: Hold-out validation (5%)

### Expected Performance
- **Training Accuracy**: ~90-95%
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~88% (as per code comment)

### Performance Monitoring
```python
scores = model.evaluate(X_dev, Y_dev, verbose=0)
print("Accuracy: %.5f%%" % (scores[1]*100))
```

## Troubleshooting

### Common Issues and Solutions

1. **Low Model Accuracy**:
   - Verify data preprocessing steps
   - Check for data leakage
   - Adjust hyperparameters
   - Increase training epochs

2. **Memory Errors**:
   - Reduce batch size
   - Use smaller embedding dimensions
   - Implement data generators

3. **Training Stalls**:
   - Check for vanishing gradients
   - Adjust learning rate
   - Implement gradient clipping

4. **Prediction Errors**:
   - Verify input data format
   - Check preprocessing consistency
   - Validate model loading

### Debug Mode
```python
# Enable verbose logging
model.fit(X_train, Y_train, verbose=1)
model.evaluate(X_test, Y_test, verbose=1)
```

### Performance Optimization
```python
# GPU acceleration
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Memory growth configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Version Information

- **API Version**: 1.0
- **Python Version**: 3.7+
- **TensorFlow Version**: 2.6+
- **Keras Version**: 2.6+
- **Last Updated**: [Current Date]

## Support

For technical support and bug reports, please refer to the project repository issues section.