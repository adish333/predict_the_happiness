# Happiness Prediction from Text Data

A machine learning project that predicts whether a user's response indicates happiness or not, based on their textual description. This project uses deep learning with Convolutional Neural Networks (CNN) and pre-trained GloVe word embeddings to analyze sentiment and classify user responses.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a text classification system that:
- Preprocesses text data by cleaning and removing stopwords
- Extracts sentiment features using TextBlob
- Incorporates browser and device usage information
- Uses pre-trained GloVe word embeddings for better text representation
- Trains a CNN model for binary classification (happy/not happy)
- Provides predictions with confidence scores

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd happiness-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

5. Download GloVe word embeddings:
   - Download `glove.6B.100d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
   - Place it in the project root directory

## Data Requirements

The project expects the following CSV files:

### Training Data (`train.csv`)
Required columns:
- `User_ID`: Unique identifier for each user
- `Description`: Text description to analyze
- `Browser_Used`: Browser type used by the user
- `Device_Used`: Device type used by the user
- `Is_Response`: Target variable ("happy" or "not_happy")

### Test Data (`test.csv`)
Required columns:
- `User_ID`: Unique identifier for each user
- `Description`: Text description to analyze
- `Browser_Used`: Browser type used by the user
- `Device_Used`: Device type used by the user

## Usage

### Basic Usage

```python
# Run the complete pipeline
python main.py
```

This will:
1. Load and preprocess the data
2. Train the model
3. Generate predictions
4. Save the model as `model.h5`
5. Create `submission.csv` with predictions

### Using Individual Functions

```python
from main import cleanData, sentenceSentiment, to_labels

# Clean text data
clean_text = cleanData("This is a sample text with punctuation!")
print(clean_text)  # Output: cleaned text without stopwords and punctuation

# Extract sentiment features
sentiment_features = sentenceSentiment("I love this product. It's amazing!")
print(sentiment_features)  # Output: [polarity_score, subjectivity_score]

# Convert prediction to label
label = to_labels(0.7)
print(label)  # Output: "happy"
```

## API Reference

### Functions

#### `cleanData(text)`

Preprocesses text data by removing punctuation, stopwords, and applying stemming.

**Parameters:**
- `text` (str): Input text to clean

**Returns:**
- `str`: Cleaned and preprocessed text

**Example:**
```python
clean_text = cleanData("Hello! This is a sample text with punctuation.")
# Returns: "sampl text punctuat"
```

#### `sentenceSentiment(text)`

Extracts sentiment features (polarity and subjectivity) from text using TextBlob.

**Parameters:**
- `text` (str): Input text to analyze

**Returns:**
- `list`: [polarity_score, subjectivity_score]
  - polarity_score: Float indicating sentiment (-1 to 1)
  - subjectivity_score: Float indicating subjectivity (0 to 1)

**Example:**
```python
sentiment = sentenceSentiment("I love this product. It's amazing!")
# Returns: [0.625, 0.9] (example values)
```

#### `to_labels(x)`

Converts prediction probability to categorical label.

**Parameters:**
- `x` (float): Prediction probability (0 to 1)

**Returns:**
- `str`: "happy" if x > 0.5, "not_happy" otherwise

**Example:**
```python
label = to_labels(0.7)  # Returns: "happy"
label = to_labels(0.3)  # Returns: "not_happy"
```

## Model Architecture

The model uses the following architecture:

1. **Embedding Layer**: Converts text tokens to dense vectors using GloVe embeddings
2. **Conv1D Layer**: Applies 1D convolution for feature extraction
3. **MaxPooling1D**: Reduces dimensionality while preserving important features
4. **Flatten Layer**: Converts 2D feature maps to 1D
5. **Dropout Layer**: Prevents overfitting (90% dropout rate)
6. **Dense Layers**: 
   - Hidden layer with 512 neurons and ReLU activation
   - Output layer with 1 neuron and sigmoid activation

**Model Configuration:**
- Maximum sequence length: 1000 tokens
- Maximum vocabulary size: 20,000 words
- Embedding dimension: 100
- Optimizer: Adam
- Loss function: Binary crossentropy

## Features

### Text Preprocessing
- Punctuation removal
- Stopword filtering
- Stemming using Porter Stemmer
- Tokenization and padding

### Feature Engineering
- Sentiment polarity extraction
- Subjectivity analysis
- Browser and device usage encoding
- One-hot encoding for categorical variables

### Model Training
- Progressive batch size training (128 → 512 → 2048)
- Validation split (95% train, 5% validation)
- Early stopping capability
- Model checkpointing

## Output

The script generates the following files:

1. **`model.h5`**: Trained Keras model file
2. **`submission.csv`**: Predictions for test data with columns:
   - `User_ID`: User identifier
   - `Is_Response`: Predicted label ("happy" or "not_happy")

## Model Performance

The model achieves approximately 88% accuracy on the test set, as indicated by the comment in the code.

## Dependencies

- Python 3.7+
- TensorFlow/Keras 2.6+
- NumPy
- Pandas
- scikit-learn
- NLTK
- TextBlob
- Pre-trained GloVe embeddings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Notes

- Ensure you have sufficient memory (RAM) for loading GloVe embeddings
- Training time depends on your hardware configuration
- The model uses GPU acceleration if available
- Consider adjusting hyperparameters for different datasets
