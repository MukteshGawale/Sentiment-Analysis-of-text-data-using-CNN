# Sentiment Analysis of text data using a Convolutional Neural Network (CNN)

## Overview
This project performs sentiment analysis, classifying text as either positive or negative. It employs a Convolutional Neural Network (CNN) for this task, leveraging common Natural Language Processing (NLP) techniques for data preprocessing, model training, and evaluation. This notebook provides a clear and complete example of how to build and train a CNN for sentiment analysis from start to finish

## Dataset
The dataset used in this project is the **IMDB Dataset**, which contains **50,000 movie reviews** labeled as either positive or negative. It is a widely used benchmark dataset for sentiment classification tasks. 
## Why CNN?
CNNs are effective for sentiment analysis because they use convolutional filters to capture local patterns and contextual information in text. This allows them to efficiently identify sentiment-bearing n-grams (like bigrams and trigrams). Unlike simpler models (e.g., bag-of-words) or sequential models (RNNs), CNNs can process text in parallel, making them computationally efficient. Pooling layers reduce dimensionality while preserving key features, improving generalization and handling longer text sequences well

## Key Features
📌 **Text Preprocessing**
- Expands contractions using `contractions` (e.g., "can't" to "cannot")
- Removes stop words using `nltk.corpus.stopwords`
- Handles duplicate reviews by identifying and removing redundant entries to improve model performance

🔠 **Tokenization**
- Converts text data into numerical sequences using `tensorflow.keras.preprocessing.text.Tokenizer`
- Applies padding to standardize sequence length

🧠 **CNN Model Implementation**
- Implements a 1D CNN architecture with an embedding layer, convolutional layers, max pooling, and dense layers
- Uses dropout layers to reduce overfitting and improve generalization

📊 **Training and Evaluation**
- Trains the model with early stopping and learning rate reduction (`EarlyStopping`, `ReduceLROnPlateau`)
- Evaluates model performance on the IMDB dataset
- Achieves **~88% accuracy** on test data

## Conclusion
The CNN-based sentiment analysis model demonstrates a strong ability to classify text sentiment efficiently. The use of convolutional layers enhances feature extraction, leading to improved performance compared to traditional machine learning approaches.

## Next Steps
- **Hyperparameter Tuning**: Experiment with different kernel sizes, learning rates, and optimizers to further boost accuracy.
- **Expand Dataset**: Train the model on a larger and more diverse dataset to enhance generalization.


