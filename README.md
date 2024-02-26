# Text Classification with Word2Vec Embeddings and K-Nearest Neighbors

This README demonstrates text classification using Word2Vec embeddings and the K-Nearest Neighbors (KNN) algorithm.

## Data Preprocessing

- The IMDb dataset is loaded using the Keras IMDb dataset loader.
- Integer sequences representing words are converted to text using the IMDb word index.
- Texts are tokenized and preprocessed by removing non-alphabetic characters, converting to lowercase, and removing stop words.

## Word Embeddings

- Word2Vec embeddings are loaded from the Google News dataset using Gensim.
- Sentence embeddings are calculated by averaging the word embeddings of valid words in each text.

## K-Nearest Neighbors Model

- The K-Nearest Neighbors model is trained on the calculated sentence embeddings and corresponding labels.
- The number of neighbors is set to 5.

## Testing and Evaluation

- The trained model is used to predict labels for the test data.
- The accuracy of the model is evaluated using scikit-learn's `accuracy_score` function.

## Dependencies

- Gensim
- NLTK
- NumPy
- TensorFlow
- Keras
- scikit-learn
