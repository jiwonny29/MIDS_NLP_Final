# Natural Language Processing - Final Project
## Sentiment Analysis Models for Online Product Reviews

<div align="center">
  <img src="assets/sentiment-analysis.jpg" alt="Sentiment Analysis Models" width="500">
</div>

## Overview

This repository conducts an in-depth exploration and comparison of sentiment analysis models tailored for online product reviews. We evaluate the effectiveness of two models – a Generative Language Model (Multinomial Naive Bayes) and a Discriminative Neural Network (Bidirectional LSTM) – aiming to provide actionable insights for sentiment classification in the realm of e-commerce.

## Authors

- **Afraa Noureen**
- **Ayush Gupta**
- **Jiwon Shin**

## Abstract

Our objective is to empower businesses with a robust sentiment analysis framework by dissecting real-world and synthetic datasets. This comparison spans various metrics, emphasizing the balance between accuracy, computational efficiency, and interpretability.

## Background

In a world driven by consumer feedback, accurate sentiment analysis is paramount for businesses. Our study delves into online product reviews from Amazon and eBay, offering valuable insights to enhance customer experience and streamline analysis efforts.

## Dataset

- Real-world dataset: Curated from Kaggle, refined to 92,100 reviews.
- Synthetic dataset: Created for unbiased testing, mirroring real-world scenarios.

## Generative Language Model

### Model Selection

- **Classifier:** Multinomial Naive Bayes
- **Features:** Bag of Words (BoW) and TF-IDF

### Implementation Highlights

- Text vectorization using CountVectorizer and TF-IDF Vectorizer.
- Vocabulary limited to 7,000 words for optimal accuracy.
- Hyperparameter tuning via Grid Search for precise sentiment analysis.

### Results

- Real Data Accuracy: BoW - 65.57%, TF-IDF - 66.07%
- Synthetic Data Accuracy: BoW - 53.41%, TF-IDF - 53.9%
- Challenges highlighted through misclassification analysis.

## Discriminative Neural Network

### Model Selection

- **Architecture:** Bidirectional Long Short-Term Memory (BiLSTM)

### Implementation Highlights

- Embedding layer for word transformation.
- Bidirectional layers to capture context.
- Dropout layers for overfitting prevention.
- Efficient computational considerations for training.

### Results

- Real Data Accuracy: 70.90%
- Synthetic Data Accuracy: 98.00%
- Acknowledgment of challenges with uncommon words and ambiguous context.

## Limitations

### Generative Language Model

- Loss of contextual information.
- Struggles with negations, modifiers, and polysemy.
- Fixed vocabulary constraints.

### Discriminative Neural Network

- Computational resource demands.
- Reliance on training data quality.
- Sensitivity to noisy data.
- Hardware acceleration considerations.

## Conclusion

Our study illuminates strengths and limitations, offering businesses informed choices for sentiment analysis in the dynamic landscape of online product reviews.

<div align="center">
  <img src="assets/conclusion.jpg" alt="Conclusion" width="600">
</div>

## Recommendations

1. **Model Selection:** Align with the nature of your data and prioritize interpretability or accuracy.
2. **Hyperparameter Tuning:** Fine-tune based on dataset characteristics for optimal performance.
3. **Context-Aware Approaches:** Implement nuanced approaches for sentiment analysis, especially in complex reviews.

These insights guide businesses in deploying effective sentiment analysis models, fostering a deeper understanding of customer sentiments.
