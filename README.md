# Natural Language Processing - Final Project
# Sentiment Analysis Models for Online Product Reviews

## Overview

This repository presents a comprehensive exploration of sentiment analysis models tailored for online product reviews. We meticulously compare a Generative Language Model (Multinomial Naive Bayes) and a Discriminative Neural Network (Bidirectional LSTM) to provide practical insights for sentiment classification in the dynamic e-commerce landscape.

## Authors

- Afraa Noureen
- Ayush Gupta
- Jiwon Shin

## Abstract

Our mission is to empower businesses with a robust sentiment analysis framework. We conduct a thorough analysis of real-world and synthetic datasets, focusing on accuracy, computational efficiency, and interpretability.

## Background

Accurate sentiment analysis is essential in consumer-driven markets. This study delves into product reviews from Amazon and eBay, providing valuable insights to enhance customer experience.

## Dataset

- **Real-world dataset:** Curated from Kaggle, refined to 92,100 reviews.
- **Synthetic dataset:** Created for unbiased testing.

## Generative Language Model

### Model Selection

- **Classifier:** Multinomial Naive Bayes
- **Features:** Bag of Words (BoW) and TF-IDF

### Implementation Highlights

- Text vectorization using CountVectorizer and TF-IDF Vectorizer.
- Vocabulary limited to 7,000 words.
- Hyperparameter tuning via Grid Search.

### Results

- Real Data Accuracy: BoW - 65.57%, TF-IDF - 66.07%
- Synthetic Data Accuracy: BoW - 53.41%, TF-IDF - 53.9%
- Challenges highlighted through misclassification analysis.

## Discriminative Neural Network

### Model Selection

- **Architecture:** Bidirectional LSTM

### Implementation Highlights

- Embedding layer for word transformation.
- Bidirectional layers for context.
- Dropout layers for overfitting prevention.

### Results

- Real Data Accuracy: 70.90%
- Synthetic Data Accuracy: 98.00%
- Acknowledgment of challenges with uncommon words and ambiguous context.

## Conclusion

**Generative Language Model (Multinomial Naive Bayes):**
- Strengths: Interpretability, simplicity.
- Weaknesses: Loss of contextual information, struggles with negations and modifiers, fixed vocabulary.

**Discriminative Neural Network (Bidirectional LSTM):**
- Strengths: High accuracy, adaptability to complex structures.
- Weaknesses: Computational demands, challenges with uncommon words.

These findings guide businesses in choosing models aligned with their priorities.

## Recommendations

1. **Model Selection:** Align with your data and prioritize interpretability or accuracy.
2. **Hyperparameter Tuning:** Fine-tune based on dataset characteristics.
3. **Context-Aware Approaches:** Implement nuanced approaches, especially in complex reviews.

These insights assist businesses in deploying effective sentiment analysis models, fostering a deeper understanding of customer sentiments.
