import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
dataset = pd.read_csv(
    "C:\\Users\\ayush\\OneDrive\\Desktop\\NLP\\Final Project\\balanced_data.csv"
)

# TF-IDF vectorization
tfidf = TfidfVectorizer()
text_count_2 = tfidf.fit_transform(dataset["review"])
vocabulary = tfidf.get_feature_names_out()

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    text_count_2, dataset["sentiment"], test_size=0.25, random_state=5
)

# Multinomial Naive Bayes model
MNB = MultinomialNB()
MNB.fit(x_train, y_train)

# Setting number of entries in synthetic data
synth_data_len = 50000

# Setting the probabilities to use when creating synthetic reviews
prior_prob = np.exp(MNB.feature_log_prob_)

# 0 is negative, 1 is positive
labels = [-1, 0, 1]

# Create synthetic data
synth_data = pd.DataFrame(columns=["review", "sentiment"])
for lab_val in labels:
    synth_sent = []
    for n in range(synth_data_len // 2):
        rand_sentence = random.choices(
            vocabulary, prior_prob[lab_val], k=random.randint(5, 500)
        )
        synth_sent.append(" ".join(rand_sentence))
    df = pd.DataFrame({"review": synth_sent, "sentiment": lab_val})
    synth_data = pd.concat([synth_data, df])

# Save synthetic data to CSV
synth_data.to_csv(
    "C:\\Users\\ayush\\OneDrive\\Desktop\\NLP\\Final Project\\synthetic_data.csv",
    index=False,
)
