import random
import pandas as pd
import nltk

nltk.download("wordnet")

from nltk.corpus import wordnet

# Sample positive, negative, and neutral words
positive_words = ["good", "great", "awesome", "excellent", "fantastic", "superb"]
negative_words = ["bad", "poor", "awful", "terrible", "horrible", "disappointing"]
neutral_words = ["okay", "average", "normal", "mediocre", "unsure", "indifferent"]

# Common sentence structures
sentence_structures = [
    "The product is {0}.",
    "I felt it was {0}.",
    "This service is {0}.",
    "Overall, the experience was {0}.",
    "It's {0}.",
    "I found it {0}.",
    "The {0} quality is decent.",
    "Not really {0}, but acceptable.",
    "Could be more {0}, to be honest.",
    "I'm feeling {0} about it.",
    "Can't complain, it's {0}.",
]

synth_data = pd.DataFrame(columns=["text", "label"])

# Create synthetic data with specified sentiments
for sentiment in [-1, 0, 1]:  # Sentiments: -1 (negative), 0 (neutral), 1 (positive)
    synth_sent = []
    if sentiment == 0:  # Neutral sentiment
        words = neutral_words
    elif sentiment == 1:  # Positive sentiment
        words = positive_words
    else:  # Negative sentiment
        words = negative_words

    for n in range(50000 // 3):  # Divide by number of sentiments
        rand_word = random.choice(words)
        syns = wordnet.synsets(rand_word)
        if syns:
            synonyms = [lemma.name() for syn in syns for lemma in syn.lemmas()]
            rand_word = random.choice(synonyms)
        rand_structure = random.choice(sentence_structures)
        review = rand_structure.format(rand_word)
        synth_sent.append(review)

    df = pd.DataFrame({"text": synth_sent, "label": sentiment})
    synth_data = pd.concat([synth_data, df])

# Save synthetic data to CSV
synth_data.to_csv(
    "C:\\Users\\ayush\\OneDrive\\Desktop\\NLP\\Final Project\\synthetic_data_nltk.csv",
    index=False,
)
