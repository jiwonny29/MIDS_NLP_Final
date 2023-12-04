import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv("C:/Users/wonny/Downloads/nlp/balanced_data.csv")


# Data processing function
def data_processing(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = re.findall(r"\b\w+\b", text)
    return " ".join(words)


# Apply data processing to the 'review' column
data["review"] = data["review"].apply(data_processing)

# Split data into features and labels
X = data["review"]
y = data["sentiment"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english", lowercase=True, max_features=7000
)

# Transform the training and testing text data into feature vectors
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define a list of alpha values to be tested
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

# Create a parameter grid for Grid Search
param_grid = {"alpha": alphas}

# Initialize the Multinomial Naive Bayes models
clf_count = MultinomialNB()
clf_tfidf = MultinomialNB()

# Create Grid Search models with cross-validation
grid_search_count = GridSearchCV(
    clf_count, param_grid, cv=10, scoring="accuracy", n_jobs=-1
)
grid_search_tfidf = GridSearchCV(
    clf_tfidf, param_grid, cv=10, scoring="accuracy", n_jobs=-1
)

# Perform Grid Search for both models
grid_search_count.fit(X_train_counts, y_train)
grid_search_tfidf.fit(X_train_tfidf, y_train)

# Print the best hyperparameters for both models
print("Best Parameters for Count Vectorizer:", grid_search_count.best_params_)
print("Best Parameters for TF-IDF Vectorizer:", grid_search_tfidf.best_params_)

# Get the best models for both vectorizers
best_model_count = grid_search_count.best_estimator_
best_model_tfidf = grid_search_tfidf.best_estimator_

# Predict labels on the test set for both models
y_pred_count = best_model_count.predict(X_test_counts)
y_pred_tfidf = best_model_tfidf.predict(X_test_tfidf)

# Calculate and print accuracy, precision, recall, and F1-score for both models
accuracy_count = accuracy_score(y_test, y_pred_count)
precision_count = precision_score(y_test, y_pred_count, average="macro")
recall_count = recall_score(y_test, y_pred_count, average="macro")
f1_count = f1_score(y_test, y_pred_count, average="macro")

accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf = precision_score(y_test, y_pred_tfidf, average="macro")
recall_tfidf = recall_score(y_test, y_pred_tfidf, average="macro")
f1_tfidf = f1_score(y_test, y_pred_tfidf, average="macro")

print("Count Vectorizer - Accuracy:", accuracy_count)
print("Count Vectorizer - Precision:", precision_count)
print("Count Vectorizer - Recall:", recall_count)
print("Count Vectorizer - F1-score:", f1_count)

print("TF-IDF Vectorizer - Accuracy:", accuracy_tfidf)
print("TF-IDF Vectorizer - Precision:", precision_tfidf)
print("TF-IDF Vectorizer - Recall:", recall_tfidf)
print("TF-IDF Vectorizer - F1-score:", f1_tfidf)
