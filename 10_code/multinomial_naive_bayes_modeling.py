import pandas as pd
import re
import time
import psutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv("C:/Users/wonny/Downloads/nlp/balanced_data.csv")
data.info()


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
count_vectorizer = CountVectorizer(
    stop_words="english", lowercase=True, max_features=7000
)
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english", lowercase=True, max_features=7000
)

# Measure the start time and memory for BoW (CountVectorizer)
start_time_bow = time.time()
start_memory_bow = psutil.Process().memory_info().rss

# Transform the training and testing text data into BoW feature vectors
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Initialize the Multinomial Naive Bayes model for BoW
clf_count = MultinomialNB()

# Create Grid Search models with cross-validation for BoW
param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search_count = GridSearchCV(
    clf_count, param_grid, cv=10, scoring="accuracy", n_jobs=-1
)

# Perform Grid Search for BoW
grid_search_count.fit(X_train_counts, y_train)

# Measure the end time and memory for BoW (CountVectorizer)
end_time_bow = time.time()
end_memory_bow = psutil.Process().memory_info().rss

# Calculate the total time taken for BoW
total_time_bow = end_time_bow - start_time_bow

# Predict labels on the test set for BoW
y_pred_count = grid_search_count.best_estimator_.predict(X_test_counts)

# Calculate accuracy, precision, recall, and F1-score for BoW
accuracy_count = accuracy_score(y_test, y_pred_count)
precision_count = precision_score(y_test, y_pred_count, average="macro")
recall_count = recall_score(y_test, y_pred_count, average="macro")
f1_count = f1_score(y_test, y_pred_count, average="macro")

# Print evaluation metrics for BoW
print("Count Vectorizer - Accuracy:", accuracy_count)
print("Count Vectorizer - Precision:", precision_count)
print("Count Vectorizer - Recall:", recall_count)
print("Count Vectorizer - F1-score:", f1_count)

# Print the total execution time for BoW
print(
    "Total execution time for BoW (CountVectorizer): {:.2f} seconds".format(
        total_time_bow
    )
)

# Print memory usage for BoW (CountVectorizer) on training data
print(
    "Memory Usage for BoW (CountVectorizer) on training data: {:.2f} KB".format(
        start_memory_bow / 1024
    )
)
# Print memory usage for BoW (CountVectorizer) on new data
print(
    "Memory Usage for BoW (CountVectorizer) on new data: {:.2f} KB".format(
        end_memory_bow / 1024
    )
)

# Measure the start time and memory for TF-IDF (TfidfVectorizer)
start_time_tfidf = time.time()
start_memory_tfidf = psutil.Process().memory_info().rss

# Transform the training and testing text data into TF-IDF feature vectors
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the Multinomial Naive Bayes model for TF-IDF
clf_tfidf = MultinomialNB()

# Create Grid Search models with cross-validation for TF-IDF
grid_search_tfidf = GridSearchCV(
    clf_tfidf, param_grid, cv=10, scoring="accuracy", n_jobs=-1
)

# Perform Grid Search for TF-IDF
grid_search_tfidf.fit(X_train_tfidf, y_train)

# Measure the end time and memory for TF-IDF (TfidfVectorizer)
end_time_tfidf = time.time()
end_memory_tfidf = psutil.Process().memory_info().rss

# Calculate the total time taken for TF-IDF
total_time_tfidf = end_time_tfidf - start_time_tfidf

# Predict labels on the test set for TF-IDF
y_pred_tfidf = grid_search_tfidf.best_estimator_.predict(X_test_tfidf)

# Calculate accuracy, precision, recall, and F1-score for TF-IDF
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf = precision_score(y_test, y_pred_tfidf, average="macro")
recall_tfidf = recall_score(y_test, y_pred_tfidf, average="macro")
f1_tfidf = f1_score(y_test, y_pred_tfidf, average="macro")

# Print evaluation metrics for TF-IDF
print("TF-IDF Vectorizer - Accuracy:", accuracy_tfidf)
print("TF-IDF Vectorizer - Precision:", precision_tfidf)
print("TF-IDF Vectorizer - Recall:", recall_tfidf)
print("TF-IDF Vectorizer - F1-score:", f1_tfidf)

# Print the total execution time for TF-IDF
print(
    "Total execution time for TF-IDF (TfidfVectorizer): {:.2f} seconds".format(
        total_time_tfidf
    )
)

# Print memory usage for TF-IDF (TfidfVectorizer) on training data
print(
    "Memory Usage for TF-IDF (TfidfVectorizer) on training data: {:.2f} KB".format(
        start_memory_tfidf / 1024
    )
)
# Print memory usage for TF-IDF (TfidfVectorizer) on new data
print(
    "Memory Usage for TF-IDF (TfidfVectorizer) on new data: {:.2f} KB".format(
        end_memory_tfidf / 1024
    )
)

# Load the new data (Replace 'path_to_new_data.csv' with your new data file path)
new_data = pd.read_csv("C:/Users/wonny/Downloads/nlp/synthetic_data_nltk.csv")
new_data.info()

# Apply the same data processing to the 'text' column
new_data["text"] = new_data["text"].apply(data_processing)

# Split the new data into features and labels
X_new = new_data["text"]
y_new = new_data["label"]

# Transform the new data into feature vectors using the same vectorizer used for training
X_new_counts = count_vectorizer.transform(X_new)  # For CountVectorizer
X_new_tfidf = tfidf_vectorizer.transform(X_new)  # For TF-IDF Vectorizer

# Use the trained models to make predictions on the new data
y_pred_count_new = grid_search_count.best_estimator_.predict(X_new_counts)
y_pred_tfidf_new = grid_search_tfidf.best_estimator_.predict(X_new_tfidf)

# Calculate accuracy, precision, recall, and F1-score for the new data
accuracy_count_new = accuracy_score(y_new, y_pred_count_new)
precision_count_new = precision_score(y_new, y_pred_count_new, average="macro")
recall_count_new = recall_score(y_new, y_pred_count_new, average="macro")
f1_count_new = f1_score(y_new, y_pred_count_new, average="macro")

accuracy_tfidf_new = accuracy_score(y_new, y_pred_tfidf_new)
precision_tfidf_new = precision_score(y_new, y_pred_tfidf_new, average="macro")
recall_tfidf_new = recall_score(y_new, y_pred_tfidf_new, average="macro")
f1_tfidf_new = f1_score(y_new, y_pred_tfidf_new, average="macro")

# Print evaluation metrics for the new data
print("Count Vectorizer (New Data) - Accuracy:", accuracy_count_new)
print("Count Vectorizer (New Data) - Precision:", precision_count_new)
print("Count Vectorizer (New Data) - Recall:", recall_count_new)
print("Count Vectorizer (New Data) - F1-score:", f1_count_new)

print("TF-IDF Vectorizer (New Data) - Accuracy:", accuracy_tfidf_new)
print("TF-IDF Vectorizer (New Data) - Precision:", precision_tfidf_new)
print("TF-IDF Vectorizer (New Data) - Recall:", recall_tfidf_new)
print("TF-IDF Vectorizer (New Data) - F1-score:", f1_tfidf_new)

# ...

# Measure the start time and memory for Count Vectorizer (New Data)
start_time_count_new = time.time()
start_memory_count_new = psutil.Process().memory_info().rss

# Transform the new data into feature vectors using the CountVectorizer
X_new_counts = count_vectorizer.transform(X_new)  # For CountVectorizer

# Use the trained CountVectorizer model to make predictions on the new data
y_pred_count_new = grid_search_count.best_estimator_.predict(X_new_counts)

# Measure the end time and memory for Count Vectorizer (New Data)
end_time_count_new = time.time()
end_memory_count_new = psutil.Process().memory_info().rss

# Calculate the total time taken for Count Vectorizer (New Data)
total_time_count_new = end_time_count_new - start_time_count_new

# Print evaluation metrics for Count Vectorizer (New Data)
print("Count Vectorizer (New Data) - Accuracy:", accuracy_count_new)
print("Count Vectorizer (New Data) - Precision:", precision_count_new)
print("Count Vectorizer (New Data) - Recall:", recall_count_new)
print("Count Vectorizer (New Data) - F1-score:", f1_count_new)

# Print the total execution time for Count Vectorizer (New Data)
print(
    "Total execution time for Count Vectorizer (New Data): {:.2f} seconds".format(
        total_time_count_new
    )
)

# Print memory usage for Count Vectorizer (New Data) on new data
print(
    "Memory Usage for Count Vectorizer (New Data) on new data: {:.2f} KB".format(
        end_memory_count_new / 1024
    )
)

# Measure the start time and memory for TF-IDF Vectorizer (New Data)
start_time_tfidf_new = time.time()
start_memory_tfidf_new = psutil.Process().memory_info().rss

# Transform the new data into feature vectors using the TF-IDF Vectorizer
X_new_tfidf = tfidf_vectorizer.transform(X_new)  # For TF-IDF Vectorizer

# Use the trained TF-IDF Vectorizer model to make predictions on the new data
y_pred_tfidf_new = grid_search_tfidf.best_estimator_.predict(X_new_tfidf)

# Measure the end time and memory for TF-IDF Vectorizer (New Data)
end_time_tfidf_new = time.time()
end_memory_tfidf_new = psutil.Process().memory_info().rss

# Calculate the total time taken for TF-IDF Vectorizer (New Data)
total_time_tfidf_new = end_time_tfidf_new - start_time_tfidf_new

# Print evaluation metrics for TF-IDF Vectorizer (New Data)
print("TF-IDF Vectorizer (New Data) - Accuracy:", accuracy_tfidf_new)
print("TF-IDF Vectorizer (New Data) - Precision:", precision_tfidf_new)
print("TF-IDF Vectorizer (New Data) - Recall:", recall_tfidf_new)
print("TF-IDF Vectorizer (New Data) - F1-score:", f1_tfidf_new)

# Print the total execution time for TF-IDF Vectorizer (New Data)
print(
    "Total execution time for TF-IDF Vectorizer (New Data): {:.2f} seconds".format(
        total_time_tfidf_new
    )
)

# Print memory usage for TF-IDF Vectorizer (New Data) on new data
print(
    "Memory Usage for TF-IDF Vectorizer (New Data) on new data: {:.2f} KB".format(
        end_memory_tfidf_new / 1024
    )
)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot two confusion matrices side by side
def plot_confusion_matrices(cm1, cm2, title1, title2, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Set up a 1x2 grid of subplots
    
    # Plot first confusion matrix
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title(title1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    
    # Plot second confusion matrix
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title(title2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    
    plt.tight_layout()  # Adjust the layout
    plt.show()

# Labels for the axes
class_labels = [-1, 0, 1]

# Calculate confusion matrices for Count Vectorizer and TF-IDF Vectorizer
confusion_matrix_count = confusion_matrix(y_new, y_pred_count_new)
confusion_matrix_tfidf = confusion_matrix(y_new, y_pred_tfidf_new)

# Plot the confusion matrices side by side
plot_confusion_matrices(confusion_matrix_count, confusion_matrix_tfidf, 
                        "Confusion Matrix (Bag-of-Words Model)", "Confusion Matrix (TF-IDF Model)", 
                        class_labels)
