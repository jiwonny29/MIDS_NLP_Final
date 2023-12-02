######## Use CountVectorizer for text processing ########
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv("C:/Users/wonny/Downloads/nlp/1_data_cleaning/data.csv")

# Split data into features and labels
X = df["review"]
y = df["sentiment"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initalize the CountVectorizer
vectorizer = CountVectorizer()

# Transform the training text data into feature vectors
X_train_counts = vectorizer.fit_transform(X_train)

# Define a list of alpha values to be tested
alphas = [0, 0.1, 0.5]

# Iterate over alpha values and train Multinomial Naive Bayes models
for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_counts, y_train)

    # Transform the testing text data into feature vecotrs
    X_test_counts = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_counts)

    # Calculate and print accuracy scores for different alpha values
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Alpha={alpha}, Accuracy: {accuracy}")

## Results:
## Alpha=0, Accuracy: 0.7431666666666666
## Alpha=0.1, Accuracy: 0.7316944444444444
## Alpha=0.5, Accuracy: 0.7231111111111111


######## Use a different text preprocessing method, NLTK library ########
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords from NLTK library
import nltk

nltk.download("stopwords")
nltk.download("punkt")

# Read the dataset and sample 50,000 data points (because dataset is too big over 300K cases)
data = pd.read_csv("C:/Users/wonny/Downloads/nlp/1_data_cleaning/data.csv")
df = data.sample(50000)

# Split data into features and labels
X = df["review"]
y = df["sentiment"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define a function for text preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Convert all characters to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)


# Apply text preprocessing to training and testing data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Transform the training text data into feature vectors
X_train_counts = vectorizer.fit_transform(X_train)

# Define a list of alpha values to be tested
alphas = [0, 0.1, 0.5]

# Iterate over alpha values and train Multinomial Naive Bayes models
for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_counts, y_train)

    X_test_counts = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_counts)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Alpha={alpha}, Accuracy: {accuracy}")

## Results
## Alpha=0, Accuracy: 0.6542
## Alpha=0.1, Accuracy: 0.6845
## Alpha=0.5, Accuracy: 0.6895
