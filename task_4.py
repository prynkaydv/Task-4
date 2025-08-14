# Spam Detection using Scikit-learn

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import seaborn as sns
except ImportError:
    print("Seaborn not found. Skipping heatmap visualization.")
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Create and Save Sample Dataset (if not exists)
sample_data = {
    "v1": ["ham", "spam", "ham", "spam", "ham"],
    "v2": [
        "Hi, how are you?",
        "Congratulations! You've won a free ticket to Bahamas. Text WIN to 12345",
        "Are we still meeting today?",
        "URGENT! You have won a 1 week FREE membership. Call now!",
        "Let's grab lunch tomorrow."
    ]
}
sample_df = pd.DataFrame(sample_data)
sample_path = 'spam.csv'
if not os.path.exists(sample_path):
    sample_df.to_csv(sample_path, index=False, encoding='latin-1')

# Step 3: Load Dataset
file_path = sample_path

# Load and preprocess the dataset
df = pd.read_csv(file_path, encoding='latin-1')
if 'v1' not in df.columns or 'v2' not in df.columns:
    raise ValueError("Expected columns 'v1' and 'v2' not found in the dataset.")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 4: Preprocess Data
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Step 5: Split Data
X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Convert Text to Vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 7: Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test_counts)

# Step 9: Evaluate the Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Step 10: Visualize Confusion Matrix
if sns:
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
