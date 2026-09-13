import csv
import math
import pickle
import re
from collections import Counter, defaultdict

# -------------------------------
# Text Cleaning
# -------------------------------

def clean_text(text):
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return words


# -------------------------------
# Load Dataset
# -------------------------------

texts = []
labels = []

with open("Database/sentiment.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        texts.append(row["text"])
        labels.append(row["sentiment"])


# -------------------------------
# Train Naive Bayes Model
# -------------------------------

word_counts = defaultdict(Counter)
class_counts = Counter()

for text, label in zip(texts, labels):

    words = clean_text(text)

    class_counts[label] += 1

    for word in words:
        word_counts[label][word] += 1


total_documents = len(labels)
classes = list(class_counts.keys())

# Vocabulary
vocabulary = set()

for label in classes:
    vocabulary.update(word_counts[label].keys())

vocabulary_size = len(vocabulary)


# -------------------------------
# Prediction Function
# -------------------------------

def predict(text):

    words = clean_text(text)

    scores = {}

    for label in classes:

        # Prior probability
        score = math.log(class_counts[label] / total_documents)

        total_words = sum(word_counts[label].values())

        for word in words:

            # Laplace smoothing
            word_probability = (
                word_counts[label][word] + 1
            ) / (
                total_words + vocabulary_size
            )

            score += math.log(word_probability)

        scores[label] = score

    return max(scores, key=scores.get)


# -------------------------------
# Test Model
# -------------------------------

test_sentences = [
    "I love this product",
    "This is terrible",
    "The product is average"
]

print("Model trained successfully!\n")

for sentence in test_sentences:

    result = predict(sentence)

    print("Text:", sentence)
    print("Prediction:", result)
    print()


# -------------------------------
# Save Model
# -------------------------------

model_data = {
    "word_counts": dict(word_counts),
    "class_counts": dict(class_counts),
    "classes": classes,
    "vocabulary_size": vocabulary_size
}

with open("model/sentiment_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Model saved successfully!")