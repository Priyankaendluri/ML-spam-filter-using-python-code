import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample spam/ham data
data = {
    'text': [
        'Win money now!!!',
        'Hi there, how are you?',
        'Limited offer, click here!',
        'Are we still meeting today?',
        'Earn $1000 fast',
        'Reminder: your appointment is tomorrow',
        'Congratulations! You won a prize',
        'Lunch at 1pm?',
        'Claim your reward now',
        'Hello, just checking in'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train classifier
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vectors)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Function to classify new messages
def classify_message(msg):
    vector = vectorizer.transform([msg])
    prediction = model.predict(vector)[0]
    return "SPAM" if prediction == 1 else "HAM"

# Test
while True:
    msg = input("\nEnter a message to classify (or type 'exit'): ")
    if msg.

