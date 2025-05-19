from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Only REAL news examples
texts = [
    "Indian government launches new health policy",
    "NASA discovers water on the moon",
    "Stock market hits record high in 2025",
    "COVID-19 vaccine proves effective in trials",
    "Prime Minister addresses climate change conference",
]

labels = ["REAL"] * len(texts)  # All are REAL

# Add FAKE for testing (optional for better model learning)
texts += [
    "Aliens built the Taj Mahal",
    "Drinking cola cures cancer",
]
labels += ["FAKE", "FAKE"]

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X, labels)

# Keyboard input loop
print("Enter a news headline to check if it's REAL or FAKE (type 'exit' to quit):")
while True:
    user_input = input("News: ")
    if user_input.lower() == 'exit':
        break
    vector = vectorizer.transform([user_input])
    prediction = model.predict(vector)
    print("Prediction:", prediction[0])