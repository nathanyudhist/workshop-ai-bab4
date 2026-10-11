# Example 4.23 - Transformers demo
# Sentiment analysis with the Hugging Face Transformers library.
# Install first: pip install transformers torch
# Note: the model is downloaded automatically on the first run.

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

result = classifier("This is a good movie.")
print(result)
