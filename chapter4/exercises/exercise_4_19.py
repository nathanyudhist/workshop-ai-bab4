# Exercise 4.19
# Task: Modify Example 4.23 (transformers, sentiment analysis). Choose several sentences,
# some positive and some negative, then add a comment.
# Note: the model is downloaded automatically on the first run.

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

sentences = [
    # clearly positive
    "This is a good movie.",
    "I love how easy this app is to use.",
    "The food was delicious and the staff were very friendly.",
    # clearly negative
    "This was a terrible waste of time.",
    "The battery died after one hour and support never replied.",
    "I am very disappointed with the quality.",
    # hard cases
    "The movie was not bad at all.",
    "Great, another delay. Just what I needed.",
    "The plot was boring, but the acting was wonderful.",
]

print("Label    | Score  | Sentence")
for sentence in sentences:
    result = classifier(sentence)[0]
    print(f"{result['label']:<8} | {result['score']:.4f} | {sentence}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: check whether the hard sentences (negation, sarcasm, mixed positive and negative)
# are classified correctly, and how confident the scores are.
