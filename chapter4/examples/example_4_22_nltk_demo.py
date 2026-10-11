# Example 4.22 - NLTK demo
# Text analysis with NLTK: tokenization, part of speech (POS) tagging,
# and named entity recognition (people, organizations, places).
# Install first: pip install nltk

import nltk

# Download the required resources (names differ between NLTK versions, so try them all)
for package in [
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(package, quiet=True)

text = """
Machine learning is a field of computer science in which programs improve with experience.
Researchers at Stanford University and companies such as Google use it for translation,
image recognition, and medical diagnosis.
"""

tokens = nltk.word_tokenize(text)
print("TOKENS:")
print(tokens)

tagged = nltk.pos_tag(tokens)
print("\nPOS TAGS:")
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print("\nENTITY TREE:")
print(entities)
