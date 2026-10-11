# Exercise 4.18
# Task: Modify the NLTK program (Example 4.22) with a different text for text analysis,
# then add a comment.

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
Sundar Pichai leads Google in California. Microsoft hired engineers in Jakarta last year,
and Apple opened a learning center in Surabaya. The new program teaches students to build
mobile apps with Swift and Flutter.
"""

tokens = nltk.word_tokenize(text)
print("TOKENS:")
print(tokens)

tagged = nltk.pos_tag(tokens)
print("\nPOS TAGS:")
print(tagged)

tree = nltk.chunk.ne_chunk(tagged)
print("\nENTITY TREE:")
print(tree)

print("\nENTITIES FOUND:")
for part in tree:
    if hasattr(part, "label"):
        entity = " ".join(word for word, _ in part.leaves())
        print(f"   {part.label():<14} {entity}")

# RESULT COMMENTS:
# (fill in after running)
# Hint: check whether names of people, companies, and cities are detected correctly,
# and whether any are mislabeled (for example, a city treated as an organization).
