# NOTE: Based on the book's Example 4.24 (book page 189), modified to
# answer Exercise 4.20. The book only gives the task, so this solution code
# is not from the book.
# Changes: The text of Example 4.24 is kept; four different questions are
#    asked, one of which has no answer in the text.

# Exercise 4.20
# Task: Modify Example 4.24 (transformers, question answering). Choose a different text
# and ask different questions, then add a comment.

from transformers import pipeline

question_answerer = pipeline('question-answering')

context = 'We created Biox Systems Ltd company back in the year of 2000.'

questions = [
    "When was the company created?",
    "Which company was created in the year 2000?",
    "What did we create back in the year of 2000?",
    "Where is the company located?",  # the answer is not in the text
]

for q in questions:
    result = question_answerer({'question': q, 'context': context})
    print(f"Question: {q}")
    print(f"Answer  : {result['answer']}  (score {result['score']:.3f})\n")