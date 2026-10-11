# NOTE: Adapted from the book's Example 4.24 (book page 189).
# Modified: print() added so the result is shown in the terminal (the book
#    is a notebook, which shows the last result automatically).

# Example 4.24
from transformers import pipeline

question_answerer = pipeline('question-answering')
result = question_answerer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
})
print(result)