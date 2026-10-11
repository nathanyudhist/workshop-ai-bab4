import tokenize
import os

files = []
for root, dirs, filenames in os.walk('.'):
    for f in filenames:
        if f.endswith('.py'):
            files.append(os.path.join(root, f))

strings_and_comments = set()
identifiers = set()

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        try:
            tokens = tokenize.generate_tokens(file.readline)
            for tok_type, tok_string, (srow, scol), (erow, ecol), line in tokens:
                if tok_type == tokenize.COMMENT:
                    strings_and_comments.add(tok_string)
                elif tok_type == tokenize.STRING:
                    strings_and_comments.add(tok_string)
                elif tok_type == tokenize.NAME:
                    identifiers.add(tok_string)
        except Exception as e:
            pass

with open('extracted_text.txt', 'w', encoding='utf-8') as out:
    out.write("--- COMMENTS & STRINGS ---\n")
    for s in strings_and_comments:
        out.write(s + "\n")
    out.write("--- IDENTIFIERS ---\n")
    for s in identifiers:
        out.write(s + "\n")
