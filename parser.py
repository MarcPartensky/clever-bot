import re

with open('text.xml', 'r') as file:
    text = file.read()

print("texte extrait:", len(text))

def clean(text):
    pattern = re.compile('<.*?>')
    text = re.sub(pattern, '', text)
    return text

text = clean(text)

print("texte nettoyé:", len(text))

with open('output.txt', 'w') as file:
    file.write(text)

print('text écrit')
