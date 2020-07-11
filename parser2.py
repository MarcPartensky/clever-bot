import re

with open('text.xml', 'r') as file:
    text = file.read()

print("texte extrait:", len(text))

def clean(text):
    pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(pattern, '', text)
    print('symbol clean', len(text))

    pattern = re.compile('<.*>')
    text = re.sub(pattern, '', text)
    print('symbol clean 2', len(text))

    pattern = re.compile(r'(\\n)+')
    text = re.sub(pattern, '\n', text)
    print('new line clean', len(text))

    pattern = re.compile('  + ')
    text = re.sub(pattern, ' ', text)
    print('space clean', len(text))

    return text

text = clean(text)

print("texte nettoyé:", len(text))

with open('output3.txt', 'w') as file:
    file.write(text)

print('text écrit')
