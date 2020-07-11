import re
import sys
# text = sys.argv[1]
text = r'gab           sg\n\n\n\n\n\n\nas                   df'


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

print(text)
