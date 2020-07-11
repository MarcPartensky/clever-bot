from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
import pickle

print('starting now')

with open('text.xml', 'r') as file:
    text = file.read()
print('done reading')

data = fromstring(text)
print('done getting xml object')

data = bf.data(data)
print('done using badger fish')

data = pickle.dumps(data)
print('done dumping to pickle object')

pickle.dump(data, 'data.pickle')
print('done dumping to pickle file')

print('done')
