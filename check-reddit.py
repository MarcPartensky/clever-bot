import sqlite3

conn = sqlite3.connect('reddit')
c = conn.cursor()

c.execute('select * from reddit')
results = c.fetchmany(10)

for result in results:
  print(result[2])
  print(result[5])