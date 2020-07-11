from html.parser import HTMLParser
import sqlite3


conn = sqlite3.connect('reddit', isolation_level=None)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS reddit (
    uid integer,
    comment_id varchar(7) PRIMARY KEY,
    parent_id varchar(7),
    score integer,
    create_utc timestamp,
    content varchar(40000)
);
""")

class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comments = [] # path in tree buffer
        # self.comment = None
        self.utc = 0

    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        if tag=="utt":
            self.comments.append(dict(attrs, content=""))
            self.utc += 1

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        if tag=="utt":
            self.utc -= 1
            # print(self.comments[-1].values())
            c.execute('insert into reddit(uid, comment_id, parent_id, score, create_utc, content) values(?,?,?,?,?,?)', tuple(self.comments[-1].values()))
            self.comments = self.comments[:-1]

    def handle_data(self, data):
        # print("Encountered some data  :", data)
        if self.utc:
            self.comments[-1]['content'] += data.strip()
            

def get_xml(name):
    with open(name, 'r') as file:
        result = file.read()
    return result

if __name__ == "__main__":
    xml = get_xml('text.xml')
    parser = MyHTMLParser()
    parser.feed(xml)
    print('done')
    # print('len:', len(parser.comments))
    # print(parser.utc)
    # print(parser.comments[:3])
    # c.execute('select * from reddit')
    # print(c.fetchall())