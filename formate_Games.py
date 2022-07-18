import gzip
import json

item_file = 'data/Games/raw/meta_Video_Games.json.gz'

g = gzip.open(item_file, 'r')
count = 0
for line in g:
    dic = json.loads(line)
    count += 1
    print("%s %s" % (dic['asin'], dic['title']))
print(count)