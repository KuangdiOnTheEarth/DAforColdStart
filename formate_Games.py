import gzip
import json

item_file = 'data/Games/raw/meta_Video_Games.json.gz'
# g = gzip.open(item_file, 'r')
# for line in g:
#     item = json.load(line)
#     print(item)

g = gzip.open(item_file, 'r')
for line in g:
    dic = json.loads(line)
    print("%s %s" % (dic['asin'], dic['title']))