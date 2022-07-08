import os
import codecs

dataset_dir = os.path.join('data', 'ml-1m', 'ml-1m_raw')
output_dir = os.path.join('data', 'ml-1m', 'formatted')

"""
Output files:
- formatted/genres: genre_id :: genre_name
- formatted/movie_newID_rawID: formatted_movie_id :: original_movie_id_in_raw_dataset
- formatted/movies: formatted_movie_id :: title :: year :: sequence_of_genre_ids
- formatted/users: uid :: sequence_of_movies
"""

Movies = {}  # Movies[mid] = ['title', 'year', 'generes id split by ,']
Users = {}
Movies_Output = []  # raw movie id -> formatted movie id (only record the reviewed movies)
Genres = []

with codecs.open(os.path.join(dataset_dir, 'movies.dat'), 'r', encoding='iso-8859-15') as f:
    print("Reading movies ...")
    for line in f:
        line_seg = line.rstrip().split('::')
        mid = int(line_seg[0])
        Movies[mid] = []
        if line_seg[1][-1] == ')':
            title = line_seg[1][0:-7]
            year = line_seg[1][-5:-1]
            Movies[mid].append(title)
            Movies[mid].append(year)
        else:
            Movies[mid].append(line_seg[1])
            Movies[mid].append('0')

        gen_str = ''
        for gen in line_seg[2].split('|'):
            if gen not in Genres:
                Genres.append(gen)
            if gen_str == '':
                gen_str += str(Genres.index(gen) + 1)
            else:
                gen_str += (',' + str(Genres.index(gen) + 1))
        Movies[mid].append(gen_str)
        # print(line_seg)
        # print(Movies[mid])

f = open(os.path.join(output_dir, 'genres.txt'), 'w')
for i in range(len(Genres)):
    f.write('%d %s\n' % (i+1, Genres[i]))
f.close()

with codecs.open(os.path.join(dataset_dir, 'ratings.dat'), 'r', encoding='iso-8859-15') as f:
    print("Reading ratings ...")
    for line in f:
        line_seg = line.rstrip().split('::')
        uid = int(line_seg[0])
        mid = int(line_seg[1])
        t = int(line_seg[3])
        if uid not in Users:
            Users[uid] = []
        if mid not in Movies_Output:  # re-index the movies in the order of being reviewed by users
            Movies_Output.append(mid)
        nid = Movies_Output.index(mid) + 1
        Users[uid].append((nid, t))
        if uid == 98 or uid == 4758:
            print(line_seg)
            print(Movies[mid])
            print(Users[uid][-1])

print("Outputting movies ...")
f_map = open(os.path.join(output_dir, 'movie_newID_rawID.txt'), 'w')
f_movie = codecs.open(os.path.join(output_dir, 'movies.txt'), 'w', encoding='iso-8859-15')
for i in range(len(Movies_Output)):
    raw_id = Movies_Output[i]
    new_id = i + 1
    f_map.write('%d %d\n' % (new_id, raw_id))
    # Movies[mid] = ['title', 'year', 'generes id split by ,']
    # formatted/movies: formatted_movie_id :: title :: year :: sequence_of_genre_ids
    f_movie.write('%d::%s::%s::%s\n' % (new_id, Movies[raw_id][0], Movies[raw_id][1], Movies[raw_id][2]))
f_movie.close()
f_map.close()

print("Sorting and outputting user rating sequences ...")
f = open(os.path.join(output_dir, 'sequences.txt'), 'w')
for uid, pair_list in Users.items():
    sequence = ''
    # sort by timestamp
    pair_list.sort(key=lambda x: x[1])
    # print(pair_list)
    for p in pair_list:
        if sequence == '':
            sequence += str(p[0])
        else:
            sequence += (' ' + str(p[0]))
    f.write('%d %s\n' % (uid, sequence))
    if uid == 98 or uid == 4758:
        print(sequence)
f.close()