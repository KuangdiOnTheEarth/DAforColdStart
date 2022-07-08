import os.path


def load_data(folder):
    dataset = []
    files = ['ws', 'ucs', 'ics', 'mcs']
    for file in files:
        path = os.path.join(folder, file + '.txt')
        for line in open(file):
            line_data = line.rstrip().split(' ')
            line_data = list(map(int, line_data))
            sid = line_data[0]
            items = line_data[2:]
            # remove the last two validation and testing items, use remaining items as training data
            if len(items) > 2:
                seq = items[0:-2]
            else:
                seq = []
            dataset.append(seq)
    return dataset
